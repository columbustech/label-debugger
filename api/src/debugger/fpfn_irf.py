'''
Created on Mar 7, 2019

@author: hzhang0418
'''

import numpy as np
from operator import itemgetter

import pymp

from sklearn.model_selection import KFold
import v3.incremental_rf
from v3.incremental_rf import IncrementalRF

from v6.detector import Detector

prefix = r'/scratch/hzhang0418/tmp/'

class FPFN_IRF(Detector):
    
    def __init__(self, features, labels, params):
        super(FPFN_IRF, self).__init__(features, labels, params)
        
        self.num_cores = params.get('num_cores', 1)
        self.nfolds = params.get('num_folds', 5)
        self.incon_indices = {} # map index to prediction probability 
        self.num_iter = 0
        
        self.fold2irf = {}
        self.last_irf = None
        
        self.indices2folds = {}
    
        
    def detect_and_rank(self):
        if self.num_iter==0:
            self.get_inconsistency_indices()
            
        ranked = self.rank_inconsistency_indices()
        
        print("Number of suspicious examples after ranking:", len(ranked))
        
        self.num_iter += 1
        
        return [ t[0] for t in ranked ]
    
    def use_feedback(self, index2correct_label):
        self.incon_indices.clear()
        
        fold_indices = {}
        for i in range(self.nfolds):
            fold_indices[i] = []
        
        for k,v in index2correct_label.items():
            self.labels[k] = v
            fold_indices[ self.indices2folds[k] ].append(k)
            
        # update each irf
        mismatching = {}
        
        for i in range(self.nfolds):
            clf, test_index, test_set_features = self.fold2irf[i]
            indices = fold_indices[i]
            clf.remove(indices) # remove from training due to label changes
            new_sample_features, new_sample_labels = self.features[indices], self.labels[indices]
            clf.add(indices, new_sample_features, new_sample_labels) # add to training
            
            proba = clf.predict(test_set_features)
            
            test_set_labels = self.labels[test_index]
            tmp = self.find_mismatching(proba, test_set_labels)
            
            for k, v in tmp.items():
                mismatching[ test_index[k] ] = v
        
        # update last irf
        test_index = list(mismatching.keys())
        # remove samples that was in training but now in testing
        last_test_index_set = set(self.last_test_index)
        indices = [ k for k in test_index if k not in last_test_index_set ]
        self.last_irf.remove(indices)
        print("Removed: ", len(indices))
        # add samples that was in testing but now in training
        test_index_set = set(test_index)
        indices = [ k for k in self.last_test_index if k not in test_index_set ]
        new_sample_features = self.features[indices]
        new_sample_labels = self.labels[indices]
        self.last_irf.add(indices, new_sample_features, new_sample_labels)
        print("Added: ", len(indices))
        
        test_set_features = self.features[test_index]
        test_set_labels = self.labels[test_index] 
        
        proba = self.last_irf.predict(test_set_features)
        tmp = self.find_mismatching(proba, test_set_labels)
        
        for k, v in tmp.items():
            self.incon_indices[ test_index[k] ] = v
        
    def get_inconsistency_indices(self):
        
        # cross validation
        mismatching = self.cross_validation(self.features, self.labels, self.nfolds)
        print("Number of suspicious examples after CV:", len(mismatching))
        
        # samples with matching labels as train set
        # samples with mismatching labels as test set
        test_index = list(mismatching.keys())
        train_index = [ i for i in range(len(self.features)) if i not in mismatching ]
        
        train_set_features, test_set_features = self.features[train_index], self.features[test_index]
        train_set_labels, test_set_labels = self.labels[train_index], self.labels[test_index]
        
        # predict again
        self.last_irf = IncrementalRF(20, train_index, self.features, self.labels)
        self.last_test_index = test_index
        proba = self.last_irf.predict(test_set_features)
    
        # find samples with mismatching labels in test set
        tmp = self.find_mismatching(proba, test_set_labels)
        
        for k, v in tmp.items():
            self.incon_indices[ test_index[k] ] = v
    
    def rank_inconsistency_indices(self):
        # Note that we use negative probability
        incons_prob = [ (k, -np.max(v)) for k,v in self.incon_indices.items() ]
        # sort in ascending order of negative probability
        # if same probability, sort in ascending order of index
        incons_prob = sorted(incons_prob, key=itemgetter(1,0))
        return incons_prob
    
    
    def cross_validation(self, features, labels, nfolds):
        
        kf = KFold(nfolds, shuffle=True, random_state = 0)
        
        mismatching = {}
        
        if self.num_cores==1:
            rf_index = 0
            for train_index, test_index in kf.split(features):
                
                train_set_features, test_set_features = features[train_index], features[test_index]
                train_set_labels, test_set_labels = labels[train_index], labels[test_index]
                
                clf = IncrementalRF(20, train_index, features, labels)
                
                proba = clf.predict(test_set_features)
                
                tmp = self.find_mismatching(proba, test_set_labels)
                
                for k, v in tmp.items():
                    mismatching[ test_index[k] ] = v
                    
                for index in train_index:
                    self.indices2folds[index] = rf_index
                    
                self.fold2irf[rf_index] = (clf, test_index, test_set_features)
                
                rf_index += 1
    
        else:
            
            fold2train = {}
            fold2test = {}
            
            fold = 0
            for train_index, test_index in kf.split(features):
                fold2train[fold] = train_index
                fold2test[fold] = test_index
                
                for index in train_index:
                    self.indices2folds[index] = fold
                
                fold += 1
            
            # parallel computing, need shared data structure
            shared_fold2irf = pymp.shared.dict()
            shared_fold2mismatching = pymp.shared.dict()
            
            with pymp.Parallel(self.num_cores) as p:
                local_mismatching = {}
                for index in p.range(self.nfolds):
                    train_index = fold2train[index]
                    test_index = fold2test[index]
                    
                    train_set_features, test_set_features = features[train_index], features[test_index]
                    train_set_labels, test_set_labels = labels[train_index], labels[test_index]
                    
                    clf = IncrementalRF(20, train_index, features, labels)
                    
                    proba = clf.predict(test_set_features)
                    
                    tmp = self.find_mismatching(proba, test_set_labels)
                    
                    for k, v in tmp.items():
                        local_mismatching[ test_index[k] ] = v
                    
                    shared_fold2mismatching[index] = local_mismatching
                    shared_fold2irf[index] = (None, test_index, test_set_features)
                    # save tree to file
                    output_file = prefix + 'tmp_'+str(index)+'.irf'
                    clf.save_to_file(output_file)
                        
            for k,v in shared_fold2irf.items():
                input_file = prefix + 'tmp_'+str(k)+'.irf'
                clf = v3.incremental_rf.load_from_file(input_file)
                self.fold2irf[k] = (clf, v[1], v[2])
            
            for _,tmp in shared_fold2mismatching.items():
                for k,v in tmp.items():
                    mismatching[k] = v
        
        return mismatching
    
    
    def find_mismatching(self, proba, labels):
        # find predicted class
        predicted = [ int(round(p)) for p in proba ]
        # find those indices whose predicted labels differ from given labels
        diff = np.where(predicted!=labels)[0]
        
        index2proba = {} # index to probability from classifier
        for index in diff:
            # note that in IRF, close to 0 or 1 means more confident
            index2proba[index] = 2*abs(proba[index]-0.5) 
            
        return index2proba
    
    