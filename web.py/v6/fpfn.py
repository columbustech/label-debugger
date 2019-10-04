'''
Created on Mar 5, 2019

@author: hzhang0418
'''
import numpy as np
from operator import itemgetter

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from v6.detector import Detector

class FPFN(Detector):
    
    def __init__(self, features, labels, params):
        super(FPFN, self).__init__(features, labels, params)
        
        self.num_cores = params.get('num_cores', 1)
        self.nfolds = params.get('num_folds', 5)
        self.incon_indices = {} # map index to prediction probability 
        self.num_iter = 0
        
        self.clf = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=self.num_cores)
    
    
    def detect_and_rank(self):
        self.get_inconsistency_indices()
        ranked = self.rank_inconsistency_indices()
        print("Number of suspicious examples after ranking:", len(ranked))
        self.num_iter += 1
        return [ t[0] for t in ranked ]
    
    
    def use_feedback(self, index2correct_label):
        for k,v in index2correct_label.items():
            self.labels[k] = v
            
        self.incon_indices.clear()
    
    
    def get_inconsistency_indices(self):
        # cross validation
        mismatching = self.cross_validation(self.clf, self.features, self.labels, self.nfolds)
        print("Number of suspicious examples after CV:", len(mismatching))
        
        # samples with matching labels as train set
        # samples with mismatching labels as test set
        test_index = list(mismatching.keys())
        train_index = [ i for i in range(len(self.features)) if i not in mismatching ]
        
        train_set_features, test_set_features = self.features[train_index], self.features[test_index]
        train_set_labels, test_set_labels = self.labels[train_index], self.labels[test_index]
        
        # predict again
        proba = self.train_and_test(self.clf, train_set_features, train_set_labels, test_set_features)
    
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
    
    
    def cross_validation(self, classifier, features, labels, nfolds):
        
        kf = KFold(nfolds, shuffle=True, random_state = 0)
        
        mismatching = {}
        
        for train_index, test_index in kf.split(features):
            train_set_features, test_set_features = features[train_index], features[test_index]
            train_set_labels, test_set_labels = labels[train_index], labels[test_index]
            
            proba = self.train_and_test(self.clf, train_set_features, train_set_labels, test_set_features)
            
            tmp = self.find_mismatching(proba, test_set_labels)
            
            for k, v in tmp.items():
                mismatching[ test_index[k] ] = v
        
        return mismatching
    
    
    def train_and_test(self, classifier, train_set_features, train_set_labels, test_set_features):
        # train
        classifier.fit(train_set_features, train_set_labels)
        # predict
        return classifier.predict_proba(test_set_features)
      
      
    def find_mismatching(self, proba, labels):
        # find predicted labels
        predicted = np.argmax(proba, axis=1)
        # find those indices whose predicted labels differ from given labels
        diff = np.where(predicted!=labels)[0]
        
        index2proba = {} # index to probability from classifier
        for index in diff:
            index2proba[index] = proba[index]
            
        return index2proba
    
