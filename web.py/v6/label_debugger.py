'''
Created on Mar 6, 2019

@author: hzhang0418
'''

import time

import v6.feature_selection as fs

import v6.fpfn as fpfn

import v6.combiner

class LabelDebugger(object):
    
    def __init__(self, features, labels, params):
        self.labels = labels
        self.features = fs.select_features(features, labels, params.get('fs_alg', 'none'))
        self.max_list_len = params.get('max_list_len', 500)
        self._start_detectors(params)
        
        self.iter_count = 0
        # indices whose label has been verified by the analyst
        self.verified_indices = set()

    def _start_detectors(self, params):
        self.detectors = []
        detector_types = params.get('detectors', 'fpfn')
        
        if detector_types=='fpfn':
            det = fpfn.FPFN(self.features, self.labels, params)
            det.set_num_cores(1)
            self.detectors.append(det)
            self.ndetectors = 1
    
    
    def find_suspicious_labels(self, top_k):
        self.iter_count += 1
        
        self.ranked_lists = []
        for det in self.detectors:
            start = time.clock()
            tmp = det.detect_and_rank()
            end = time.clock()
            #if self.iter_count==1:
            #    print('First iteration time:', end-start)
            self.ranked_lists.append([t for t in tmp if t not in self.verified_indices][:self.max_list_len] )
        
        if self.ndetectors == 1:
            top_suspicious_indices = self.ranked_lists[0][:top_k]
        #elif self.ndetectors == 2:
        #    top_suspicious_indices = combiner.combine_two_lists(ranked_lists[0], ranked_lists[1])[:top_k]
        else:
            top_suspicious_indices = combiner.combine_all_lists(self.ranked_lists)[:top_k]
            
        self.verified_indices.update(top_suspicious_indices)
            
        return top_suspicious_indices
    
    
    def correct_labels(self, index2correct_label):
        error_index2correct_label = {}
        for index, label in index2correct_label.items():
            if label!=self.labels[index]:
                error_index2correct_label[index] = label
            
            self.labels[index] = label
        
        for detector in self.detectors:
            #start = time.clock()
            detector.use_feedback(error_index2correct_label)
            #end = time.clock()
            #print('Incremental update time:', end-start)
            
    
    # it's used to analyze current iteration
    # must be called before correct_labels
    def analyze(self, index2correct_label):
        num_errors = 0
        error_indices = []
        for index, label in index2correct_label.items():
            if label!=self.labels[index]:
                num_errors += 1
                error_indices.append(index)
                
        det_error_poses = []
        for rlist in self.ranked_lists:
            index_pos = [] # pair of (error_index, pos_in_list)
            det_error_count = 0 # number of errors detected
            for index in error_indices:
                found = False
                for pos, v in enumerate(rlist):
                    if v == index:
                        found = True
                        index_pos.append( (index, pos) )
                        det_error_count += 1
                        break
                if not found:
                    index_pos.append( (index, -1) )
            det_error_poses.append( (det_error_count, index_pos) )
            
        return self.iter_count, num_errors, error_indices, det_error_poses    
                
    def set_num_cores(self, num_cores):
        for det in self.detectors:
            det.set_num_cores(num_cores)          
        
