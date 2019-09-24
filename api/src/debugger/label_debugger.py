'''
Created on Mar 6, 2019

@author: hzhang0418
'''
import feature_selection as fs

import combiner

class LabelDebugger(object):
    
    def __init__(self, features, labels, params):
        self.labels = labels
        self.features = fs.select_features(features, labels, params.get('fs_alg', 'none'))
        self.max_list_len = params.get('max_list_len', 200)
        self._start_detectors(params)
        
        # indices whose label has been verified by the analyst
        self.verified_indices = set()
    
    def _start_detectors(self, params):
        self.detectors = []
        detector_types = params.get('detectors', 'fpfn')
        
        if detector_types=='fpfn':
            self.detectors.append()
        elif detector_types=='mono':
            self.detectors.append()
        elif detector_types=='both':
            self.detectors.append()
    
    def find_suspicious_labels(self, top_k):
        ranked_lists = [ detector.detect_and_rank()[:self.max_list_len] for detector in self.detectors]
        
        if self.ndetectors == 1:
            return ranked_lists[0][:top_k]
        elif self.ndetectors == 2:
            return combiner.combine_two_lists(ranked_lists[0], ranked_lists[1])[:top_k]
        else:
            return combiner.combine_all_lists(ranked_lists)[:top_k]
    
    def correct_labels(self, corrected_labels):
        for detector in self.detectors:
            detector.use_feedback()