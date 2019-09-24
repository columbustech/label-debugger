'''
Created on Mar 5, 2019

@author: hzhang0418
'''

import pymp

from operator import itemgetter

from v6.detector import Detector

import v3.mvc as mvc
import v3.gvc as gvc

class Mono(Detector):
    
    def __init__(self, features, labels, params):
        super(Mono, self).__init__(features, labels, params)
        
        self.num_cores = params.get('num_cores', 1)
        
        self.min_con_dim = params.get('min_con_dim', 1)
        
        self.is_counting_only = params.get('counting_only', False)
        if self.is_counting_only:
            self.index2count = {}
        else:
            self.index2incons = {}
            
        self.match_indices = []
        self.nonmatch_indices = []
        
        for index, label in enumerate(labels):
            if label==1:
                self.match_indices.append(index)
            else:
                self.nonmatch_indices.append(index)
            
        self.num_iter = 0
    
    def detect_and_rank(self):
        if self.num_iter == 0:
            if self.is_counting_only:
                self._count_inconsistencies()
            else:
                self._get_inconsistency_indices()
                count = 0
                for _,v in self.index2incons.items():
                    count+= len(v)
                print("Total number of inconsistencies: ", count)
            
        if self.is_counting_only:
            index_counts = [ (index, -count) for index, count in self.index2count.items() if count>0 ]
        else:
            index_counts = [ (index, -len(v)) for index, v in self.index2incons.items()]
                              
        index_counts.sort(key=itemgetter(1,0))
        
        ranked_indices = []
        for t in index_counts:
            if t[1]<0:
                ranked_indices.append(t[0])
            else:
                break
        
        self.num_iter += 1
        
        return ranked_indices
    
    def use_feedback(self, index2correct_label):
        corrected_match_indices = []
        corrected_nonmatch_indices = []
        
        for index, label in index2correct_label.items():
            if label == 1:
                corrected_match_indices.append(index)
            else:
                corrected_nonmatch_indices.append(index)
        
        if self.is_counting_only:
            self._update_inconsistency_count(corrected_match_indices, corrected_nonmatch_indices)
        else:
            self._update_inconsistency_indices(corrected_match_indices, corrected_nonmatch_indices)
        
        
    def compare_features(self, match_features, nonmatch_features, min_con_dim):
        '''
        Check whether the given match_features and nonmatch_features are inconsistent
        '''
        is_incon = True
        num_cons_dim = 0
        i = 0
        for nf in nonmatch_features:
            if nf < match_features[i]:
                num_cons_dim += 1
                if num_cons_dim >= min_con_dim:
                    is_incon = False
                    break
            i += 1
            
        return is_incon
    
    '''
    methods for counting only
    '''
    def _count_inconsistencies(self):
        print("Child class must override this function! ")    
        
    def _update_inconsistency_count(self, corrected_match_indices, corrected_nonmatch_indices):
        '''
        Params:
        corrected_match_indices: the list of indices that were wrongly labeled as nonmatch
        corrected_nonmatch_indices: the list of indices that were wrongly labeled as match
        
        1. For each corrected index, find its original list of inconsistencies
        2. For each corrected index, reduce the count of its inconsistent indices by 1
        3. Correct labels
        4. For each corrected index, find its new list of inconsistencies
        5. For each corrected index, increase the count of its new inconsistent indices by 1
        '''
    
        for index in corrected_match_indices:
            incons = self._find_inconsistent_list(index, 0)
            for k in incons:
                self.index2count[k] -= 1
            #print(index, len(incons))
            self.index2count[index] -= len(incons)
        
        for index in corrected_nonmatch_indices:
            incons = self._find_inconsistent_list(index, 1)
            for k in incons:
                self.index2count[k] -= 1
            #print(index, len(incons))
            self.index2count[index] -= len(incons)
                
        # update nonmatch_indices and match_indices
        self.match_indices = [ index for index in self.match_indices if index not in corrected_nonmatch_indices ]
        self.match_indices.extend(corrected_match_indices)
        
        self.nonmatch_indices = [ index for index in self.nonmatch_indices if index not in corrected_match_indices ]
        self.nonmatch_indices.extend(corrected_nonmatch_indices)
        
        for index in corrected_match_indices:
            incons = self._find_inconsistent_list(index, 1)
            for k in incons:
                if k in self.index2count:
                    self.index2count[k] += 1
                else:
                    self.index2count[k] = 1
        
        for index in corrected_nonmatch_indices:
            incons = self._find_inconsistent_list(index, 0)
            for k in incons:
                if k in self.index2count:
                    self.index2count[k] += 1
                else:
                    self.index2count[k] = 1
        
            
    def _find_inconsistent_list(self, index, label):
        
        incons = []
        
        threads2incons = pymp.shared.dict()
        feature = self.features[index]
        if label==1:
            num_nonmatch = len(self.nonmatch_indices)
            with pymp.Parallel(self.num_cores) as p:
                local_incons = []
                for i in p.range(num_nonmatch):
                    k = self.nonmatch_indices[i] 
                    if self.compare_features(feature, self.features[k], self.min_con_dim)==True:
                        local_incons.append(k)
                threads2incons[p.thread_num] = local_incons       
                
            for tmp in threads2incons.values():
                incons.extend(tmp)
                
        else:
            num_match = len(self.match_indices)
            with pymp.Parallel(self.num_cores) as p:
                local_incons = []
                for i in p.range(num_match):
                    k = self.match_indices[i] 
                    if self.compare_features(self.features[k], feature, self.min_con_dim)==True:
                        local_incons.append(k)
                threads2incons[p.thread_num] = local_incons       
                
            for tmp in threads2incons.values():
                incons.extend(tmp)
            
        return incons


    '''
    methods for storing all inconsistencies
    '''
    def _get_inconsistency_indices(self):
        print("Child class must override this function! ")
    
    
    def _update_inconsistency_indices(self, corrected_match_indices, corrected_nonmatch_indices):
        '''
        Params:
        corrected_match_indices: the list of indices that were wrongly labeled as nonmatch
        corrected_nonmatch_indices: the list of indices that were wrongly labeled as match
        
        to update index2incons:
        First, update nonmatch_indices and match_indices
        
        For each index (x) in corrected_match_indices:
        1. get its list of incon indices,
        2. for each index (y) in the above list, remove x from y's list of incon indices
        
        For each index (x) in corrected_nonmatch_indices:
        1. get its list of incon indices,
        2. for each index (y) in the above list, remove x from y's list of incon indices

        
        For each index (x) in corrected_match_indices:
        3. compute the new list of incon indices for x by comparing x with all nonmatch indices 
        4. for each index (z) in the new list, add x to z's list of incon indices
        
        For each index (x) in corrected_nonmatch_indices:
        3. compute the new list of incon indices for x by comparing x with all match indices 
        4. for each index (z) in the new list, add x to z's list of incon indices
        '''
        
        # test function _find_inconsistent_list
        '''
        for index in corrected_match_indices:
            incons = self._find_inconsistent_list(index, 0)
            if len(incons)!=len(self.index2incons[index]):
                print(index, len(incons), len(self.index2incons[index]))
        
        for index in corrected_nonmatch_indices:
            incons = self._find_inconsistent_list(index, 1)
            if len(incons)!=len(self.index2incons[index]):
                print(index, len(incons), len(self.index2incons[index]))
        '''
        
        
        # update nonmatch_indices and match_indices
        self.match_indices = [ index for index in self.match_indices if index not in corrected_nonmatch_indices ]
        self.match_indices.extend(corrected_match_indices)
        
        self.nonmatch_indices = [ index for index in self.nonmatch_indices if index not in corrected_match_indices ]
        self.nonmatch_indices.extend(corrected_nonmatch_indices)
        
        # process x in corrected_match_indices
        for x in corrected_match_indices:
            x_incons = self.index2incons[x]
            
            for y in x_incons:
                self.index2incons[y].remove(x)
                
        # process x in corrected_nonmatch_indices
        for x in corrected_nonmatch_indices:
            x_incons = self.index2incons[x]
            for y in x_incons:
                self.index2incons[y].remove(x)
                
        #compute incons for those in feedback
        for x in corrected_match_indices:
            x_incons = []
            match_features = self.features[x]
            for z in self.nonmatch_indices:
                is_incon = self.compare_features(match_features, self.features[z], self.min_con_dim)
                if is_incon:
                    x_incons.append(z)
                    if z in self.index2incons:
                        self.index2incons[z].append(x)
                    else:
                        self.index2incons[z] = [x]
            self.index2incons[x] = x_incons 
            
        for x in corrected_nonmatch_indices:
            x_incons = []
            nonmatch_features = self.features[x]
            for z in self.match_indices:
                is_incon = self.compare_features(self.features[z], nonmatch_features, self.min_con_dim)
                if is_incon:
                    x_incons.append(z)
                    if z in self.index2incons:
                        if x not in self.index2incons[z]:
                            self.index2incons[z].append(x)
                    else:
                        self.index2incons[z] = [x]
            self.index2incons[x] = x_incons 
            
        # test function _find_inconsistent_list
        '''
        for index in corrected_match_indices:
            incons = self._find_inconsistent_list(index, 1)
            if len(incons)!=len(self.index2incons[index]):
                print(index, len(incons), len(self.index2incons[index]))
        
        for index in corrected_nonmatch_indices:
            incons = self._find_inconsistent_list(index, 0)
            if len(incons)!=len(self.index2incons[index]):
                print(index, len(incons), len(self.index2incons[index]))
        '''
    
    
    def _getMVC(self):
            
        matches = {}
        for pos in self.match_indices:
            matches[pos] = 1;
        
        left = {}
        right = {}
         
        for index, incon_list in self.index2incons.items():
            if len(incon_list)==0:
                continue
            if index in matches:
                left[index] = list(incon_list)
            else:
                right[index] = list(incon_list);

        print("Before MVC: ", len(left)+len(right))
        return mvc.min_vertex_cover(left, right)
    
    
    def _getGVC(self):
        print("Before GVC: ", len([index for index, incon_list in self.index2incons.items() if len(incon_list)>0]))
        
        return gvc.greedy_vertex_cover(self.index2incons)
    
    