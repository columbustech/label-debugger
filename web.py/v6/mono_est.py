'''
Created on Mar 10, 2019

@author: hzhang0418
'''
import numpy as np

from v6.detector import Detector
from operator import itemgetter

class Mono_Est(Detector):
    
    def __init__(self, features, labels, params):
        super(Mono_Est, self).__init__(features, labels, params)    
        
        self.match_indices = []
        self.nonmatch_indices = []
        
        for index, label in enumerate(labels):
            if label==1:
                self.match_indices.append(index)
            else:
                self.nonmatch_indices.append(index)
                
        (_, ncol) = self.features.shape
        self.nfeatures = ncol
                
        self.match_buckets = self._build_buckets(self.match_indices)
        self.nonmatch_buckets = self._build_buckets(self.nonmatch_indices)
        
        self.num_iter = 0
    
                                  
    def detect_and_rank(self):
        #indices = self._est_all()
        indices = self._est_all_with_count()
            
        # sort in ascending order of negative incon_perc
        # if same negative incon_perc, sort in ascending order of indices
        indices.sort(key=itemgetter(1,0))
        
        #print(indices[:20])
        
        return [ t[0] for t in indices ]
    
    def use_feedback(self, index2correct_label):
        for index, label in index2correct_label.items():
            self.labels[index] = label
            
        self._update_buckets()
    
    
    def _build_buckets(self, indices):
        buckets = []
        for i in range(self.nfeatures):
            # statistics of values at this feature
            tmp = {}
            for k in indices:
                v = self.features[k,i]
                if v in tmp:
                    tmp[v] += 1
                else:
                    tmp[v] = 1
            buckets.append(tmp)
            
        return buckets
    
    
    def _est_all(self):
        indices = []
        for index, label in enumerate(self.labels):
            incon_perc = self._est_inconsistent_percetage(index, label) 
            if incon_perc>0:
                # use negative incon_perc
                indices.append( (index, -incon_perc) )
            
        return indices
    
    
    def _est_inconsistent_percetage(self, index, label):
        if label==1:
            # est using nonmatch buckets
            num_nonmatch = len(self.nonmatch_indices)
            perc = 1
            for i in range(self.nfeatures):
                tmp = self.nonmatch_buckets[i]
                count = 0
                for k,v in tmp.items():
                    if k>=self.features[index, i]:
                        count += v
                perc *= 1.0*count/num_nonmatch
                
            incon_perc = perc
        else:
            # est using match buckets
            num_match = len(self.match_indices)
            perc = 1
            for i in range(self.nfeatures):
                tmp = self.match_buckets[i]
                count = 0
                for k,v in tmp.items():
                    if k<=self.features[index, i]:
                        count += v
                perc *= 1.0*count/num_match
                
            incon_perc = perc
        
        return incon_perc
    
    
    def _est_all_with_count(self):
        mv_counts_list, nv_counts_list = self._count_all()
        
        num_match = len(self.match_indices)
        num_nonmatch = len(self.nonmatch_indices)
        
        indices = []
        for index, label in enumerate(self.labels):
            if label==1:
                perc = 1
                for i in range(self.nfeatures):
                    mv = self.features[index, i]
                    count = mv_counts_list[i][mv]
                    perc *= 1.0*count/num_nonmatch
                    
                incon_perc = perc*num_nonmatch
                
            else:
                perc = 1
                for i in range(self.nfeatures):
                    nv = self.features[index, i]
                    count = nv_counts_list[i][nv]
                    perc *= 1.0*count/num_match
                    
                incon_perc = perc*num_match
                
            if incon_perc>0:
                # use negative incon_perc
                indices.append( (index, -incon_perc) )
                
        return indices    
        
    
    def _count_all(self):
        mv_counts_list = []
        nv_counts_list = []
        
        for i in range(self.nfeatures):
            mv2count_sum, nv2count_sum = self._count_feature(i)
            mv_counts_list.append(mv2count_sum)
            nv_counts_list.append(nv2count_sum)
            
        return mv_counts_list, nv_counts_list
            
    
    def _count_feature(self, feature_index):
        match_buckets = list(self.match_buckets[feature_index].keys())
        nonmatch_buckets = list(self.nonmatch_buckets[feature_index].keys())
        
        match_buckets.sort()
        nonmatch_buckets.sort()
        
        # for each mv, find #nv that has same or greater value
        mv2count_sum = {} 
        
        mv_index = len(match_buckets)-1
        nv_index = len(nonmatch_buckets)-1
        
        nv = nonmatch_buckets[nv_index]
        count = 0
        while mv_index>=0:
            mv = match_buckets[mv_index]
            while nv_index>=0 and nv>=mv:
                count += self.nonmatch_buckets[feature_index][nv]
                nv_index -= 1
                nv = nonmatch_buckets[nv_index]
                
            mv2count_sum[mv] = count
            mv_index -= 1
            
        # for each nv, find #mv that has same or smaller value    
        nv2count_sum = {} 
        
        mv_index = 0
        nv_index = 0
        
        mv = match_buckets[mv_index]
        count = 0
        while nv_index<len(nonmatch_buckets):
            nv = nonmatch_buckets[nv_index]
            while mv_index<len(match_buckets) and nv>=mv:
                count += self.match_buckets[feature_index][mv]
                mv_index += 1
                if mv_index>=len(match_buckets):
                    break
                mv = match_buckets[mv_index]
                
            nv2count_sum[nv] = count
            nv_index += 1
    
        return mv2count_sum, nv2count_sum
    
    
    def _update_buckets(self):
        pass