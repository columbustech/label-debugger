'''
Created on Mar 7, 2019

@author: hzhang0418
'''

import pymp

from v6.mono import Mono

class BruteForce(Mono):
    
    def __init__(self, features, labels, params):
        super(BruteForce, self).__init__(features, labels, params)
        
    
    def _count_inconsistencies(self):
        if self.num_cores==1:
            for ni in self.nonmatch_indices:
                self.index2count[ni] = 0
            
            for mi in self.match_indices:
                match_features = self.features[mi]
                count = 0
                for ni in self.nonmatch_indices:
                    inconsistent = self.compare_features(match_features, self.features[ni], self.min_con_dim)
                    if inconsistent == True:
                        count += 1
                        self.index2count[ni] += 1
                self.index2count[mi] = count
                
        else:
            nmatch = len(self.match_indices)
        
            threads2incons_count = pymp.shared.dict()
            
            with pymp.Parallel(self.num_cores) as p:
                local_index2incons_count = {}
                for index in p.range(nmatch):
                    mi = self.match_indices[index]
                    match_features = self.features[mi]
                    
                    count = 0
                    
                    for ni in self.nonmatch_indices:
                        inconsistent = self.compare_features(match_features, self.features[ni], self.min_con_dim)
                        if inconsistent == True:
                            count += 1
                            if ni in local_index2incons_count:
                                local_index2incons_count[ni] += 1
                            else:
                                local_index2incons_count[ni] = 1
                    
                    if count>0:
                        local_index2incons_count[mi] = count
                    
                threads2incons_count[p.thread_num] = local_index2incons_count
            
            for _, local_index2incons_count in threads2incons_count.items():
                for index, count in local_index2incons_count.items():
                    if index in self.index2count:
                        self.index2count[index] += count
                    else:
                        self.index2count[index] = count 
                    
        return self.index2count
    
    
    def _get_inconsistency_indices(self):
        
        if self.num_cores==1:
        
            for mi in self.match_indices:
                match_features = self.features[mi]
                incons_indices = []
                for ni in self.nonmatch_indices:
                    inconsistent = self.compare_features(match_features, self.features[ni], self.min_con_dim)
                    if inconsistent == True:
                        incons_indices.append(ni)
                        
                if len(incons_indices)>0:
                    self.index2incons[mi] = incons_indices
                    for ni in incons_indices:
                        if ni in self.index2incons:
                            self.index2incons[ni].append(mi)
                        else:
                            self.index2incons[ni] = [mi]
                            
        else:
            
            nmatch = len(self.match_indices)
        
            threads2incons = pymp.shared.dict()
            
            with pymp.Parallel(self.num_cores) as p:
                local_index2incons = {}
                for index in p.range(nmatch):
                    mi = self.match_indices[index]
                    match_features = self.features[mi]
                    
                    incons_indices = []
                    
                    for ni in self.nonmatch_indices:
                        inconsistent = self.compare_features(match_features, self.features[ni], self.min_con_dim)
                        if inconsistent == True:
                            incons_indices.append(ni)
                    
                    if len(incons_indices)>0:
                        local_index2incons[mi] = incons_indices
                    
                threads2incons[p.thread_num] = local_index2incons
            
            for _, local_index2incons in threads2incons.items():
                for mi, ni_indices in local_index2incons.items():
                    self.index2incons[mi] = ni_indices
                    for ni in ni_indices:
                        if ni in self.index2incons:
                            self.index2incons[ni].append(mi)
                        else:
                            self.index2incons[ni] = [mi]
        
        return self.index2incons