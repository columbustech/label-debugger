'''
Created on Mar 9, 2019

@author: hzhang0418
'''

import pymp

import bitarray as bar

from v6.mono import Mono

class SortProbing(Mono):
    
    def __init__(self, features, labels, params):
        super(SortProbing, self).__init__(features, labels, params)
        
        (_, ncol) = self.features.shape
        self.nfeatures = ncol
        
        self.feature_to_sorted_nonmatch = []
        
        # map pos to its index
        self.match_pos_to_index = {}
        self.nonmatch_pos_to_index = {}
        
        # list of dicts of lists, where the size of the first list == self.nfeatures
        # for each dict, a key is a feature value, and the associated value 
        # is the list of poses of matches (or nonmatches)  
        self.clusters_to_match_poses = []
        self.clusters_to_nonmatch_poses = []
        
        # list of tuples
        # each tuple is (match_clusters, nonmatch_clusters, the end_index map)
        self.feature_to_sorted_clusters = []
        
        self.match_clusters_to_nonmatch_bitarray = []
        
    
        
    def _count_inconsistencies(self):
        # clustering
        self.cluster_indices_for_all_features()
        
        #self.create_mixes()
        self.create_sets()
        #self.create_bitarrays()
        
        # probing
        threads2incons_count = pymp.shared.dict()
        tmp = list(self.match_pos_to_index.items())
        num = len(tmp)
        
        with pymp.Parallel(self.num_cores) as p:
            local_index2incons_count = {}
            
            #for match_pair_pos, match_index in p.iterate(self.match_pos_to_index.items()):
            for i in p.range(num):
                match_pair_pos, match_index = tmp[i]
                #incons_nonmatch_indices = self.get_list_of_inconsistent_indices_mixes(match_pair_pos)
                incons_nonmatch_indices = self.get_list_of_inconsistent_indices_vset(match_pair_pos)
                #incons_nonmatch_indices = self.get_list_of_inconsistent_indices_vbitarray(match_pair_pos)
                
                if len(incons_nonmatch_indices) == 0:
                    continue
                
                local_index2incons_count[match_index] = len(incons_nonmatch_indices)
                for nonmatch_index in incons_nonmatch_indices:
                    if nonmatch_index in local_index2incons_count:
                        local_index2incons_count[nonmatch_index] += 1
                    else:
                        local_index2incons_count[nonmatch_index] = 1
                
            threads2incons_count[p.thread_num] = local_index2incons_count   
            #print(p.num_threads, p.thread_num)
            
        for _, local_index2incons_count in threads2incons_count.items():
                for index, count in local_index2incons_count.items():
                    if index in self.index2count:
                        self.index2count[index] += count
                    else:
                        self.index2count[index] = count 
                    
        return self.index2count
        
    
    '''
    
    '''    
    def _get_inconsistency_indices(self):
        
        # clustering
        self.cluster_indices_for_all_features()
        
        #self.create_mixes()
        self.create_sets()
        #self.create_bitarrays()
        
        if self.num_cores==1:
            # probing
            for match_pair_pos, match_index in self.match_pos_to_index.items():
                #incons_nonmatch_indices = self.get_list_of_inconsistent_indices_mixes(match_pair_pos)
                incons_nonmatch_indices = self.get_list_of_inconsistent_indices_vset(match_pair_pos)
                #incons_nonmatch_indices = self.get_list_of_inconsistent_indices_vbitarray(match_pair_pos)
                
                if len(incons_nonmatch_indices) == 0:
                    continue
                
                self.index2incons[match_index] = incons_nonmatch_indices
                for nonmatch_index in incons_nonmatch_indices:
                    if nonmatch_index in self.index2incons:
                        self.index2incons[nonmatch_index].append(match_index)
                    else:
                        self.index2incons[nonmatch_index] = [ match_index ] 
            
        
        else:
            # probing
            threads2incons = pymp.shared.dict()
            tmp = list(self.match_pos_to_index.items())
            num = len(tmp)
            
            with pymp.Parallel(self.num_cores) as p:
                local_index2incons = {}
                
                #for match_pair_pos, match_index in p.iterate(self.match_pos_to_index.items()):
                for i in p.range(num):
                    match_pair_pos, match_index = tmp[i]
                    #incons_nonmatch_indices = self.get_list_of_inconsistent_indices_mixes(match_pair_pos)
                    incons_nonmatch_indices = self.get_list_of_inconsistent_indices_vset(match_pair_pos)
                    #incons_nonmatch_indices = self.get_list_of_inconsistent_indices_vbitarray(match_pair_pos)
                    
                    if len(incons_nonmatch_indices) == 0:
                        continue
                    
                    local_index2incons[match_index] = incons_nonmatch_indices
                    
                threads2incons[p.thread_num] = local_index2incons   
                #print(p.num_threads, p.thread_num)
                
            for _, local_index2incons in threads2incons.items():
                for mi, ni_indices in local_index2incons.items():
                    self.index2incons[mi] = ni_indices
                    for ni in ni_indices:
                        if ni in self.index2incons:
                            self.index2incons[ni].append(mi)
                        else:
                            self.index2incons[ni] = [mi]
                        
    def find_end_index(self, match_clusters, nonmatch_clusters, feature_index, cluster_to_nonmatch_poses):
        # for each match cluster, find the first nonmatch cluster with same or greater value
        end_index = {}
        # count the number of nonmatch indices with same or greater value
        index_count = {}
        
        i = 0
        k = 0
        
        count = len(self.nonmatch_indices)
        
        while i<len(match_clusters):
            while k<len(nonmatch_clusters) and match_clusters[i]>nonmatch_clusters[k]:
                count -= len(cluster_to_nonmatch_poses[ nonmatch_clusters[k] ])
                k += 1
            end_index[ match_clusters[i] ] = k
            index_count[ match_clusters[i] ] = count
            i += 1
            
        return end_index, index_count

    def cluster_indices_for_all_features(self):
                
        for pos, index in enumerate(self.match_indices):
            self.match_pos_to_index[pos] = index
                    
        for pos, index in enumerate(self.nonmatch_indices):
            self.nonmatch_pos_to_index[pos] = index
                    
        # parallel on features        
        shared_clusters_to_match_poses = pymp.shared.dict()
        shared_clusters_to_nonmatch_poses = pymp.shared.dict()
        shared_feature_to_sorted_clusters = pymp.shared.dict()
        
        with pymp.Parallel(self.num_cores) as p:
            for i in p.range(self.nfeatures):
                cluster_to_match_poses = {}
                
                for pos, index in enumerate(self.match_indices):
                    f = self.features[index][i]
                    if f in cluster_to_match_poses:
                        cluster_to_match_poses[f].append(pos)
                    else:
                        cluster_to_match_poses[f] = [pos]
                        
                cluster_to_nonmatch_poses = {}
                    
                for pos, index in enumerate(self.nonmatch_indices):
                    f = self.features[index][i]
                    if f in cluster_to_nonmatch_poses:
                        cluster_to_nonmatch_poses[f].append(pos)
                    else:
                        cluster_to_nonmatch_poses[f] = [pos]
                    
                shared_clusters_to_match_poses[i] = cluster_to_match_poses
                shared_clusters_to_nonmatch_poses[i] = cluster_to_nonmatch_poses
                
                match_clusters = list(cluster_to_match_poses.keys())
                nonmatch_clusters = list(cluster_to_nonmatch_poses.keys())
                match_clusters.sort()
                nonmatch_clusters.sort()
                tmp = (match_clusters, nonmatch_clusters, self.find_end_index(match_clusters, nonmatch_clusters, i, cluster_to_nonmatch_poses))
                shared_feature_to_sorted_clusters[i] = tmp
                
            #print(p.num_threads, p.thread_num)
            
        for i in range(self.nfeatures):
            self.clusters_to_match_poses.append(shared_clusters_to_match_poses[i])
            self.clusters_to_nonmatch_poses.append(shared_clusters_to_nonmatch_poses[i])
            self.feature_to_sorted_clusters.append(shared_feature_to_sorted_clusters[i])
        
    '''
    def sort_indices_for_all_features(self):
        for i in range(self.nfeatures):
            list_of_index_feature = [ (index, self.feature_vector[index][i]) for index in self.nonmatch_indices ]
            list_of_index_feature.sort(key=lambda t: t[1]) # ascending order of feature value
            self.feature_to_sorted_nonmatch.append(list_of_index_feature)
    '''
            
    def create_mixes(self):
        for feature_index in range(self.nfeatures):
            tmp = {}
            for cluster_value in self.clusters_to_match_poses[feature_index].keys():
                
                triple = self.feature_to_sorted_clusters[feature_index]
                nonmatch_clusters = triple[1]
                end_indices, index_counts = triple[2]
                
                end_index = end_indices[cluster_value]
                count = index_counts[cluster_value]
                
                cluster_to_nonmatch_pos = self.clusters_to_nonmatch_poses[feature_index]
                
                if count>2000: # create bitarray
                    a = bar.bitarray(len(self.nonmatch_indices))
                    a.setall(True)
                    
                    # set bits for incons nonmatches
                    for i in range(end_index, len(nonmatch_clusters)):
                        for pos in cluster_to_nonmatch_pos[ nonmatch_clusters[i] ]:
                            a[pos] = False
                    
                    tmp[cluster_value] = a
                else:
                    # use set instead
                    a = set()
                    for i in range(end_index, len(nonmatch_clusters)):
                        a.update(cluster_to_nonmatch_pos[ nonmatch_clusters[i] ])
                        
                    tmp[cluster_value] = a
                    
            self.match_clusters_to_nonmatch_bitarray.append(tmp)
    
    
    def get_list_of_inconsistent_indices_mixes(self, match_pair_pos):
        
        match_index = self.match_pos_to_index[match_pair_pos]
        
        cluster_value = self.features[match_index][0]
        tmp = self.match_clusters_to_nonmatch_bitarray[0][cluster_value]
        
        if isinstance(tmp, set):
            pos_bitarray = set()
            pos_bitarray.update(tmp)
        else:
            pos_bitarray = bar.bitarray(len(self.nonmatch_indices)) # all init to zero
            pos_bitarray.setall(False)
            pos_bitarray |= tmp
        
        for feature_index in range(1, self.nfeatures):
            cluster_value = self.features[match_index][feature_index]
            incons_bitarray = self.match_clusters_to_nonmatch_bitarray[feature_index][cluster_value]
            if isinstance(incons_bitarray, set):
                if isinstance(pos_bitarray, set):
                    pos_bitarray.intersection_update(incons_bitarray)        
                else:
                    pos_bitarray = set([ pos for pos in incons_bitarray if pos_bitarray[pos]==False ])
            else:
                if isinstance(pos_bitarray, set):
                    pos_bitarray = set([ pos for pos in pos_bitarray if incons_bitarray[pos]==False ])
                else:
                    pos_bitarray |= incons_bitarray
            
        incons_nonmatch_indices = []
        
        if isinstance(pos_bitarray, set):
            for pos in pos_bitarray:
                incons_nonmatch_indices.append(self.nonmatch_pos_to_index[pos])
        else:
            for i in range(len(self.nonmatch_indices)):
                if pos_bitarray[i] == False:
                    incons_nonmatch_indices.append(self.nonmatch_pos_to_index[i])
                
        return incons_nonmatch_indices
    
    
                    
    '''
    using set
    '''
    def create_sets(self):
        for feature_index in range(self.nfeatures):
            tmp = {}
            for cluster_value in self.clusters_to_match_poses[feature_index].keys():
                
                triple = self.feature_to_sorted_clusters[feature_index]
                nonmatch_clusters = triple[1]
                end_indices, index_counts = triple[2]
                
                end_index = end_indices[cluster_value]
                #count = index_counts[cluster_value]
                
                cluster_to_nonmatch_pos = self.clusters_to_nonmatch_poses[feature_index]
                
                # use set instead
                a = set()
                for i in range(end_index, len(nonmatch_clusters)):
                    a.update(cluster_to_nonmatch_pos[ nonmatch_clusters[i] ])

                tmp[cluster_value] = a
                    
            self.match_clusters_to_nonmatch_bitarray.append(tmp)
    
    
    def get_list_of_inconsistent_indices_vset(self, match_pair_pos):
        
        match_index = self.match_pos_to_index[match_pair_pos]
        
        pos_bitarray = set()
        
        cluster_value = self.features[match_index][0]
        incons_bitarray = self.match_clusters_to_nonmatch_bitarray[0][cluster_value]
        
        pos_bitarray.update(incons_bitarray)
        
        for feature_index in range(1, self.nfeatures):
            cluster_value = self.features[match_index][feature_index]
            incons_bitarray = self.match_clusters_to_nonmatch_bitarray[feature_index][cluster_value]
            pos_bitarray.intersection_update(incons_bitarray)        

        incons_nonmatch_indices = []
        
        for pos in pos_bitarray:
            incons_nonmatch_indices.append(self.nonmatch_pos_to_index[pos])
                
        return incons_nonmatch_indices
    
    '''
    using bitarray 
    '''
    def create_bitarrays(self):
        for feature_index in range(self.nfeatures):
            tmp = {}
            for cluster_value in self.clusters_to_match_poses[feature_index].keys():
                a = bar.bitarray(len(self.nonmatch_indices))
                a.setall(True)
                
                # set bits for incons nonmatches
                triple = self.feature_to_sorted_clusters[feature_index]
                nonmatch_clusters = triple[1]
                end_indices, _ = triple[2]
                end_index = end_indices[cluster_value]
                
                cluster_to_nonmatch_pos = self.clusters_to_nonmatch_poses[feature_index]
                for i in range(end_index, len(nonmatch_clusters)):
                    for pos in cluster_to_nonmatch_pos[ nonmatch_clusters[i] ]:
                        a[pos] = False
                
                tmp[cluster_value] = a
                 
            self.match_clusters_to_nonmatch_bitarray.append(tmp)
        
            
    def mark_incons_poses(self, pos_bitarray, cluster_value, feature_index):
        incons_bitarray = self.match_clusters_to_nonmatch_bitarray[feature_index][cluster_value]
        pos_bitarray |= incons_bitarray
    
    def get_list_of_inconsistent_indices_vbitarray(self, match_pair_pos):
        pos_bitarray = bar.bitarray(len(self.nonmatch_indices)) # all init to zero
        pos_bitarray.setall(False)
        
        match_index = self.match_pos_to_index[match_pair_pos]
        
        for feature_index in range(self.nfeatures):
            self.mark_incons_poses(pos_bitarray, self.features[match_index][feature_index], feature_index)
            
        incons_nonmatch_indices = []
        for i in range(len(self.nonmatch_indices)):
            if pos_bitarray[i] == False:
                incons_nonmatch_indices.append(self.nonmatch_pos_to_index[i])
                
        return incons_nonmatch_indices