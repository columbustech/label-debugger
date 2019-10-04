'''
Created on Mar 5, 2019

@author: hzhang0418
'''

import os
import time

import pandas as pd

import v6.data_io
import v6.feature_selection as fs

import v6.fpfn as fpfn
import v6.fpfn_irf as irf

import v6.brute_force as bf
import v6.sort_probing as sp

import v6.mono_est as mest


import utils.myconfig

def run():
    
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/cora_large.config'
    
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_large.config'
    
    config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_tools.config'
    
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono2019/citations_new.config'
    
    params = utils.myconfig.read_config(config_file)
    
    basedir = params['basedir']
    hpath = os.path.join(basedir, params['hpath'])
    
    exclude_attrs = ['_id', 'ltable.id', 'rtable.id']
    
    #features, labels = v6.data_io.read_feature_file(hpath, exclude_attrs)
    table_H = pd.read_csv(hpath)
    features, labels = v6.data_io.get_feature_from_df(table_H, exclude_attrs)
    
    print(features.shape)
    print(len(labels))
    
    selected_features = fs.select_features(features, labels, 'model')
    print(selected_features.shape)
    
    params = {}
    params['num_cores'] = 8
    
    #detector = fpfn.FPFN(selected_features, labels, params)
    #detector = irf.FPFN_IRF(selected_features, labels, params)
    
    params['counting_only'] = True
    #detector = bf.BruteForce(selected_features, labels, params)
    detector = sp.SortProbing(selected_features, labels, params)
    
    #detector = mest.Mono_Est(selected_features, labels, params)
    
    start = time.time()
    ranked_list = detector.detect_and_rank()
    end = time.time()
    print("Time for first iteration: ", (end-start))
    
    print(len(ranked_list))
    print(ranked_list[:20])
    
    #'''
    tmp = ranked_list[:20]
    index2corret_label = {}
    for t in tmp:
        if labels[t]==0:
            index2corret_label[t] = 1
        else:
            index2corret_label[t] = 0
            
    detector.set_num_cores(2)
    
    start = time.time()
    detector.use_feedback(index2corret_label)
    ranked_list = detector.detect_and_rank()
    end = time.time()
    print("Time for second iteration: ", (end-start))
    
    print(len(ranked_list))
    print(ranked_list[:20])
    #'''
    