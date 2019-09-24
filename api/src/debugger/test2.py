'''
Created on Mar 15, 2019

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
    
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations.config'
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/cora_large.config'
    
    #config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations_large.config'
    
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_tools.config'
    
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
    
    #selected_features = fs.select_features(features, labels, 'model')
    selected_features = fs.select_features(features, labels, 'none')
    print(selected_features.shape)
    
    params = {}
    params['num_cores'] = 2
    
    #detector = fpfn.FPFN(selected_features, labels, params)
    #detector = irf.FPFN_IRF(selected_features, labels, params)
    
    params['counting_only'] = True
    #detector = bf.BruteForce(selected_features, labels, params)
    detector = sp.SortProbing(selected_features, labels, params)
    
    sp_list = find_suspicious_pairs(detector)
    
    detector = mest.Mono_Est(selected_features, labels, params)
    
    est_list = find_suspicious_pairs(detector)
    
    print(len(sp_list), len(est_list), len(count_common(sp_list, est_list)))
    
    print(200, len(count_common(sp_list[:200], est_list[:200])))
    
    print(100, len(count_common(sp_list[:100], est_list[:100])))
    
    print(50, len(count_common(sp_list[:50], est_list[:50])))
    
    print(20, len(count_common(sp_list[:20], est_list[:20])))
    
def find_suspicious_pairs(detector):
    
    start = time.time()
    ranked_list = detector.detect_and_rank()
    end = time.time()
    print("Time for first iteration: ", (end-start))
    
    print(len(ranked_list))
    print(ranked_list[:20])
    
    return ranked_list
    
def count_common(first, second):
    first_set = set(first)
    second_set = set(second)
    
    return first_set.intersection(second_set)