'''
Created on Jun 7, 2019

@author: hzhang0418
'''

import os
#import gc
import time
import pandas as pd

import v6.data_io
import v6.label_debugger
import v6.feature_selection
import v6.exp_analyst

import utils.myconfig

def run():
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/beer.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/bike.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/books1.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/citations.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/movies1.config'
    config_file=r'/scratch/hzhang0418/projects/datasets/mono/restaurants4.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/fodors_zagats.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/cora_new.config'
    
    #config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/abt_buy.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/amazon_google.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/dblp_acm.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/dblp_googlescholar.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/walmart_amazon.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/citations_500k.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/songs_1m.config'
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/songs_small.config'
    
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_clothing.config'
    #config_file = r'/scratch/hzhang0418/projects/datasets/mono2019/trunc_tools.config'
    
    params = utils.myconfig.read_config(config_file)
    
    dataset_name = params['dataset_name']
    datasets_with_golden = set(['beer', 'bike', 'books1', 'citations', 'movies1', 'restaurants4', 'fodors_zagats', 'cora_new'])
    datasets_with_manual = set(['abt_buy', 'amazon_google', 'dblp_acm', 'dblp_googlescholar', 'walmart_amazon',
                                'citations_500k', 'songs_1m', 'songs_small', 'trunc_tools', 'trunc_clothing'])
    if dataset_name in datasets_with_golden:
        run_with_golden(params)
    elif dataset_name in datasets_with_manual:
        
        if dataset_name=='trunc_tools':
            history_prefix = 'tools_0529/tools_0529'
            num_files = 40
            correct_label_attr = 'correct_label'
        elif dataset_name=='trunc_clothing':
            history_prefix = 'clothing_0524/clothing_0524'
            num_files = 40
            correct_label_attr = 'correct_label'
        elif dataset_name=='abt_buy':
            history_prefix = 'history/abt_buy_0411'
            num_files = 4
            correct_label_attr = 'correct_label'
        elif dataset_name=='amazon_google':
            history_prefix = 'history/amazon_google_0411'
            num_files = 4
            correct_label_attr = 'correct_label'
        elif dataset_name=='dblp_acm':
            history_prefix = 'history/dblp_acm_0410'
            num_files = 1
            correct_label_attr = 'label'
        elif dataset_name=='dblp_googlescholar':
            history_prefix = 'history/dblp_googlescholar_0410'
            num_files = 1
            correct_label_attr = 'label'
        elif dataset_name=='walmart_amazon':
            history_prefix = 'history/walmart_amazon_0413'
            num_files = 7
            correct_label_attr = 'correct_label'
        elif dataset_name=='citations_500k':
            history_prefix = 'history/citations_500k_0421'
            num_files = 13
            correct_label_attr = 'correct_label'  
        elif dataset_name=='songs_1m':
            history_prefix = 'history/songs_1m_0419'
            num_files = 7
            correct_label_attr = 'correct_label' 
        elif dataset_name=='songs_small':
            history_prefix = 'songs_small/songs_small_0612'
            num_files = 10
            correct_label_attr = 'correct_label'   
        
        run_with_analyst(params, history_prefix, num_files, correct_label_attr)    
    
        
    else:
        print("Unknown dataset: ", dataset_name)

def run_with_golden(params):
    basedir = params['basedir']
    hpath = os.path.join(basedir, params['hpath'])
    gpath = os.path.join(basedir, 'golden.csv')

    exclude_attrs = ['_id', 'ltable.id', 'rtable.id']
    
    features, labels, pair2index, index2pair = v6.data_io.read_feature_file(hpath, exclude_attrs)
    pair2golden = v6.data_io.read_golden_label_file(gpath)
    
    # label errors
    all_errors = []
    for index, p in index2pair.items():
        if labels[index]!=pair2golden[p]:
            all_errors.append(index)
            
    print(params['dataset_name'])
    
    # config params
    params['fs_alg'] = 'model'
    params['max_list_len'] = 500
    params['detectors'] = 'both'
    
    params['num_cores'] = 8
    params['num_folds'] = 5
    
    params['min_con_dim'] = 1
    params['counting_only'] = True
    
    selected_features = v6.feature_selection.select_features(features, labels, params['fs_alg'])
    
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    
    table_A = pd.read_csv(apath)
    table_B = pd.read_csv(bpath)
    
    debugger = v6.label_debugger.LabelDebugger(selected_features, labels, params)
    
    all_detected_errors, iter_times = debug_labels(debugger, index2pair, pair2golden)
    
    print("Total number of label errors: ", len(all_errors))
    print("Number of iterations: ", debugger.iter_count)
    print("Number of checked pairs: ", len(debugger.verified_indices))
    print("Number of detected errors: ", len(all_detected_errors))
    
    print("First iteration (secs): ", iter_times[0])
    print("For other iterations: ")
    print("Min (secs): ", min(iter_times[1:]))
    print("Max (secs): ", max(iter_times[1:]))
    print("Ave (secs): ", sum(iter_times[1:])/len(iter_times[1:]))


def debug_labels(debugger, index2pair, pair2golden):
    
    iter_times = []
    
    top_k = 20
    num_iter_without_errors = 0
    all_detected_errors = []
    total_num_iters = 0
    
    start = time.clock()
    
    while True:
        top_suspicious_indices = debugger.find_suspicious_labels(top_k)
        end = time.clock()
        
        iter_time = end-start
        iter_times.append(iter_time)
        
        # find their correct labels
        index2correct_label = { index:pair2golden[ index2pair[index] ]  for index in top_suspicious_indices}
        iter_count, num_errors, error_indices, det_error_poses  = debugger.analyze(index2correct_label)
            
        all_detected_errors.extend(error_indices)
            
        if num_errors==0:
            num_iter_without_errors += 1
        else:
            num_iter_without_errors = 0
            
        if num_iter_without_errors>=3:
            break
        
        total_num_iters += 1
        
        start = time.clock()
        debugger.correct_labels(index2correct_label)
        
        if total_num_iters>=40:
            break
        
    return all_detected_errors, iter_times

def run_with_analyst(params, history_prefix, num_files, correct_label_attr):
    basedir = params['basedir']
    hpath = os.path.join(basedir, params['hpath'])

    exclude_attrs = ['_id', 'ltable.id', 'rtable.id']
    
    #features, labels, pair2index, index2pair = v6.data_io.read_feature_file(hpath, exclude_attrs)
    features, labels, pair2index, index2pair = v6.data_io.read_feature_file_v2(hpath, exclude_attrs)
    features, labels, pair2index, index2pair = v6.data_io.read_feature_file_v3(hpath, exclude_attrs)
    
    print(params['dataset_name'])
    
    # config params
    params['fs_alg'] = 'model'
    params['max_list_len'] = 500
    params['detectors'] = 'both'
    
    params['num_cores'] = 8
    params['num_folds'] = 5
    
    params['min_con_dim'] = 1
    params['counting_only'] = True
    
    selected_features = v6.feature_selection.select_features(features, labels, params['fs_alg'])
    features = None
    #gc.collect()
    
    #apath = os.path.join(basedir, params['apath'])
    #bpath = os.path.join(basedir, params['bpath'])
    
    #table_A = pd.read_csv(apath)
    #table_B = pd.read_csv(bpath)
    
    debugger = v6.label_debugger.LabelDebugger(selected_features, labels, params)
    selected_features = None
    #gc.collect()
    
    pair2golden = v6.exp_analyst.read_history(history_prefix, num_files, correct_label_attr)
    
    all_detected_errors, iter_times = debug_labels_with_analyst(debugger, index2pair, pair2golden, labels)
    
    #print("Total number of label errors: ", len(all_errors))
    print("Number of iterations: ", debugger.iter_count)
    print("Number of checked pairs: ", len(debugger.verified_indices))
    print("Number of detected errors: ", len(all_detected_errors))
    
    print("First iteration (secs): ", iter_times[0])
    print("For other iterations: ")
    print("Min (secs): ", min(iter_times[1:]))
    print("Max (secs): ", max(iter_times[1:]))
    print("Ave (secs): ", sum(iter_times[1:])/len(iter_times[1:]))
    
def debug_labels_with_analyst(debugger, index2pair, pair2golden, labels):
    print("Debugging begins:")
    
    iter_times = []
    
    top_k = 20
    num_iter_without_errors = 0
    all_detected_errors = []
    total_num_iters = 0
    
    start = time.clock()
    
    while True:
        top_suspicious_indices = debugger.find_suspicious_labels(top_k)
        end = time.clock()
        
        iter_time = end-start
        iter_times.append(iter_time)
        print(total_num_iters, iter_time)
        
        #gc.collect()
        
        # find their correct labels
        index2correct_label = { }
        for index in top_suspicious_indices:
            p = (int(index2pair[index][0]), int(index2pair[index][1]))
            if p in pair2golden:
                index2correct_label[p] = pair2golden[p]
            else:
                index2correct_label[p] = labels[index]  
        iter_count, num_errors, error_indices, det_error_poses  = debugger.analyze(index2correct_label)
            
        all_detected_errors.extend(error_indices)
            
        if num_errors==0:
            num_iter_without_errors += 1
        else:
            num_iter_without_errors = 0
            
        if num_iter_without_errors>=3:
            break
        
        total_num_iters += 1
        
        start = time.clock()
        debugger.correct_labels(index2correct_label)
        
        if total_num_iters>=40:
            break
        
    return all_detected_errors, iter_times