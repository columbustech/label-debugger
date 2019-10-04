'''
Created on May 27, 2019

@author: hzhang0418
'''

import os

import pandas as pd

import v6.data_io
import v6.label_debugger
import v6.feature_selection

import utils.myconfig

def run():
    config_file = r'/scratch/hzhang0418/projects/datasets/labeldebugger/cora_new.config'
    
    params = utils.myconfig.read_config(config_file)
    
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
    params['fs_alg'] = 'none'
    params['max_list_len'] = 500
    params['detectors'] = 'both'
    
    params['num_cores'] = 1
    params['num_folds'] = 5
    
    params['min_con_dim'] = 1
    params['counting_only'] = True
    
    #index = 324
    #print( labels[index], pair2golden[index2pair[index]] )
    
    selected_features = v6.feature_selection.select_features(features, labels, params['fs_alg'])
    
    debugger = v6.label_debugger.LabelDebugger(selected_features, labels, params)
    
    all_detected_errors = debug_labels(debugger, index2pair, pair2golden)
    
    print("Total number of label errors: ", len(all_errors))
    print("Number of iterations: ", debugger.iter_count)
    print("Number of checked pairs: ", len(debugger.verified_indices))
    print("Number of detected errors: ", len(all_detected_errors))
    
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    
    table_A = pd.read_csv(apath)
    table_B = pd.read_csv(bpath)
    
    #show all errors
    table_A['id'] = table_A['id'].astype(str)
    table_B['id'] = table_B['id'].astype(str)
    all_error_pairs = []
    for index in all_errors:
        p = index2pair[index]
        label = labels[index]
        left = table_A.loc[ table_A['id'] == str(p[0])]
        right = table_B.loc[ table_B['id'] == str(p[1])]
        tmp = {}
        for col in left:
            tmp['ltable.'+col] = left.iloc[0][col]
        for col in right:
            tmp['rtable.'+col] = right.iloc[0][col]
            tmp['label'] = label
        all_error_pairs.append(tmp)
        
    if len(all_error_pairs)>0:
        df = pd.DataFrame(all_error_pairs)
        output_file = params['dataset_name']+'_all_errors.csv'
        df.to_csv(output_file, index=False)
        
    '''
    # show missed errors
    missed_error_pairs = []
    for index in all_errors:
        if index not in all_detected_errors:
            p = index2pair[index]
            label = labels[index]
            left = table_A.loc[ table_A['id'] == int(p[0])]
            right = table_B.loc[ table_B['id'] == int(p[1])]
            tmp = {}
            for col in left:
                tmp['ltable.'+col] = left.iloc[0][col]
            for col in right:
                tmp['rtable.'+col] = right.iloc[0][col]
                tmp['label'] = label
            missed_error_pairs.append(tmp)
        
    if len(missed_error_pairs)>0:
        df = pd.DataFrame(missed_error_pairs)
        output_file = params['dataset_name']+'missed_errors.csv'
        df.to_csv(output_file, index=False)
    '''

def debug_labels(debugger, index2pair, pair2golden):
    
    top_k = 20
    num_iter_without_errors = 0
    all_detected_errors = []
    total_num_iters = 0
    while True:
        top_suspicious_indices = debugger.find_suspicious_labels(top_k)
        
        # find their correct labels
        index2correct_label = { index:pair2golden[ index2pair[index] ]  for index in top_suspicious_indices}
        iter_count, num_errors, error_indices, det_error_poses  = debugger.analyze(index2correct_label)
        print('Iteration: ', iter_count)
        print("Number of errors found: ", num_errors)
        #print("Error indices: ", error_indices)
        #print("Detector performance: ")
        #for n, (count, pos) in enumerate(det_error_poses):
        #    print("Detector ", n, "found ", count, " errors")
        #    print("Positions: ", pos)
            
        all_detected_errors.extend(error_indices)
           
        ''' 
        if num_errors==0:
            num_iter_without_errors += 1
        else:
            num_iter_without_errors = 0
            
        if num_iter_without_errors>=3:
            break
        '''
        debugger.correct_labels(index2correct_label)
        
        if len(top_suspicious_indices)==0:
            print("No suspicious labels at iteration: ", total_num_iters)
            break
        
        total_num_iters += 1
        '''
        if total_num_iters>=40:
            break
        '''
        
    return all_detected_errors