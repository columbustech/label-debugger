'''
Created on Mar 26, 2019

@author: hzhang0418
'''
import os

import pandas as pd

import py_entitymatching as em

import v6.data_io
import v6.label_debugger
import v6.feature_selection

import utils.myconfig

# global data
#features = None
labels = None
pair2index = None
index2pair = None

selected_features = None
debugger = None

table_A = None
table_B = None

all_detected_errors = []
num_iter_without_errors = 0

def read_config(config_file):
    return utils.myconfig.read_config(config_file)


def read_tables(params):
    global table_A, table_B
    
    basedir = params['basedir']
    apath = os.path.join(basedir, params['apath'])
    bpath = os.path.join(basedir, params['bpath'])
    
    table_A = em.read_csv_metadata(apath, key='id')
    table_B = em.read_csv_metadata(bpath, key='id')


def prepare_debugger(params):
    global labels, pair2index, index2pair, selected_features, debugger, num_iter_without_errors
    
    basedir = params['basedir']
    hpath = os.path.join(basedir, params['hpath'])

    exclude_attrs = ['_id', 'ltable.id', 'rtable.id']
    
    features, labels, pair2index, index2pair = v6.data_io.read_feature_file(hpath, exclude_attrs)
    
    print(params['dataset_name'])
    
    # config params
    params['fs_alg'] = 'model'
    params['max_list_len'] = 500
    params['detectors'] = 'both'
    
    params['num_cores'] = 4
    params['num_folds'] = 5
    
    params['min_con_dim'] = 1
    params['counting_only'] = True
    
    selected_features = v6.feature_selection.select_features(features, labels, params['fs_alg'])
    
    debugger = v6.label_debugger.LabelDebugger(selected_features, labels, params)
    num_iter_without_errors = 0


def find_suspicious_labels(top_k, file_prefix):
    top_suspicious_indices = debugger.find_suspicious_labels(top_k)

    # combine those suspicious pairs into a dataframe and save to file
    all_pairs_with_label = []
    for index in top_suspicious_indices:
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
        all_pairs_with_label.append(tmp)
    df = pd.DataFrame(all_pairs_with_label)

    output_file = file_prefix + '_iter_' + str(debugger.iter_count) + '.csv'
    df.to_csv(output_file, index=False)
    
    
def read_feedback_file(feedback_file, correct_label_attr):
    df = pd.read_csv(feedback_file)
    index2correct_label = {}
    for _, row in df.iterrows():
        p = (str(row['ltable.id']), str(row['rtable.id']))
        label = int(row[correct_label_attr])
        index = pair2index[p]
        index2correct_label[index] = label
        
    return index2correct_label
    

def use_feedback(index2correct_label):
    global num_iter_without_errors
    iter_count, num_errors, error_indices, det_error_poses  = debugger.analyze(index2correct_label)
    print('Iteration: ', iter_count)
    print("Number of errors found: ", num_errors)
    print("Error indices: ", error_indices)
    print("Detector performance: ")
    for n, (count, pos) in enumerate(det_error_poses):
        print("Detector ", n, "found ", count, " errors")
        print("Positions: ", pos)
        
    all_detected_errors.extend(error_indices)
        
    if num_errors==0:
        num_iter_without_errors += 1
    else:
        num_iter_without_errors = 0
        
    if num_iter_without_errors>=3:
        print("Three iterations without errors...")
    
    debugger.correct_labels(index2correct_label)
 
   
def finish():
    #print("Total number of label errors: ", len(all_errors))
    print("Number of iterations: ", debugger.iter_count)
    print("Number of checked pairs: ", len(debugger.verified_indices))
    print("Number of detected errors: ", len(all_detected_errors))
    
    
def read_history(history_prefix, num_files, correct_label_attr):
    pairs2label = {}
    for k in range(num_files):
        history_file = history_prefix + '_iter_' + str(k+1) + '.csv'
        df = pd.read_csv(history_file)
        for _, row in df.iterrows():
            p = (row['ltable.id'], row['rtable.id'])
            label = row[correct_label_attr]
            pairs2label[p] = label
    
    return pairs2label

def use_history(input_file, history_pairs2label):
    df = pd.read_csv(input_file)
    for _, row in df.iterrows():
        p = (row['ltable.id'], row['rtable.id'])
        if p in history_pairs2label:
            print(p, history_pairs2label[p])
        else:
            print(p, 'Not found')
    
    
def simulate(pairs2label, top_k, file_prefix, correct_label_attr='correct_label'):
    while True:
        top_suspicious_indices = debugger.find_suspicious_labels(top_k)

        # combine those suspicious pairs into a dataframe and save to file
        all_pairs_with_label = []
        num_with_labels = 0
        for index in top_suspicious_indices:
            p = index2pair[index]
            label = labels[index]
            left = table_A.loc[ table_A['id'] == int(p[0])]
            right = table_B.loc[ table_B['id'] == int(p[1])]
            p_tmp = (int(p[0]), int(p[1]))
            if p_tmp in pairs2label:
                correct_label = pairs2label[p_tmp]
                num_with_labels += 1
            else:
                correct_label = -1
                
            tmp = {}
            tmp[correct_label_attr] = correct_label
            for col in left:
                tmp['ltable.'+col] = left.iloc[0][col]
            for col in right:
                tmp['rtable.'+col] = right.iloc[0][col]
            tmp['label'] = label
            all_pairs_with_label.append(tmp)
        df = pd.DataFrame(all_pairs_with_label)
    
        output_file = file_prefix + '_iter_' + str(debugger.iter_count) + '.csv'
        df.to_csv(output_file, index=False)
        
        if num_with_labels<len(top_suspicious_indices):
            print("Not all suspicious pairs have correct labels!!!")
            break
    
        index2correct_label = read_feedback_file(output_file, correct_label_attr)
        use_feedback(index2correct_label)