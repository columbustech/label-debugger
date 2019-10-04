'''
Created on Mar 5, 2019

@author: hzhang0418
'''

import csv
#import gc

import numpy as np
import pandas as pd

'''

'''
def read_feature_file(feature_file, exclude_attrs=['id', 'ltable.id', 'rtable.id'], label_attr='label'):

    with open(feature_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', dialect=csv.excel_tab)
        rows = [ row for row in reader]
    
    nrows = len(rows)
    if nrows==0:
        return np.array([], dtype=np.float), np.array([], dtype=np.int)
    
    if label_attr not in rows[0]:
        raise Exception('Missing label attribute: '+label_attr)
    
    attributes = list(rows[0].keys())
    attributes.remove(label_attr)  
    
    for attr in exclude_attrs:
        if attr in attributes:
            attributes.remove(attr)
            
    nfeatures = len(attributes)
    
    features = np.empty( (nrows, nfeatures), dtype=np.float)
    labels = np.empty(nrows, dtype=np.int)
    
    pair2index = {}
    index2pair = {}
    for index, row in enumerate(rows):
        labels[index] = int(row[label_attr])
        p = (row['ltable.id'], row['rtable.id'])
        pair2index[p] = index
        index2pair[index] = p
        for k, attr in enumerate(attributes):
            features[index][k] = float( "{0:.2f}".format(float(row[attr])))
            
    #gc.collect()
    
    return features, labels, pair2index, index2pair 

def read_feature_file_v3(feature_file, exclude_attrs=['id', 'ltable.id', 'rtable.id'], label_attr='label'):

    with open(feature_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', dialect=csv.excel_tab)
        rows = [ row for row in reader]
    
    nrows = len(rows)-1
    if nrows==0:
        return np.array([], dtype=np.float), np.array([], dtype=np.int)
    
    if label_attr not in rows[0]:
        raise Exception('Missing label attribute: '+label_attr)
    
    with open(feature_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', dialect=csv.excel_tab)
        first_row = reader.next()
        
    attributes = list(first_row.keys())
    attributes.remove(label_attr)  
    
    #attributes = set(rows[0])
    #attributes.remove(label_attr)  
    
    for attr in exclude_attrs:
        if attr in attributes:
            attributes.remove(attr)
            
    nfeatures = len(attributes)
    
    ltable_id_index = -1
    rtable_id_index = -1
    label_index = -1
    
    attr2index = {}
    
    for index, attr in enumerate(rows[0]):
        if attr == label_attr:
            label_index = index
        elif attr == 'ltable.id':
            ltable_id_index = index
        elif attr == 'rtable.id':
            rtable_id_index = index
        elif attr in attributes:
            attr2index[attr] = index
            
    if ltable_id_index==-1 or rtable_id_index==-1 or label_index==-1:
        print("IDs or lable attribute is missing...") 
    
    features = np.empty( (nrows, nfeatures), dtype=np.float)
    labels = np.empty(nrows, dtype=np.int)
    
    pair2index = {}
    index2pair = {}
    for index, row in enumerate(rows[1:]):
        labels[index] = int(row[label_index])
        p = (row[ltable_id_index], row[rtable_id_index])
        pair2index[p] = index
        index2pair[index] = p
        for k, attr in enumerate(attributes):
            features[index][k] = float( "{0:.2f}".format(float(row[attr2index[attr]])))
            
    #gc.collect()
    
    return features, labels, pair2index, index2pair 

def read_feature_file_v2(feature_file, exclude_attrs=['id', 'ltable.id', 'rtable.id'], label_attr='label'):
    table = pd.read_csv(feature_file)
    features, labels, pair2index, index2pair =  get_feature_from_df(table, exclude_attrs, label_attr)
    del table
    #gc.collect()
    
    return features, labels, pair2index, index2pair


def get_feature_from_df(table, exclude_attrs=['id', 'ltable.id', 'rtable.id'], label_attr='label'):
    nrows = len(table)
    if nrows==0:
        return np.array([], dtype=np.float), np.array([], dtype=np.int)
    
    attributes = table.columns.values.tolist()
    
    if label_attr not in attributes:
        raise Exception('Missing label attribute: '+label_attr)
    
    attributes.remove(label_attr)  
    
    for attr in exclude_attrs:
        if attr in attributes:
            attributes.remove(attr)
            
    nfeatures = len(attributes)
    
    features = np.empty( (nrows, nfeatures), dtype=np.float)
    labels = np.array(table[label_attr], dtype=np.int)
    
    pair2index = {}
    index2pair = {}
    for index, p in enumerate(zip(table['ltable.id'].values, table['rtable.id'].values)):
        pair = (str(p[0]), str(p[1]))
        pair2index[p] = index
        index2pair[index] = pair  
    
    for k, attr in enumerate(attributes):
        values = table[attr].values
        for index, v in enumerate(values):
            features[index][k] = float( "{0:.2f}".format(float(v)))
    #gc.collect()
        
    return features, labels, pair2index, index2pair     
        
        
def read_golden_label_file(golden_label_file, ltable_id_attr='ltable.id', rtable_id_attr='rtable.id', golden_label_attr='golden'):
    with open(golden_label_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', dialect=csv.excel_tab)
        rows = [ row for row in reader]
        
    pair2golden = {}
    for row in rows:
        p = (row[ltable_id_attr], row[rtable_id_attr])
        pair2golden[p] = int(row[golden_label_attr])
        
    #gc.collect()
        
    return pair2golden
    