'''
Created on Mar 5, 2019

@author: hzhang0418
'''

import csv

import numpy as np

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
    
    for index, row in enumerate(rows):
        labels[index] = int(row[label_attr])
        for k, attr in enumerate(attributes):
            features[index][k] = float( "{0:.2f}".format(float(row[attr])))
    
    return features, labels 

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
    
    for k, attr in enumerate(attributes):
        values = table[attr].values
        for index, v in enumerate(values):
            features[index][k] = float( "{0:.2f}".format(float(v)))
        
    return features, labels    
        