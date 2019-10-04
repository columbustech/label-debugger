'''
Created on Mar 5, 2019

@author: hzhang0418
'''

class Detector(object):
    
    def __init__(self, features, labels, params):
        self.features = features
        self.labels = labels
        self.params = params
    
    def detect_and_rank(self):
        pass
    
    def use_feedback(self, index2correct_label):
        pass
    
    def set_num_cores(self, num_cores):
        self.num_cores = num_cores