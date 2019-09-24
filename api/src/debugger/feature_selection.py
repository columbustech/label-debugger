'''
Created on Mar 5, 2019

@author: hzhang0418
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def select_features_from_model(features, labels):
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf = clf.fit(features, labels)
    #print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    return model.transform(features)

def select_features(features, labels, alg='none'):
    if alg=='none':
        return features
    elif alg=='model':
        return select_features_from_model(features, labels)
    else:
        raise Exception('Unsupported algorithm: '+alg)