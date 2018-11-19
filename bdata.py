# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 15:05:28 2018

@author: samka
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

accuracies=[]
for i in range(10):
    df = pd.read_csv('breastdata.txt')
    df.replace('?',-9999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    
    X = np.array(df.drop(['label'], 1 ))
    y = np.array(df['label'])
    
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)
    
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    
    accuracy= clf.score(X_test, y_test)
    print('accuracy:' , accuracy)

#print(x_test)
# =============================================================================
# 
# measure=np.array([[3,1,4,4,4,1,2,2,2], [3,1,2,1,1,1,2,2,2], [3,1,2,1,1,1,2,2,2]])
# measure=measure.reshape(len(measure),-1)
# 
# predict=clf.predict(measure)
# print('prediction:', predict)
# =============================================================================
    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies)) 
    
