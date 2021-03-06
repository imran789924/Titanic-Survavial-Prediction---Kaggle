#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 20:19:53 2020

@author: imran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_data = pd.read_csv('test.csv')
test = pd.read_csv('test.csv')
dataset = pd.read_csv('train.csv')
X = dataset.drop('Survived', axis = 1)
y = dataset['Survived'].values

test.drop(test.columns[[0, 2, 7, 8, 9]], axis = 1, inplace = True)
test = pd.get_dummies(test)
test.drop(test.columns[8], axis = 1, inplace = True)
X.drop(X.columns[[0, 2, 7, 8, 9]], axis = 1, inplace = True)
X = pd.get_dummies(X)
X.drop(X.columns[8], axis = 1, inplace = True)


X = X.fillna(X.groupby('Pclass').transform('mean'))
test = test.fillna(X.groupby('Pclass').transform('mean'))

'''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X) 
test = imputer.transform(test)
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
test = sc.transform(test)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=180, criterion='gini', random_state=0)
classifier.fit(X, y)
y_pred = classifier.predict(test)

np.count_nonzero(y_pred)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
