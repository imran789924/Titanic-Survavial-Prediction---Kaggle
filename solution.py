#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 20:19:53 2020

@author: imran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('train.csv')
X = dataset.drop('Survived', axis = 1)
y = dataset['Survived']

X.drop(X.columns[[0, 2, 7, 9]], axis = 1, inplace = True)
X = pd.get_dummies(X)
X.drop(X.columns[[6, 9]], axis = 1, inplace = True)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()


'''
features = ["Pclass", "Sex", "SibSp", "Parch"]
Xor = pd.get_dummies(dataset[features])
X_test = pd.get_dummies(test_data[features])
'''
