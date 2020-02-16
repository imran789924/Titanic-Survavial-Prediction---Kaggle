#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:26:30 2020

@author: imran
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('train.csv')
X = dataset.drop('Survived', axis = 1)
y = dataset['Survived'].values


X.drop(X.columns[[0, 2, 7, 8, 9]], axis = 1, inplace = True)
X = pd.get_dummies(X)
X.drop(X.columns[8], axis = 1, inplace = True)


X = X.fillna(X.groupby('Pclass').transform('mean'))

'''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X = imputer.fit_transform(X) 
'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=10)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim=5, init='uniform', activation='relu', input_dim=8))
classifier.add(Dense(output_dim=10, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy' ,metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=20, epochs=2000)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
'''
