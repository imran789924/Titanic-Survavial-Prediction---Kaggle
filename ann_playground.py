#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:36:08 2020

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



import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

classifier.add(Dense(output_dim=5, init='uniform', activation='relu', input_dim=8))
classifier.add(Dense(output_dim=7, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy' ,metrics=['accuracy'])

classifier.fit(X, y, batch_size=20, epochs=150)

y_pred = classifier.predict(test)
y_pred = (y_pred > 0.5)

#y_pred = (y_pred == 'true')
y_pred2 = 1*y_pred

l = y_pred2.ravel()



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': l})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''