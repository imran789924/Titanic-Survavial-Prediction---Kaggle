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

'''
features = ["Pclass", "Sex", "SibSp", "Parch"]
Xor = pd.get_dummies(dataset[features])
X_test = pd.get_dummies(test_data[features])
'''