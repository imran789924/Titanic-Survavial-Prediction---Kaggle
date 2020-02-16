#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:05:28 2020

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)



import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim=5, init='uniform', activation='relu', input_dim=8))
classifier.add(Dense(output_dim=10, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy' ,metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=20, epochs=150)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



from sklearn.svm import SVC
classifier = SVC(kernel='rbf', gamma=0.1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()


from sklearn.model_selection import GridSearchCV
parameter = [{'C':[0.5,1], 'kernel': ['rbf'], 'gamma': [0.01,0.1]}]
gs = GridSearchCV(estimator=classifier, param_grid=parameter, scoring='accuracy', cv=10, n_jobs=-1, verbose=100)
gs.fit(X_train, y_train)
print(gs.best_estimator_)
b_param = gs.best_params_
scroes = gs.best_score_


'''
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
'''

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




from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=180, criterion='gini', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



from sklearn.model_selection import GridSearchCV
parameter = [{'n_estimators':[180, 185, 175], 'criterion':['gini','entropy'], 'max_depth':[4, 5, 7]}]
gs = GridSearchCV(estimator=classifier, param_grid=parameter, scoring='accuracy', cv=10, n_jobs=-1, verbose=100)
gs.fit(X_train, y_train)
print(gs.best_estimator_)
b_param = gs.best_params_
scroes = gs.best_score_




'''
features = ["Pclass", "Sex", "SibSp", "Parch"]
Xor = pd.get_dummies(dataset[features])
X_test = pd.get_dummies(test_data[features])
'''
