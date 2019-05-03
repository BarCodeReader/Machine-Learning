# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:26:08 2019

@author: Mike
"""

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split

#error rate
def err_rate(pred,test):
    err = 0
    for i in range(len(pred)):
        if pred[i] != test[i]:
            err += 1
    return err/len(pred)

#data
iris = datasets.load_iris()
iris_data = iris.data
iris_target = iris.target

iris_train, iris_test, label_train,label_test = train_test_split(iris_data, iris_target, test_size=0.2, random_state=42)

#fit data
clf = GaussianNB()
clf.fit(iris_train,label_train)

#predict
pred = clf.predict(iris_test)
proba = clf.predict_proba(iris_test)

#error report
err_rate = err_rate(pred, label_test)
print('error rate: ',err_rate)