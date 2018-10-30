# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import functionlist as fl
import math
import random
from collections import Counter
import matplotlib.pyplot as plt
#import data
dataMat, labelMat, col = fl.importdata('KNN.txt',11)
#nomalize data
dataMat = fl.normalize(dataMat,col)
#random picking test data
testRate = 0.3 #use 10% of the data to test model
trainSet, testSet, trainLabel, testLabel = fl.trainNtest(dataMat,labelMat, testRate)
#calculate result

error = list()
for i in range(1,50):
    err = 0
    for j in range(1,10):
        predictLabel = fl.KNN(i, trainSet, testSet,trainLabel,col, stdmode='mix') # K is 10, col means the total colume number of data set(exclude label)
        #error rate
        err += fl.errRate(predictLabel, testLabel)
    error.append(err/10)
    
plt.plot(range(1,50),error,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
'''
predictLabel = fl.KNN(24, trainSet, testSet,trainLabel,col, stdmode='mix')
err = fl.errRate(predictLabel, testLabel)
'''