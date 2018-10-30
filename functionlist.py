# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:33:47 2018

@author: Mike
"""
import pandas as pd
import numpy as np
import math
import random
from collections import Counter
#input: file path, and total column of data(including label column)
#output: data matrix (line by line), label matrix, column of data(minus last column)
def importdata(filepath,numberOfCol):
    file = open(filepath,'r') #read the file
    numberOfLines = len(file.readlines()) #calculate how many lines are in the txt file
    returnMat = np.zeros([numberOfLines,numberOfCol-1]) #create a 0 matrix to contain each line. this is a numpy array.the reason to -1 is to ignore the title line(first line)
    classLabelVector = [] #empty list for overall data
    index = 0
    file = open(filepath,'r') # to let the .readlines() function read from begining again
    for line in file.readlines(): #readlines will return a list of lines
        line = line.strip() #to get each seperate line
        listFromLine = line.split(',') #to cut each line into individual strings
        returnMat[index,:] = listFromLine[0:numberOfCol-1] 
        classLabelVector.append(int(listFromLine[-1])) #this is the vector to store the label, -1 means read from right to left
        index += 1
    return returnMat, classLabelVector, numberOfCol-1

#input: raw data martix, column number of data(minus label)
#output: normalized data matrix
def normalize(dataMat,numberOfCol):
    lendataMat = len(dataMat)
    for i in range(lendataMat):
        maxi = max(dataMat[i])
        mini = min(dataMat[i])
        for j in range(numberOfCol):
            dataMat[i][j] = (dataMat[i][j] - mini)/(maxi - mini)
    return dataMat

#trimed data matrix, label matrix, testing rate
#output: training set, testing set, trainlabel set, test label set. based on the ratio defined
#important: the actual number of elements in testing set will be lesser, this is because when generating random numbers, some of them are the same, thus the unique index in less than our expectation
def trainNtest(dataMat,labelMat, testRate):
    lendataMat = len(dataMat)
    testVolume = math.floor(lendataMat*testRate) #the largest integer less than lendataMat*testRate
    seed = set() #index number
    for i in range(testVolume):
        seed.add(random.randint(0,lendataMat-1)) # -1 is because the end point is included
    index = [i for i in range(lendataMat)]
    remainder = np.delete(index, list(seed))

    testSet = [dataMat[i] for i in seed]
    testLabel = [labelMat[i] for i in seed]
    trainSet = [dataMat[i] for i in remainder]
    trainLabel = [labelMat[i] for i in remainder]
    return trainSet, testSet, trainLabel, testLabel

#input: value K, training set, testing set, training classification, column of attributes, and running mode
#output: the prediction / classification
#important: running mode has 3 types: 
    #hist: this is the standard KNN, judge the result by highest probability
    #dist: this is the distance based judgement, ignore probability, just based on closest distance
    #mix: this is the combination of above 2, when the chance is 50-50, use the distance to judge, when the chance is not 50-50, use probability
def KNN(K, trainSet, testSet,trainLabel, col, stdmode = 'hist'):
    lentrainSet = len(trainSet)
    lentestSet = len(testSet)
    predict = list() #for label prediction
    label = set(trainLabel) #this will give type of labels
    hist = dict()
    for i in range(lentestSet):
        result = list() #for distance
        for j in range(lentrainSet):
            suma = 0
            for k in range(col):
                suma += (testSet[i][k] - trainSet[j][k])**2
            result.append(math.sqrt(suma)) #the distance
        sortarray = np.argsort(result) # index number in ascending arrangement
        karray = [trainLabel[i] for i in sortarray[:K]] #the first K elements in label matrix
        darray = [result[i] for i in sortarray[:K]] #the first K elements in distance matrix
        #print(darray)
        hist = {i:karray.count(i) for i in label} #label is a set with all classification inside
        if stdmode == 'hist': #use the highest portion, i.e. the probability to predict the classification
            test = max([hist[i] for i in hist]) 
            final = [i for i in hist if hist[i] == test][0] #if the probability is 50-50, always use the first one.
            predict.append(final)
        if stdmode == 'dist': #use the mean average distance to determine the classification
            matchTable = {karray[i]:[] for i in range(K)}
            for i in range(K):
                matchTable[karray[i]].append(darray[i])
            mean = {karray[i]:(sum(matchTable[karray[i]])/len(matchTable[karray[i]])) for i in label}
            minimum = min([mean[i] for i in mean])
            result = [i for i in mean if mean[i] == minimum][0]
            predict.append(result)
        if stdmode == 'mix':
            test = max([hist[i] for i in hist]) 
            if test == K/2: #if the probability is 50-50
                matchTable = {karray[i]:[] for i in range(K)}
                for i in range(K):
                    matchTable[karray[i]].append(darray[i])
                mean = {karray[i]:(sum(matchTable[karray[i]])/len(matchTable[karray[i]])) for i in label}
                minimum = min([mean[i] for i in mean])
                result = [i for i in mean if mean[i] == minimum][0]
                predict.append(result)
            else:
                final = [i for i in hist if hist[i] == test][0]
                predict.append(final)
    return predict

#input: predicted list, true value
#output: a dictionary of {True value : Predicted Value}, and an error rate
def errRate(predict, test):
    if len(predict) != len(test):
        print("2 sets dimension not the same: " + str(len(predict)) + " vs " + str(len(test)))
    else:
        wrong = 0
        table = [{predict[i] : test[i]} for i in range(len(test))]
        for i in range(len(test)):
            if predict[i] != test[i]:
                wrong+=1
        #print("True value : Predicted Value")
        #print(table)
        err = wrong/len(test)
        #print("Error Rate: " + str(wrong/len(test)))
        return err