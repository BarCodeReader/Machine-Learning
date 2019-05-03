# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:24:41 2019

@author: Mike
"""

#SVM
#The implementation is based on libsvm. 
#The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples.
'''
svm.SVC(C=1.0,  \                                # Penalty parameter C of the error term.
        kernel='rbf',  \                         # It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable. If none is given, 'rbf' will be used.
        degree=3,  \                             # Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
        gamma='auto_deprecated', \               # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        coef0=0.0, \
        shrinking=True,  \
        probability=False, \
        tol=0.001,\
        cache_size=200, \                        # Specify the size of the kernel cache (in MB).
        class_weight=None, \
        verbose=False, \
        max_iter=-1, \
        decision_function_shape='ovr', \
        random_state=None)                       # The seed of the pseudo random number generator used when shuffling the data for probability estimates.
'''
#Similar to SVC but uses a parameter to control the number of support vectors.
'''
svm.NuSVC(nu=0.5,  \                             # An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1].
          kernel=’rbf’,  \
          degree=3,  \
          gamma=’auto_deprecated’,  \
          coef0=0.0,  \
          shrinking=True,  \
          probability=False,  \
          tol=0.001,  \
          cache_size=200,  \
          class_weight=None,  \
          verbose=False,  \
          max_iter=-1,  \
          decision_function_shape=’ovr’,  \
          random_state=None)
'''
#Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, 
#so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
'''
svm.LinearSVC(penalty=’l2’,  \                   # Specifies the norm used in the penalization. The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads to coef_ vectors that are sparse.
              loss=’squared_hinge’,  \
              dual=True,  \
              tol=0.0001,  \
              C=1.0,  \
              multi_class=’ovr’,  \
              fit_intercept=True,  \
              intercept_scaling=1,  \
              class_weight=None,  \
              verbose=0,  \
              random_state=None,  \
              max_iter=1000)
'''
#====================================================================================================================================
from sklearn import svm
from sklearn.model_selection import train_test_split


#data
from sklearn import datasets
iris = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target

iris_train,iris_test,train_label,test_label = train_test_split(iris_data, iris_label, test_size=0.2, random_state=42)

#SVM model

clf = svm.SVC(gamma = 'scale')
clf.fit(iris_train,train_label)
pred = clf.predict(iris_test)

#error rate
err = 0
for i in range(len(pred)):
    if pred[i] != test_label[i]:
        err += 1
        
print('error rate: ',err/len(pred))
