# -*- coding: utf-8 -*-
"""
Created on Sat May  4 11:30:36 2019

@author: Mike
"""

'''
MLPClassifier(hidden_layer_sizes=(100, ), \
              activation=’relu’, \                  #Activation function for the hidden layer.
                                                      ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
                                                      ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
                                                      ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
                                                      ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
              solver=’adam’, \                      #The solver for weight optimization.
                                                      ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
                                                      ‘sgd’ refers to stochastic gradient descent.
                                                      ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
              alpha=0.0001, \                       #L2 penalty (regularization term) parameter.
              batch_size=’auto’, \
              learning_rate=’constant’, \
              learning_rate_init=0.001, \
              power_t=0.5, \
              max_iter=200, \
              shuffle=True, \                       #Whether to shuffle samples in each iteration. Only used when solver=’sgd’ or ‘adam’.
              random_state=None, \
              tol=0.0001, \                         #Tolerance for the optimization. 
                                                      When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, 
                                                      unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.
              verbose=False, \
              warm_start=False, \
              momentum=0.9, \                       #Momentum for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’.
              nesterovs_momentum=True, \
              early_stopping=False, \               #Whether to use early stopping to terminate training when validation score is not improving. 
                                                      If set to true, it will automatically set aside 10% of training data as validation 
                                                      and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. 
                                                      Only effective when solver=’sgd’ or ‘adam’
              validation_fraction=0.1, \
              beta_1=0.9, \
              beta_2=0.999, \
              epsilon=1e-08, \
              n_iter_no_change=10)
'''

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#error rate
def err_rate(pred,test):
    err = 0
    for i in range(len(pred)):
        if pred[i] != test[i]:
            err += 1
    return err/len(pred)

#data preparation
Data = datasets.load_digits()
X = Data.data
Y = Data.target

#training
X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
clf = MLPClassifier(activation = 'relu',solver='adam', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1, max_iter=800)
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)

#error report
err_rate = err_rate(pred, Y_test)
print('Accuracy: ',1-err_rate)

#confusion matrix
cfm = confusion_matrix(Y_test, pred)

#classification report
target_names = ['digit 0', 'digit 1', 'digit 2', 'digit 3', 'digit 4', 'digit 5', 'digit 6', 'digit 7', 'digit 8', 'digit 9']
print(classification_report(Y_test, pred, target_names=target_names))