#!/usr/bin/env python

from __future__ import division
from numpy import *

def knn(train, train_label, test, test_label, k):
    "train and test k nearest neighbor classifier"
    N = train.shape[0]
    n = test.shape[0]
    y = array([], dtype=int)
    for i in xrange(n):
        # compute distance between ith test data and training example
        diff = sum((train - test[i,:]) ** 2, axis = 1)
        ind = argsort(diff)
        yk = train_label[ind[:k]]
        hist = bincount(yk)
        # tie breaking
        tie = where(hist == hist.max())[0]
        y = append(y, tie[random.randint(tie.size)])
    error = 1 - 1./n * sum(y == test_label)
    return error, y

# load data
train_data = loadtxt("hw2train.txt")
valid_data = loadtxt("hw2validate.txt")
test_data = loadtxt("hw2test.txt")

train = train_data[:,:-1]; train_label = train_data[:,-1].astype(int);
valid = valid_data[:,:-1]; valid_label = valid_data[:,-1].astype(int);
test = test_data[:,:-1]; test_label = test_data[:,-1].astype(int);

# build k-NN classifiers from the training data
K = array([1, 3, 5, 11, 16, 21])
train_err = zeros(6)
valid_err = zeros(6)
counter = 0
for k in K:
    train_err[counter] = knn(train, train_label, train, train_label, k)[0]
    valid_err[counter] = knn(train, train_label, valid, valid_label, k)[0]
    print "For k = %d, training error is %f; validation error is %f." %(k, train_err[counter], valid_err[counter])
    counter += 1

# choose the best k that performs well on the validation data
best_k = valid_err.argmin() + 1

# test on test data with the best k
err = knn(train, train_label, test, test_label, best_k)[0]
print "The test error is %f when k = %d." %(err, best_k)

# For k = 3, construct a 3-NN classifier and its confusion matrix
error, y = knn(train, train_label, test, test_label, 3)
N = bincount(test_label)
C = zeros((10, 10))
n = test.shape[0]
for i in xrange(n):
    C[y[i], test_label[i]] += 1
for j in xrange(10):
    C[:,j] /= N[j]

# highest accuracy for examples that belong to which class
maximum = C.diagonal().argmax()
print "The 3-NN classifier has the highest accuracy for examples that belong to class %d." % maximum

# least accuracy for examples that belong to which class
minimum = C.diagonal().argmin()
print "The 3-NN classifier has the least accuracy for examples that belong to class %d." % minimum

# often mistakenly classified
C_nondiag = C - diag(C.diagonal())
mistake = C_nondiag.argmax()
a = int(str(mistake)[0])
b = int(str(mistake)[1])
print "The 3-NN classifier most lften mistakenly classifies an example in class %d as belonging to class %d." %(b, a)



