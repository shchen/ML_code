#!/usr/bin/env python

from __future__ import division
from numpy import *

train = loadtxt("hw3train.txt")
test = loadtxt("hw3test.txt")

train_data = train[:,:-1]
train_label = train[:,-1]
test_data = test[:,:-1]
test_label = test[:,-1]
N = train_data.shape[0]
n = test_data.shape[0]

# preprocess the data such that the hyperplanes
# do not need to pass through the origin
x = column_stack([ones(N), train_data])
x_test = column_stack([ones(n), test_data])

y = train_label.copy()
y[y == 6] = 1
y[y == 0] = -1

y_test = test_label.copy()
y_test[y_test == 6] = 1
y_test[y_test == 0] = -1

### perceptron ###
for i in xrange(N):
    if i == 0:
        w = y[0] * x[0,:]
    elif y[i] * x[i,:].dot(w) <= 0:
        w = w + y[i]*x[i,:]

# test on test data
y_perceptron = sign(x_test.dot(w))
error_perceptron = 1. - 1./x_test.shape[0] * sum(y_perceptron == y_test)

print "The test error of perceptron is ", error_perceptron


### voted perceptron ###
for i in xrange(N):
    if i == 0:
        w_voted = y[0] * x[0,:]
        all_w = w_voted
        m = 1
        c = array([1])
    elif y[i] * x[i,:].dot(w_voted) <= 0:
        w_voted = w_voted + y[i] * x[i,:]
        all_w = column_stack([all_w, w_voted])
        m += 1
        c = append(c, 1)
    else:
        c[-1] += 1

# test on test data
y_voted = sign(sign(x_test.dot(all_w)).dot(c.T))
error_voted = 1. - 1./x_test.shape[0] * sum(y_voted == y_test)

print "The test error of voted perceptron is ", error_voted


### averaged perceptron ###
y_averaged = sign(x_test.dot(all_w.dot(c)))
error_averaged = 1. - 1./x_test.shape[0] * sum(y_averaged == y_test)

print "The test error of averaged perceptron is ", error_averaged
