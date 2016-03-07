import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import mltools.linear
import os 

#os.chdir('/Users/CARA/Desktop/libsvm-3-2.21')

from svmutil import *

y, x = svm_read_problem('data/heart_scale')
X1 = np.genfromtxt("data/kaggle.X1.train.txt",delimiter=',')
Y = np.genfromtxt("data/kaggle.Y.train.txt",delimiter=',')

#print x;
print X1.shape;

n,m = X1.shape;
prob_x = []

for i in range (n):
    xi = {};
    for j in range (m):
        xi[j] = X1[i,j];
    prob_x += [xi];

prob = svm_problem(Y, prob_x)
m = svm_train(prob,'-s 4')
p_label, p_acc, p_val = svm_predict(Y, X1, m)

