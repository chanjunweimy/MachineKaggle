import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import mltools.linear
import os 

def convertX(X1):
    n,m = X1.shape;

    prob_x = []
    
    for i in range (n):
        xi = {};
        for j in range (m):
            xi[j] = X1[i,j];
        prob_x += [xi];
    return prob_x;

#os.chdir('/Users/CARA/Desktop/libsvm-3-2.21')

from svmutil import *

X1 = np.genfromtxt("data/kaggle.X1.train.txt",delimiter=',')
Y = np.genfromtxt("data/kaggle.Y.train.txt",delimiter=',')
Xte = np.genfromtxt("data/kaggle.X1.test.txt",delimiter=',')
    
prob_x = convertX(X1);
pred_x = convertX(Xte);

y = np.zeros(Xte.shape[0]);
    
prob = svm_problem(Y, prob_x)

print ("start svm training");

m = svm_train(prob,'-s 4 -t 0 -n 0.5 -c 1')
p_label, p_acc, p_val = svm_predict(y, pred_x, m)
# print(len(p_val))
fh = open('predictions0.csv','w')    # open file for upload
fh.write('ID,Prediction\n')         # output header line
for i,fake_yi in enumerate(p_val):
  fh.write('{},{}\n'.format(i+1,fake_yi)) # output each prediction
fh.close()
    