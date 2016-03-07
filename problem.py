import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import mltools.linear
import os 

os.chdir('/Users/CARA/Desktop/libsvm-3-2.21')

from svmutil import *

#y, x = svm_read_problem('/Users/CARA/Documents/MachineKaggle/data/heart_scale')
X1 = np.genfromtxt("/Users/CARA/Documents/MachineKaggle/data/kaggle.X1.train.txt",delimiter=',')
Y = np.genfromtxt("/Users/CARA/Documents/MachineKaggle/data/kaggle.Y.train.txt",delimiter=',')
prob = svm_problem(Y, X1)
m = svm_train(prob,'-s 4')
p_label, p_acc, p_val = svm_predict(Y, X1, m)

