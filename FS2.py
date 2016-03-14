import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

def convertX(X1):
    n,m = X1.shape;

    prob_x = []
    
    for i in range (n):
        xi = {};
        for j in range (m):
            xi[j] = X1[i,j];
        prob_x += [xi];
    return prob_x;

from svmutil import *

# Note: file is comma-delimited
X = np.genfromtxt("data/kaggle.X1.train.txt",delimiter=',')
Y = np.genfromtxt("data/kaggle.Y.train.txt",delimiter=',')
# also load features of the test data (to be predicted)
Xe1= np.genfromtxt("data/kaggle.X1.test.txt",delimiter=',')
print X.shape
print Y.shape



nBag = 20;

m,n = X.shape
classifiers = [ None ] * nBag # Allocate space for learners


Xnew = np.zeros((m,0));
tolerant = 0.3;

running = 61;

errT = np.zeros((n,))
nFolds = 5
errX = np.zeros((n,nFolds))
learners = np.arange(n);
prechosen = np.loadtxt("chosenIndexes-3.txt");

i = 0;
for c in prechosen:
    if i >= running:
        break;

    c = int(c);
    temp = X[:,c];
    temp = temp.reshape(temp.shape[0],1);
    Xnew = np.append(Xnew[:,], temp, 1);
    i = i + 1;


    
prechosen = np.loadtxt("chosenIndexes-4.txt");
print i;

for c in prechosen:
    if i >= running:
        break;

    c = int(c);
    temp = X[:,c];
    temp = temp.reshape(temp.shape[0],1);
    Xnew = np.append(Xnew[:,], temp, 1);
    i = i + 1;

print i;
prechosen = np.loadtxt("chosenIndexes-5.txt");

for c in prechosen:
    if i >= running:
        break;
    c = int(c);
    temp = X[:,c];
    temp = temp.reshape(temp.shape[0],1);
    Xnew = np.append(Xnew[:,], temp, 1);
    i = i + 1;

print Xnew.shape;

prob_x = convertX(Xnew);
pred_x = convertX(Xe1);

y = np.zeros(Xe1.shape[0]);
    
prob = svm_problem(Y, prob_x)

print ("start svm training");

m = svm_train(prob,'-s 4 -t 1 -n 0.5 -c 1')
p_label, p_acc, p_val = svm_predict(y, pred_x, m)
# print(len(p_val))
fh = open('predictions0.csv','w')    # open file for upload
fh.write('ID,Prediction\n')         # output header line
for i,fake_yi in enumerate(p_val):
  fh.write('{},{}\n'.format(i+1,fake_yi)) # output each prediction
fh.close()