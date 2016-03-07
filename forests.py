import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

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

m,n = X.shape;

Xnew = np.zeros((m,0));
isVisited = [False] * n;
tolerant = 0.3;

running = 0;
errT = np.zeros((running,))
nFolds = 5
errX = np.zeros((running,nFolds))

learners = np.arange(running);
chosen = np.zeros(running);

prechosen = np.loadtxt("chosenIndexes-3.txt");

for c in prechosen:
    c = int(c);
    temp = X[:,c];
    temp = temp.reshape(temp.shape[0],1);
    Xnew = np.append(Xnew[:,], temp, 1);
    isVisited[c] = True;
    
prechosen = np.loadtxt("chosenIndexes-4.txt");

for c in prechosen:
    c = int(c);
    temp = X[:,c];
    temp = temp.reshape(temp.shape[0],1);
    Xnew = np.append(Xnew[:,], temp, 1);
    isVisited[c] = True;
    
prechosen = np.loadtxt("chosenIndexes-5.txt");

for i in range(14):
    c = int(prechosen[i]);
    temp = X[:,c];
    temp = temp.reshape(temp.shape[0],1);
    Xnew = np.append(Xnew[:,], temp, 1);
    isVisited[c] = True;    
    
X = Xnew;

for i in range(nBag):
    Xi, Yi = ml.bootstrapData(X,Y);
    classifiers[i] = ml.dtree.treeRegress(Xi, Yi , maxDepth=20, minParent=1024,nFeatures=60) # Train a model on data Xi, Yi

# test on data Xtest
mTest = Xe1.shape[0]
predict = np.zeros( (mTest, nBag) ) # Allocate space for predictions from each model
for i in range(nBag):
    temp = classifiers[i].predict(Xe1) # Apply each classifier
    predict[:,i] = temp[:,0];
# Make overall prediction by majority vote
p = np.mean(predict, axis=1)
#p = p[:,0]
print p.shape

import csv

fh = open('predictions.csv','w')    # open file for upload
fh.write('ID,Prediction\n')         # output header line
for i,yi in enumerate(p):
  fh.write('{},{}\n'.format(i+1,yi)) # output each prediction
fh.close()                          # close the file