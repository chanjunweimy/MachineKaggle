import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import copy

# Note: file is comma-delimited
X = np.genfromtxt("data/kaggle.X1.train.txt",delimiter=',')
Y = np.genfromtxt("data/kaggle.Y.train.txt",delimiter=',')
# also load features of the test data (to be predicted)
Xe1= np.genfromtxt("data/kaggle.X1.test.txt",delimiter=',')

m,n = X.shape;

Xnew = np.zeros((m,0));
isVisited = [False] * n;
tolerant = 0.3;

errT = np.zeros((40,))
nFolds = 5
errX = np.zeros((40,nFolds))

learners = np.arange(40);
chosen = np.zeros(40);

print X.shape
print Y.shape
print learners.shape

for i in range(40):
    best = -1;
    err = -1;
    for j in range (n):
        if isVisited[j]:
            continue;
        temp = X[:,j];
        temp = temp.reshape(temp.shape[0],1);
        Xtemp = np.append(Xnew[:,], temp, 1);
        lr = ml.dtree.treeRegress( Xtemp, Y , maxDepth=20, minParent=1024); # create and train model
        tempErr = lr.mse(X,Y);
        if err == -1:
            err = tempErr;
            best = j;
    if err > tolerant:
        print i;
        print err;
        temp = X[:,best];
        temp = temp.reshape(temp.shape[0],1);
        Xnew = np.append(Xnew[:,], temp, 1);
        errT[i] = err;
        isVisited[best] = True;
        chosen[i] = best;
        for iFold in range(nFolds):
            [Xti,Xvi,Yti,Yvi] = ml.crossValidate(Xnew,Y,nFolds,iFold)
            Xi, Yi = ml.bootstrapData(Xti,Yti);
            classifier = ml.dtree.treeRegress(Xi, Yi , maxDepth=20, minParent=1024) # Train a model on data Xi, Yi
            errX[i, iFold] = classifier.mse(Xvi,Yvi) # Apply each classifier

    else:
        print "can tolerant";
        break;

print Xnew.shape;

errX = np.mean(errX, axis=1);

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.semilogy(learners,errT,'r-', # training error (from P1)
#learners,errV,'g-', # validation error (from P1)
learners,errX,'m-', # cross-validation estimate of validation error
linewidth=2);   
plt.axis([1,25,0,0.5]);
plt.show();

lr = ml.dtree.treeRegress( Xnew, Ynew , maxDepth=20, minParent=1024); # create and train model
p = lr.predict(Xe1);
p = p[:,0]
print p.shape;

import csv

fh = open('predictions.csv','w')    # open file for upload
fh.write('ID,Prediction\n')         # output header line
for i,yi in enumerate(p):
  fh.write('{},{}\n'.format(i+1,yi)) # output each prediction
fh.close()                          # close the file