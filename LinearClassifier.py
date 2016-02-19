import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

# Note: file is comma-delimited
X = np.genfromtxt("data/kaggle.X1.train.txt",delimiter=',')
Y = np.genfromtxt("data/kaggle.Y.train.txt",delimiter=',')
# also load features of the test data (to be predicted)
Xe1= np.genfromtxt("data/kaggle.X1.test.txt",delimiter=',')
print Xe1.shape

lr = ml.linear.linearRegress( X, Y ); # create and train model

p = lr.predict(Xe1);
p = p[:,0]
print p.shape;

import csv

fh = open('predictions.csv','w')    # open file for upload
fh.write('ID,Prediction\n')         # output header line
for i,yi in enumerate(p):
  fh.write('{},{}\n'.format(i+1,yi)) # output each prediction
fh.close()                          # close the file