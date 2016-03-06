import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

X = np.genfromtxt("data/kaggle.X1.train.txt",delimiter=',')
Y = np.genfromtxt("data/kaggle.Y.train.txt",delimiter=',')
Xe1= np.genfromtxt("data/kaggle.X1.test.txt",delimiter=',')
 
depths = np.array(range(0,16))
errT = np.zeros((16,))
errV = np.zeros((16,))


nFolds = 5
errX = np.zeros((16,5))
for d in depths: 
    for iFold in range(nFolds):
        print d;
        [Xti,Xvi,Yti,Yvi] = ml.crossValidate(X,Y,nFolds,iFold)
        lr = ml.dtree.treeRegress( Xti,Yti, maxDepth=d )
        errX[d, iFold] = lr.mse( Xvi,Yvi )

errX = np.mean(errX, axis=1) 
print errX.shape;
print depths.shape;

for d in depths:
    lr = ml.dtree.treeRegress(X, Y, maxDepth=d)
    errT[d] = lr.mse(X,Y);

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.semilogy(depths,errT,'r-', # training error (from P1)
#depths,errV,'g-', # validation error (from P1)
depths,errX,'m-', # cross-validation estimate of validation error
linewidth=2);
plt.axis([0,18,0,1e2]);
plt.show();

for d in depths:
    print d;
    print errT[d];
    print errV[d];
