from numpy import *
from scipy.io import loadmat
import numpy as np
data = loadmat('C:/Users/vmestha/Downloads/ex7data1.mat')  
X = data['X']

import  matplotlib.pyplot  as plt
plt.scatter(X[:,0],X[:,1])



def pca(X):  
    # normalize the features
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V
U, S, V=pca(X)

def project_data(X, U, k):  
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)

Z = project_data(X, U, 1)
print(Z)
#plt.figure()
#plt.plot(Z)
