import pandas as pd
import seaborn as sb
import math
import numpy as np
from numpy import *
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression

points=genfromtxt("D:/Data Science with Python/linear_regression_live-master/linear_regression_live-master/data.csv", delimiter=",")
X=points[:,0]
y=points[:,1]
#X=preprocessing.scale(X)
#y=preprocessing.scale(y)
X=X.reshape(-1, 1)

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.3)
clf=LinearRegression()

clf.fit(X_train,y_train)

Accuracy=clf.score(X_test,y_test)
print(Accuracy)
print('Coefficient: \n', clf.coef_)
print('Intercept: \n', clf.intercept_)