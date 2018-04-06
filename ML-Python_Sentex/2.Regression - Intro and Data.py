import pandas as pd
import quandl 
import seaborn as sb
import math
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression


df=quandl.get("WIKI/GOOGL", authtoken="SjWfQrbqTXLCNy7ngaXk")
Corellatpn_data=df.corr()
print(df.info())
#sb.heatmap(Corellatpn_data,square=True)

df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HP_PCT']= (df['Adj. High']-df['Adj. Low'])/df['Adj. Low']*100
df['PCT_change']= (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100
df=df[['Adj. Close','HP_PCT','PCT_change']]

forecast_col='Adj. Close'
df.fillna(value=-99999,inplace=True)
forecast_out=int(math.ceil(len(df)*0.01))
df['label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X=np.array(df.drop(['label'],1))
y=np.array(df['label'])

X=preprocessing.scale(X)
y=np.array(df['label'])


X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

clf=LinearRegression()
clf.fit(X_train,y_train)
Accuracy=clf.score(X_test,y_test)
print("Accuracy"+str(Accuracy))

for k in ['linear','poly','rbf','sigmoid']:
    clf2=svm.SVR(kernel=k)
    clf2.fit(X_train,y_train)
    Acuracy=clf2.score(X_test,y_test)
    print(k,Acuracy)


