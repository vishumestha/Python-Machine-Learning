import pandas as pd
import quandl,math,datetime,pickle
import seaborn as sb
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

#Extract the data from quandl
df=quandl.get("WIKI/GOOGL", authtoken="SjWfQrbqTXLCNy7ngaXk")
#Find the corelattion
Corellatpn_data=df.corr()
print(df.info())
#sb.heatmap(Corellatpn_data,square=True)

#Extact the specific columsn from the data frame
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#create the percentage change of the Adj. High and Adj. Low , Adj. Open and Adj. Close
df['HP_PCT']= (df['Adj. High']-df['Adj. Low'])/df['Adj. Low']*100
df['PCT_change']= (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100
df=df[['Adj. Close','HP_PCT','PCT_change']]

forecast_col='Adj. Close'
df.fillna(value=-99999,inplace=True)
forecast_out=int(math.ceil(len(df)*0.01))

df['label']=df[forecast_col].shift(-forecast_out)

X=np.array(df.drop(['label'],1))
X=preprocessing.scale(X)
X_later=X[-forecast_out:]
X=X[:-forecast_out]


df.dropna(inplace=True)
y=np.array(df['label'])


X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)


#clf=LinearRegression()
#clf.fit(X_train,y_train)
#
#with open('linearRegression.pickle','wb') as file:
#    pickle.dump(clf,file)
pickle_in=open('D:/Data Science with Python/ML-Python_Sentex/linearRegression.pickle','rb')
clf=pickle.load(pickle_in)

    
Accuracy=clf.score(X_test,y_test)
print("Accuracy"+str(Accuracy))

for k in ['linear','poly','rbf','sigmoid']:
    clf2=svm.SVR(kernel=k)
    clf2.fit(X_train,y_train)
    Acuracy=clf2.score(X_test,y_test)
    print(k,Acuracy)

forecast_set=clf.predict(X_later)
print(forecast_set,Accuracy,forecast_out)

df['Forecast']=np.nan
last_date=df.iloc[-1].name
last_uinx=last_date.timestamp()
one_day = 86400
next_unix = last_uinx + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



