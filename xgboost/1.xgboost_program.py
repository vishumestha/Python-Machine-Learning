import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV 


df_data=pd.read_csv("C:/Users/vmestha/Downloads/New folder (5)/train_modified.csv")
y=df_data.loc[:,'Disbursed']
#X=df_data.loc[:,'Disbursed':'Source_2'].drop("ID",axis=1)
#X=df_data.loc[:,'Disbursed':'Source_2']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#X_Dmatrix=xgb.DMatrix(data=X,label=y)

#clf=XGBClassifier()



#params={"objective":"reg:logistic","max_depth":2}
#
#cv_results=xgb.cv(dtrain=X_Dmatrix,params=params,num_boost_round=5,nfold=5,metrics='auc',seed=123,as_pandas=True)
#print(cv_results)
#
#gbm_param={"learning_rate":[0.01],
#            "n_estimators":[100],
#              "max_depth":[ 4],
#              "min_child_weight": [6],
#              "subsample": [0.8],
#              "colsample_bytree": [0.8],
#              "reg_alpha":[0.01]
#              }
#
#
#
#gbm_model=GridSearchCV(estimator=gbm,param_grid=gbm_param,cv=2,scoring='roc_auc',verbose=1)
#gbm_model.fit(X,y)
#xgb.plot_tree(gbm,num_trees=1)
#plt.show()


xgb4 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=50,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
xgb4.fit(X,y)
#xgb.plot_tree(xgb4,num_trees=1)
#plt.show()
#print(gbm_model.best_params_)
#print(gbm_model.best_score_)
#print(gbm_model.scorer_)






