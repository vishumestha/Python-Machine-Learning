import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

#Reading the data from the csv 
#Train data set
train=pd.read_csv("D:/Data Science with Python/project_income_us/train.csv",na_values=[""," ","?","NA"])

#Reading the test data set
test=pd.read_csv("D:/Data Science with Python/project_income_us/test.csv",na_values=[""," ","?","NA"])
#sb.heatmap(train.corr())

#Mapping the labels to 0,1 train data
train.income_level=train.income_level.map({-50000:0,50000:1})
print(train.income_level.value_counts()/len(train.income_level)*100)

#Mapping the labels to 0,1 test data
test.income_level=test.income_level.apply(lambda x: 0 if x=='-50000' else 1)
#test.income_level.map({'-50000':0,'50000+.':1})
print(test.income_level.value_counts()/len(test.income_level)*100)

#print("Null Values in the each features")
print(train.isnull().sum())
#sb.pairplot(train)

#Histogram for various numerical variables
#for x in train._get_numeric_data().columns:
#    sb.distplot(train[x])
#    plt.figure()

#To get the categorical and  numeric values
train_cat=train.iloc[:,[1, 2, 3, 4, 6 ,7, 8,9,10,11,12,13,14,15,19,20,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,39,40]]
num=set(train.columns)- set(train_cat.columns)
train_num=train[sorted(list(num))]
print(train_num.isnull().sum())

#plt.scatter(x=train_num["age"], y=train_num["wage_per_hour"],color=train_cat.income_level)
#sb.regplot(x=train_num["age"], y=train_num["wage_per_hour"],color=train_cat.income_level)
#sb.jointplot(data=train_num, x='age', y='wage_per_hour', kind='reg', color='g')

#sns.plt.show()
#To get the categorical and  numeric values
test_cat=test.iloc[:,[1, 2, 3, 4, 6 ,7, 8,9,10,11,12,13,14,15,19,20,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,39,40]]
num_features=set(test.columns)- set(test_cat.columns)
test_num=test[sorted(list(num_features))]
print(test_num.isnull().sum())

#No Need of below steps
#train_num=train[train._get_numeric_data().columns]
#test_num=test[test._get_numeric_dat().columns]
#To get the categoriacal values
#cat=set(train.columns)- set(train._get_numeric_data().columns)
#train_cat=train[list(cat)]


#Exclude the features with more than 0.7 corelation
import numpy as np 
r = np.corrcoef(train_num.T)
#plt.imshow(r, cmap = "coolwarm")
#plt.colorbar();

idx = np.abs(np.tril(r, k= -1)) < .7
idx_drop = np.all(idx, axis=1)
train_num = train_num.iloc[:,idx_drop]
#The train set will eliminate weeks_worked_in_year
#so remove this column from test data as well
##Selecting the only the numerical columns which are used for the training
test_num=test_num[train_num.columns]



print(train_cat.isnull().sum())
#Percetage of the categorical missing values
missing_percentage=train_cat.apply(lambda x: ((x.isnull().sum())/(len(train_cat)))*100, axis=0)
#test_num = test_num.iloc[:,idx_drop]

#Removing the columns with more than 49% of the missing values
missing_percentage=missing_percentage[missing_percentage.apply(lambda x : x <5)].index
train_cat=train_cat[missing_percentage]
train_cat=train_cat.fillna("Unavailable",axis=0)


print(test_cat.isnull().sum())
#Selecting the only the categorical columns which are used for the training
test_cat=test_cat[train_cat.columns]

#values count less then 0.05 should be filled with 'Other' value
p=5/100  
for x in train_cat.columns:
    series = pd.value_counts(train_cat[x])
    mask = (series/series.sum() * 100)<5
    train_cat[x] = np.where(train_cat[x].isin(series[mask].index),'Other',train_cat[x])

    
for x in test_cat.columns:
    series = pd.value_counts(test_cat[x])
    mask = (series/series.sum() * 100)<5
    test_cat[x] = np.where(test_cat[x].isin(series[mask].index),'Other',test_cat[x])
#Keep all the values in each columss of the test data as same as train data
#In train data less frequency values are removed 
#for col in test_cat.columns:
#    train_col_values=list(train_cat[col].unique())
#    #print(type(train_col_values),train_col_values)
#    test_cat[col]=test_cat[col].apply(lambda x : str(x) if str(x) in train_col_values else 'Other')


    
#bin age variable 0-30 31-60 61 - 90
#lambda symbol: 'X' if symbol==True else 'O' if symbol==False else ' '
#training data
train_num.age=train_num.age.apply(lambda x :"0-30" if x <=30 else "31-60" if x <=60  else "61-90" )

#Bin numeric variables with Zero and MoreThanZero
train_num.wage_per_hour=train_num.wage_per_hour.apply(lambda x : "zero" if x==0 else "MoreThanZero" )
train_num.capital_gains=train_num.capital_gains.apply(lambda x : "zero" if x==0 else "MoreThanZero" )
train_num.capital_losses=train_num.capital_losses.apply(lambda x : "zero" if x==0 else "MoreThanZero" )
train_num.dividend_from_Stocks=train_num.dividend_from_Stocks.apply(lambda x : "zero" if x==0 else "MoreThanZero" )



#Same steps as above 
test_num.age=test_num.age.apply(lambda x :"0-30" if x <=30 else "31-60" if x <=60  else "61-90" )
test_num.wage_per_hour=test_num.wage_per_hour.apply(lambda x : "zero" if x==0 else "MoreThanZero" )
test_num.capital_gains=test_num.capital_gains.apply(lambda x : "zero" if x==0 else "MoreThanZero" )
test_num.capital_losses=test_num.capital_losses.apply(lambda x : "zero" if x==0 else "MoreThanZero" )
test_num.dividend_from_Stocks=test_num.dividend_from_Stocks.apply(lambda x : "zero" if x==0 else "MoreThanZero" )

#combing the categorical and numerical data of training data set
frames=[train_cat,train_num]
train=pd.concat(frames,axis=1)


#combing the categorical and numerical data of testing data  set
frames=[test_cat,test_num]
test=pd.concat(frames,axis=1)

#Selecting the target variable and dependent  variables of training data
train_y=train.loc[:,['income_level']]
train_y=train_y['income_level'].astype('category').cat.codes
train_X=train.drop('income_level',axis=1)
#train_X=pd.get_dummies(train_X)


#Selecting the target variable and dependent  variables of training data
test_y=test.loc[:,['income_level']]
test_y=test_y['income_level'].astype('category').cat.codes
test_X=test.drop('income_level',axis=1)

#num_data=dataset_training.select_dtypes(include=['number'])
#cat_data=dataset_training.select_dtypes(exclude=['number']) 

#encoding the only the categorical data of training data set
cat_X_columns=train_X.select_dtypes(exclude=['number']).columns
for x in cat_X_columns:
    train_X[x]=train_X[x].astype('category').cat.codes

#encoding the only the categorical data of testing data set
cat_X_columns=test_X.select_dtypes(exclude=['number']).columns
for x in cat_X_columns:
    test_X[x]=test_X[x].astype('category').cat.codes

#zero variance removal

#from sklearn.feature_selection import VarianceThreshold
#var=VarianceThreshold()
#
#train_X=var.fit_transform(train_X)

#train_X[train_X._get_numeric_data().columns]=var.fit_transform(train_X._get_numeric_data())

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
clf=ExtraTreeClassifier(criterion="entropy",random_state=100)
#clf=RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=100)
clf.fit(train_X,train_y)
print(clf.feature_importances_)
    
for name, importance in zip(train_X.columns, clf.feature_importances_):
    print(name, "=", importance)
    
imp_datafram=pd.DataFrame(list(zip(train_X.columns, clf.feature_importances_)))

#import matplotlib.pyplot as plt
#
#y_pos = np.arange(len(imp_datafram[0]))
#performance = imp_datafram[1]
# 
#plt.bar(y_pos, performance, align='center', alpha=0.5)
#plt.xticks(y_pos, imp_datafram[0])

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
def cross_validaion(headline,model,Xtrain,y_train):
    print(headline)
    cros=cross_val_score(estimator=model,X=Xtrain,y=y_train,cv=10)
    print("cross {}".format(cros.mean()))
    #print("cross {}".format(cros))

from imblearn.under_sampling import RandomUnderSampler
# Apply the random under-sampling
rus = RandomUnderSampler(return_indices=True)
X_under_resampled, y_under_resampled, idx_resampled_under = rus.fit_sample(train_X, train_y)
from collections import Counter
print('Resampled dataset shape RandomUnderSampler {}'.format(Counter(y_under_resampled)))


from imblearn.over_sampling import RandomOverSampler
rus = RandomOverSampler(random_state=42)
X_over_resampled, y_over_resampled = rus.fit_sample(train_X, train_y)
from collections import Counter
print('Resampled dataset shape RandomOverSampler {}'.format(Counter(y_over_resampled)))



from imblearn.over_sampling import SMOTE
sm = SMOTE(ratio='minority',random_state=42,)
X_smote_res, y_smote_res = sm.fit_sample(train_X, train_y)
from collections import Counter
print('Resampled dataset shape SMOTE {}'.format(Counter(y_smote_res)))


from sklearn.naive_bayes import GaussianNB
classifier_cv=GaussianNB()
cross_validaion('RandomUnderSampler',classifier_cv,X_under_resampled,y_under_resampled)
cross_validaion('RandomOverSampler',classifier_cv,X_over_resampled,y_over_resampled)
cross_validaion('SMOTE',classifier_cv,X_smote_res,y_smote_res)
cross_validaion('Normal',classifier_cv,train_X, train_y)

def print_results(headline, true_value, pred):
    print(headline)
    print("confusion matrix {}".format(confusion_matrix(true_value, pred)))
    tn, fp, fn, tp = confusion_matrix(true_value, pred).ravel()
    print(tn, fp, fn, tp)
    specificity=tn/(tn+fp)
    false_true=tn/(tn+fn)
    print('specificity')
    print(specificity)
    print('false_true')
    print(false_true)
        
    
    print("accuracy: {}".format(accuracy_score(true_value, pred)))
    print("precision: {}".format(precision_score(true_value, pred)))
    print("recall: {}".format(recall_score(true_value, pred)))
    print("f1: {}".format(f1_score(true_value, pred))) 

classifier=GaussianNB().fit(X_smote_res,y_smote_res) 
y_smote_predict=classifier.predict(test_X)
print_results("smote",test_y,y_smote_predict)
#
#
#import xgboost as xgb
#xgb=
    



#X_smote_res_train,X_smote_res_test,y_smote_res_train,y_smote_res_test=train_test_split(X_smote_res,y_smote_res,test_size=0.3)
#classifier=GaussianNB().fit(X_smote_res_train,y_smote_res_train)
#y_smote_res_predict=classifier.predict(X_smote_res_test)
#print_results("smote",y_smote_res_test,y_smote_res_predict)
#
#
#X_over_res_train,X_over_res_test,y_over_res_train,y_over_res_test=train_test_split(X_over_resampled,y_over_resampled,test_size=0.3)
#classifier=GaussianNB().fit(X_over_res_train,y_over_res_train)
#y_over_res_predict=classifier.predict(X_over_res_test)
#print_results("over",y_over_res_test,y_over_res_predict)
#
#X_under_res_train,X_under_res_test,y_under_res_train,y_under_res_test=train_test_split(X_under_resampled,y_under_resampled,test_size=0.3)
#classifier=GaussianNB().fit(X_under_res_train,y_under_res_train)
#y_under_res_predict=classifier.predict(X_under_res_test)
#print_results("under",y_under_res_test,y_under_res_predict)
#
#X_train,X_test,y_train,y_test=train_test_split(train_X, train_y,test_size=0.3)
#classifier=GaussianNB().fit(X_train,y_train)
#y_predict=classifier.predict(X_test)
#print_results("normal",y_test,y_predict)
#----------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn 
from sklearn.model_selection import GridSearchCV
target ='income_level'

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print( "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
#Choose all predictors except target & IDcols

#predictors = X_smote_res.c

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
#xgtrain = xgb.DMatrix(X_smote_res, y_smote_res)
#
#cv_folds=5
#early_stopping_rounds=50
#xgb_param = xgb1.get_xgb_params()
#cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=cv_folds,
#            metrics='auc', early_stopping_rounds=early_stopping_rounds)

#xgb1.set_params(n_estimators=cvresult.shape[0])

xgb1.fit(train_X, train_y)


y_pre=xgb1.predict(test_X)#.as_matrix())

print_results("smote",test_y,y_pre)
#modelfit(xgb1, train, predictors)

xgb1 = XGBClassifier(
 n_estimators=1000,
 gamma=0,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

from sklearn.model_selection import RandomizedSearchCV
param={'max_depth':np.arange(3,11)}
#param={'max_depth':np.arange(3,11),'reg_lambda':np.arange(0.05,0.5,0.1),'learning_rate':np.arange(0.1,0.5,0.1),'subsample':np.arange(0.50,1,0.10),'min_child_weight':np.arange(2,10,1),'colsample_bytree':np.arange(0.50,0.8,0.1)}
model=RandomizedSearchCV(estimator=xgb1,param_distributions=param,cv=10,n_iter=8)
#model.fit(X_smote_res, y_smote_res)
model.fit(train_X, train_y)



#y_pre=model.predict((test_X)).as_matrix())

print_results("smote",test_y,y_pre)



import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
xg_model= XGBClassifier(max_depth=10,n_estimators=1000)
xg_model.fit(train_X, train_y)
y_pre=xg_model.predict((test_X))
print_results("smote",test_y,y_pre)
#----------------------------------------------------------------------------------------------------------

def modelfit(alg, train_X, train_y, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(train_X, train_y)
        
    #Predict training set:
    dtrain_predictions = alg.predict(train_X)
    dtrain_predprob = alg.predict_proba(train_X)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, train_X, train_y, cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print ("\nModel Report")
    print( "Accuracy : %.4g" % metrics.accuracy_score(train_y, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, dtrain_predprob))
    
    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, train_X.columns).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, train_X, train_y)


param_test1 = {'n_estimators':range(30,100,10)}

gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.2, min_samples_split=750,min_samples_leaf=25,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10,verbose=2), 
param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train_X, train_y)


param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=100,learning_rate=0.2,min_samples_leaf=25,max_features='sqrt',subsample=0.8,random_state=10,verbose=2), 
param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train_X, train_y)
#---------------------------------------------------------------------------------------------------

import xgboost as xgb
from  xgboost.sklearn import XGBClassifier

def modelfit(alg, train_x, train_y,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train_x, label=train_y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(train_x, train_y,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(train_x)
    dtrain_predprob = alg.predict_proba(train_x)[:,1]
    #Predict testing set:
    dtrain_predictions = alg.predict(test_X.as_matrix())
    dtrain_predprob = alg.predict_proba(test_X.as_matrix())[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print( "Accuracy : %.4g" % metrics.accuracy_score(test_y.values, dtrain_predictions))
    print( "AUC Score (Train): %f" % metrics.roc_auc_score(test_y.values, dtrain_predprob))
    print("confusion matrix")
    print_results("smote",test_y.values, dtrain_predictions)
                    
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')

xgb1 = XGBClassifier(
 learning_rate =0.8,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

modelfit(xgb1, X_smote_res,y_smote_res)

import pytesseract



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model=DecisionTreeClassifier(random_state=100)
model.fit(X_smote_res,y_smote_res)
dtrain_predictions=model.predict(test_X.as_matrix())
print_results("smote",test_y.values, dtrain_predictions)

from  sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(model, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("testing")


#from  sklearn.ensemble import RandomForestClassifier
#from boruta import BorutaPy
#rfc = RandomForestClassifier(n_estimators=200, n_jobs=4, class_weight='balanced', max_depth=6)
#boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
#boruta_selector.fit(X, y)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# load X and y
# NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
X = train_X.values
y = train_y.values
y = y.ravel()

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features - 5 features should be selected
feat_selector.fit(X, y)

# check selected features - first 5 features are selected
feat_selector.support_

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)


























