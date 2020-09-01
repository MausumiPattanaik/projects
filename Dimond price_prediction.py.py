# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 00:28:15 2020

@author: mausu
"""

import os
os.chdir(r"D:\DATA SCIENCE CLASS\Machine Learning\project\ ML_MODELS")
os.getcwd()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR  
from  sklearn.neighbors import KNeighborsRegressor

from  sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
import math
#-------------------------------------------------------------------------------


'''____________________________ IMPORT DATASET " dimond.csv " ___________________________ '''


dataset = pd.read_csv("diamonds.csv")

dataset.head()
dataset.info()
dataset.describe()
dataset.columns

''' ___________________________ DATA CLEANING  AND PREPROCESSING ____________________________________'''


#drop the unwanted column unnamed
dataset.drop(["slno"],axis=1,inplace= True)
df= dataset.copy(True)

#cheking missing value
df.isnull().sum()

#unique value count
for i in df.columns:
    print(df[i].value_counts())
    print()

#LABEL  ENCODING:

df.cut  = df.cut.map({ 'Fair':1,'Good':2, 'Very Good':3,'Premium':4,'Ideal': 5 })
df.clarity  = df.clarity.map({ 'SI1':1,'VS2':2, 'SI2':3,'VS1':4,'VVS2': 5,'VVS1': 6, 'IF':7,'I1': 8})
df.color  = df.color.map({ 'G':1,'E':2, 'F':3,'H':4,'D': 5 ,'I': 6 ,'J': 7 })

df.isnull().sum()

# Assign variables
x= df.drop("price",axis =1)
y= df.price

x.describe()
y.describe()


# CORRELATION  MATRIX

corr= df.corr()
sns.heatmap(corr, annot = True,fmt='.1g',cmap= 'coolwarm', linewidths=3, linecolor='black')


#SCALING:
from sklearn.preprocessing import StandardScaler

sscaler = StandardScaler()
sscaler.fit(x) 
x = pd.DataFrame(sscaler.transform(x),columns= x.columns)


# split the data into traingng and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

x_train.shape
x_test.shape


''' __________________________ FEATURE ENGINEERING _________________'''

import statsmodels.api as sm

x2 = sm.add_constant(x_train) # addjust one extra column of constant value 1
ols = sm.OLS(y_train,x2)
lr = ols.fit()
print(lr.summary())


while (lr.pvalues.max()>0.05):
    x2.drop(lr.pvalues.idxmax(),axis=1,inplace=True)
    x_test.drop(lr.pvalues.idxmax(),axis=1,inplace=True)
    ols = sm.OLS(y_train,x2)
    lr = ols.fit()
#drop const from X2 and rename it as X_train
x_train = x2.drop('const',axis=1)
x_train.columns

#----------------------------------------------------------------------------------------------------

'''  ________________________________ MODEL1( LINEAR REGRESSION) _________________________________'''


model1 = LinearRegression()

model1.fit(x_train,y_train)

model1.score(x_train,y_train)
model1.score(x_test,y_test)

b0 = reg.intercept_
b1 = reg.coef_
cross_val_score(model,x,y,cv=4).mean()

''' 
  MODEL SCORE   :  0.9232473981135328
  Test Score    :  0.9317790102607806 
  Crossval_score : 0.9838731334165782
        
 '''

'''' _________GRIDSEARCH ___________ ''''

params = {
                'fit_intercept':[True,False], 
                'normalize':[True,False],
                'copy_X':[True, False]
                }

clf1 =  GridSearchCV( LinearRegression(),params,cv = 5)     
clf1 .fit(x_train,y_train)  
       
clf1.best_params_ 
clf1.best_score_
R2 = clf1.score(x_test,y_test)   
 
''' {'copy_X': True, 'fit_intercept': True, 'normalize': True}
     model score : 0.919893586261469 
     test score  : 0.9317790102607806
 '''

y_pred = clf1.predict(x_test)

mean_abs_error = mean_absolute_error(y_test,y_pred)
mean_sq_error = mean_squared_error(y_test,y_pred)
rmse = math.sqrt(mean_sq_error)
print(rmse,mean_abs_error,b0,R2)

'''
    rmse                :  227.2982554118352
    mean_absolute_error :  166.83206267345176 
    b0                  :  2534.3014559785884 
    r2                  :  0.9317790102607806

    '''

cross_val_score(clf1,x,y,cv=4).mean()

'''ACCURACY : 0.9053583619213256 '''


''' ___________________________MODEL-2( RANDOM FOREST REGRESSOR) _______________________'''



model2 = RandomForestRegressor(n_estimators=1000)
model2 .fit(x_train,y_train)

model2.score(x_train,y_train)
model2.score(x_test,y_test)
cross_val_score(model2, x,y,cv=4).mean() 

'''     model score      : 0.9990465716563692 
        test score       : 0.994908704518591 
        crossvalue_score : 0.9859325805706423
'''

'''' _________GRIDSEARCH ___________ ''''


 params = { 
            "n_estimators"      : [10,20,30],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "bootstrap": [True, False]
            }


clf2 =  GridSearchCV( model2,params,cv = 5)     
clf2.fit(x_train,y_train)  
       
clf2.best_params_ 
clf2.best_score_ 
R2 = clf2.score(x_test,y_test)
'''
best parameters: {'bootstrap': False,
                  'max_features': 'log2',
                  'min_samples_split': 8,
                  'n_estimators': 10} 
    
 training score : 0.9931744785689076
 test score     : 0.9946639880724557  
 
 '''
 y_pred =clf2.predict(x_test)

mean_abs_error = mean_absolute_error(y_test,y_pred)
mean_sq_error = mean_squared_error(y_test,y_pred)

import math

rmse = math.sqrt(mean_sq_error)

print(rmse,mean_abs_error,b0,R2)

'''
    rmse :      63.5690162046322 
    mean_abs_error  : 51.11265142857144 
    b0              : 2534.3014559785884 
    R2              : 0.9946639880724557  
'''

cross_val_score(clf2,x,y,cv=4).mean()

'''ACCURACY :   0.9856457138991128 '''



'''  ______________MODEL3 (SUPPORT VECTOR REGRESSOR :SVR ) ____________________________'''

    
model3  =  SVR ( )

model3.fit(x_train,y_train)

r2 =model3.score(x_train,y_train)
model3.score(x_test,y_test)
cross_val_score(model3, x,y,cv=4).mean()


'''   
model score r2  : -0.036764597072209027 
test score      : -0.06419761727852902
crossvalue_score: -0.08900248032340896 

'''

'''' _________GRIDSEARCH ___________ ''''

params = {
            'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
            'C' : [0.01,0.1,1],
            'degree' : [2,3,4],
            'gamma' : ['0.1','1',0.01],
                      
          }

clf3 =  GridSearchCV(model3 ,params,cv = 5)     
clf3.fit(x_train,y_train)  
       
clf3.best_params_ 
clf3.best_score_ 
R2 = clf3.score(x_test,y_test)
'''  
best parameter   : {'C': 1, 'degree': 2, 'gamma': 0.01, 'kernel': 'linear'}
best model score : 0.585471273748518
best test score  :  0.6789742417340416
 '''

y_pred = clf3.predict(x_test)

mean_abs_error = mean_absolute_error(y_test,y_pred)
mean_sq_error = mean_squared_error(y_test,y_pred)

rmse = math.sqrt(mean_sq_error)
print(rmse,mean_abs_error,R2)

'''
    rmse           : 493.06824623502627 
    mean_abs_error : 300.56802716181863 
    r2             : 0.6789742417340416


'''
cross_val_score(clf3,x,y,cv=4).mean()

''''-0.143595366568386'''



'''  ____________________model4 ( KNN  )  ____________________________'''


model4 =  KNeighborsRegressor(n_neighbors=3,weights="distance",p=1)


#hyper-parameter tuning
best_score = -1000
for k in range(1,21):
    for w in ['uniform','distance']:
        for i in range(1,6):
            reg=KNeighborsRegressor(n_neighbors=k,weights=w,p=i)
            score=cross_val_score(reg, x,y,cv=4).mean()
            if score > best_score:
                best_score = score
                best_params = (k,w,i)

print('Best params for the model: {} with the highest R2 of: {}'.format(best_params,best_score))

'''
best_params : (19, 'uniform', 1) 
best_score  : 0.9847037828464249
R2          : 0.9847037828464249
'''


'''' _________GRIDSEARCH ___________ ''''


params_dict = {
                'n_neighbors': range(1,21),
                'weights': ['uniform','distance'],
                'p': range(1,6)
              }


clf4 = GridSearchCV (KNeighborsRegressor(),params_dict,cv=4)
clf4.fit(x_train,y_train) 
clf4.best_params_
clf4.best_score_
R2 = clf4.score(x_test,y_test)
'''
 Best parameters : {'n_neighbors': 2, 'p': 1, 'weights': 'uniform'} 
 best modelscore : 0.9897800332371895
'''

y_pred = clf4.predict(x_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error

mean_abs_error = mean_absolute_error(y_test,y_pred)
mean_sq_error = mean_squared_error(y_test,y_pred)
rmse = math.sqrt(mean_sq_error)

print(rmse,mean_abs_error,R2)

'''
    rmse            : 71.58271206560049
    mean abs error  : 55.038666666666664 
    R2              : 0.9932338424188195
'''

cross_val_score(clf4,x,y,cv=4).mean()
''' 0.9838731334165782 '''










