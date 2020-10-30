
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------
#READ THE DATA SET

dataset = pd.read_csv("FuelConsumptionCo2.csv")

dataset.head()
dataset.info()


df= dataset.copy()
df.describe()


#CHEK THE MISSING VALUES
df.isnull().sum()


#DROP THE COLUMN "MODELYEAR"
df.drop("MODELYEAR",axis=1,inplace= True)


#EXTRACT TARGET AND PREDICTOR VARIABLE

y = df. CO2EMISSIONS
x = df.drop(['MAKE','MODEL','VEHICLECLASS','CO2EMISSIONS', 'TRANSMISSION','FUELTYPE'], axis=1)

x.describe()
y.describe()


#SCATTERDENSITY PAIRPLOT:
sns.pairplot(df, diag_kind="kde")

# CORRELATION HEAT MAP:
corr= df.corr()
sns.heatmap(corr, annot = True,fmt='.1g',cmap= 'coolwarm', linewidths=3, linecolor='black')


#SCALING:
from sklearn.preprocessing import StandardScaler

sscaler = StandardScaler()
sscaler.fit(x) 
x_scaled = pd.DataFrame(sscaler.transform(x),columns= x.columns)



# split the data into traingng and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,random_state=0)

x_train.shape
x_test.shape


# feature engineering to eliminate the un wanted variaables from traindata and test data: 

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


# import the model and create the instant
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# trained the model using the training set and cross validate by prdict y value.
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


b0=model.intercept_
b1= model.coef_
R2= model.score(x_train,y_train)

# calculate'mean absolute error,mean squared error  and root mean squared error
from sklearn.metrics import mean_absolute_error,mean_squared_error

mean_abs_error = mean_absolute_error(y_test,y_pred)
mean_sq_error = mean_squared_error(y_test,y_pred)

import math

rmse = math.sqrt(mean_sq_error)

print(rmse,mean_abs_error,b0,b1,R2)


#k-fold cross validation
from sklearn.model_selection import cross_val_score
cross_val_score(model,x,y,cv=4)
cross_val_score(model,x,y,cv=4).mean()


