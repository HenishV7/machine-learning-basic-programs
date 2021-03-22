#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('50_Startups.csv')

#Independent Data
X = dataset.iloc[:, :-1].values

#Dependent Data
Y = dataset.iloc[:, -1].values

#Encoding Independent Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('encode', OneHotEncoder(), [3])], remainder= 'passthrough')
X = ct.fit_transform(X)

X = X[:, 1:]

#Splitting the Dataset into Training Set And Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

#Training the Multiple Linear Regression on the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the Test Set Result
Y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

#Performing Backward Elimination

import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#Check p value by running "regrssor_OLS.summary()" and remove highest p valued column from X_opt and repeat this steps till you get value of p= 0.05
X_opt = X[:, [0,1,2,4,5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
X_opt = X[:, [0,2,4,5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
X_opt = X[:, [0,4,5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
X_opt = X[:, [4,5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
