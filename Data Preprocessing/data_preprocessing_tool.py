#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('Data.csv')

#Dependent Variable
Y = dataset.iloc[:, -1].values

#independent Variable
X = dataset.iloc[:, :-1].values

#print(X)
#print(Y)

#Taking care of Missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

#Encoding data

#Encoding Independent data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encode', OneHotEncoder(), [0])], remainder= 'passthrough')
X = ct.fit_transform(X)

#Encoding Dependent Data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

#Splitting Dataset Into Training Set And Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
"""
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
"""

#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

#print(X_train)
#print(X_test)