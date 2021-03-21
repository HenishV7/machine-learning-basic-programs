#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')

#Independent Variable
X = dataset.iloc[:, :-1].values

#Dependent Variable
Y = dataset.iloc[:, -1].values


#Splitting the dataset into Training Set And Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

"""
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
"""

#Training The Simple Linear Regression Model on The Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Prdicting the Test Set Result
Y_pred = regressor.predict(X_test)

#Visualising the Training Set results
plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary Vs Experiance(Training Set)")
plt.xlabel("Experiance")
plt.ylabel("Salary")
plt.show()

#Visualising the Test Set Results
plt.scatter(X_test, Y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary Vs Experiance(Test Set)")
plt.xlabel("Experiance")
plt.ylabel("Salary")
plt.show()

s = float(input("How much is your Experiance: "))
s_pred = regressor.predict([[s]])
print(f"You might get {s_pred[0]}")