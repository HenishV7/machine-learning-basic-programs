#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')

#Independent Variable
X = dataset.iloc[:, 1:-1].values

#Dependent Variable
Y = dataset.iloc[:, -1].values

#Training the Linear Regression Model on whole Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)


#Training the Polynomial Regression Model on whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

#Visualising Linear Regression results
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Truth Or Bluff(Linear Modle)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


#Visualising Polynomial Regression results
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg_2.predict(X_poly), color = "blue")
plt.title("Truth Or Bluff(Polynomial Modle)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Visualising Polynomial Regression with Higher degree and resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, Y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Truth Or Bluff(Polynomial Modle)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Predicting new result with Linear Regression
print(lin_reg.predict([[6.5]]))

#Predicting new Result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))