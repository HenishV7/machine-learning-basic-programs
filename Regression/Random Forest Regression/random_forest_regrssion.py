#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv("Position_Salaries.csv")

#Independent Variable
X = dataset.iloc[:, 1:-1].values

#Dependent Variable 
Y = dataset.iloc[:, -1].values

#Training the Random Forest Regression model on Whole Dataset
from sklearn.ensemble import RandomForestRegressor
r_forest_regressor = RandomForestRegressor(n_estimators = 10)
r_forest_regressor.fit(X, Y)

#Predicting a new result
print(r_forest_regressor.predict([[6.5]]))

#Visualising The random Forest Regression Model Result
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, Y, color= 'red')
plt.plot(X_grid, r_forest_regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff(Random Forest Regression)")
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()