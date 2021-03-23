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

#Training Decision Tree Regression Model on Whole Dataset
from sklearn.tree import DecisionTreeRegressor
d_tree_regressor = DecisionTreeRegressor()
d_tree_regressor.fit(X, Y)

#Predicting New Results
print(d_tree_regressor.predict([[6.5]]))

#Visualising the Decision Tree Regression Model on the whole Dataset
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y, color='red')
plt.plot(X_grid, d_tree_regressor.predict(X_grid), color='blue')
plt.title("Truth or Bluff(Decision Tree Regression model)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()