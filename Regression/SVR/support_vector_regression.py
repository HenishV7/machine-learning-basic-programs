#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv("Position_Salaries.csv")

#Independent Variable
X = dataset.iloc[:, 1:-1].values

#Dependent Variable
Y = dataset.iloc[:, -1].values

Y = Y.reshape(len(Y),1)

#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

#Training SVR model on the whole Dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, Y)

#Predicting New Result
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

#Visuallising the SVR result
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title("Truth or Bluff(Support Vector Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#Visualising the SVR Result(for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)),max(sc_X.inverse_transform(X)),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X_grid, sc_Y.inverse_transform(Y), color = 'red')
plt.plot(sc_X.inverse_transform(X_grid), sc_Y.inverse_transform(regressor.predict(X_grid)), color = 'blue')
plt.title("Truth or Bluff(Support Vector Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()