#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('breast_cancer.csv')

#Independene Variable
X = dataset.iloc[:, 1:-1].values

#Dependent Variable
Y = dataset.iloc[:, -1].values

#Taking care of missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy = 'mean')
imputer.fit(X)
X = imputer.transform(X)

#Splitting Dataset into Training Set And Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

#Training the Logistic Regression model on the Training Set 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train, Y_train)

#Predicting new Result
print(classifier.predict([[8,10,10,10,10,8,7,8,3]]))

#Predicting the Test Set Result
Y_pred = classifier.predict(X_test)

print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test),1)),1))

#Making The Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print(accuracy_score(Y_test, Y_pred))

#Computing the accuracy with K-Fold cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X  =X_train, y = Y_train, cv = 10)

print("Accuracies {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation {:.2f} %".format(accuracies.std()*100))