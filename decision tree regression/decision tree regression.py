import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
# print(dataset)

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1]
print(y)

#Decision tree classifier from the lib.
from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier(random_state = 0)
regressor.fit(X, y)



#optimisation for visualisation

#add no.s from min(X) to max(X) with a width of 0.01
X_new = np.arange(min(X), max(X), 0.01)

#add len(X_new) rows in one column
X_new = X_new.reshape(len(X_new), 1)
y_predict = regressor.predict(X_new)
print(X_new)

