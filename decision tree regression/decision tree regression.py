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


