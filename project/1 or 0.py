import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data2.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = pd.get_dummies(data = X, columns = ['month'])
X = X.drop(columns = ['month_sep'])

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# y_predict = regressor.predict(X_test)

X = np.append(arr = np.ones((515, 1)).astype(int), values = X, axis = 1)
X = pd.DataFrame(X)
