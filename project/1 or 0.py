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

print(X)

# import statsmodels.api as sm
X_opt = X.drop([4,9,5, 6, 12, 13, 8, 7, 2, 10], axis=1)
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# print(regressor_OLS.summary())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)
y_opt = y_test.iloc[:].values
print(y_opt, y_predict)