import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
print(dataset)

X = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2]
print(X)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
X_poly = pd.DataFrame(poly.fit_transform(X))

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_poly, y)

y_predict = regressor.predict(X_poly)

plt.scatter(X, y, color = 'red')
plt.plot(X, y_predict, color = 'blue')
plt.show()