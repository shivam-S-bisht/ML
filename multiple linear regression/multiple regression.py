import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')



X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]



#return dataframe
X = pd.get_dummies(X, columns = ['State'])



#avoiding the dummy variable trap
X = X.drop(columns=['State_New York'])





#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#prediction on test set
y_predict = regressor.predict(X_test)



import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
print(pd.DataFrame(X))

