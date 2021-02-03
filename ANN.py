import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:-1]
y = dataset.iloc[:, -1]
# print(X)

X = pd.get_dummies(data = X, columns = ['Geography', 'Gender'])
# print(X)

from sklearn.preprocessing import StandardScaler
ssc_X = StandardScaler()
X = ssc_X.fit_transform(X)
# print(pd.DataFrame(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = X_train.shape[1], activation = 'relu', input_dim = X_train.shape[1]))
classifier.add(Dense(units = 7, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
