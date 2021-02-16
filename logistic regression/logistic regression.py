import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')


X = dataset.iloc[:, 2:-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)
