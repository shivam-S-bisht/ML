import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:-1]
y = dataset.iloc[:, -1]
# print(X)
