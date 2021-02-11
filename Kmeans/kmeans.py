import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')


X = dataset.iloc[:, [3, 4]].values
