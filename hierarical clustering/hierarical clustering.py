import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Mall_Customers.csv')

print(dataset)

X = dataset.iloc[:, [3, 4]].values
