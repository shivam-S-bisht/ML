import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, 2:-1]
y = dataset.iloc[:, -1]

