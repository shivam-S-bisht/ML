import pandas as pd
dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [3, 4]]
y = dataset.iloc[:, -1]
