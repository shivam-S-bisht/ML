import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, 2:-1]
y = dataset.iloc[:, -1]



from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X = ss_X.fit_transform(X)

