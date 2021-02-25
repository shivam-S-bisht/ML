import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')

print(dataset)

X = dataset.iloc[:, 2:-1]
y = dataset.iloc[:, -1]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

