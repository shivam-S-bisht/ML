import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, 2:-1]
y = dataset.iloc[:, -1]



# from sklearn.preprocessing import StandardScaler
# ss_X = StandardScaler()
# X = ss_X.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion = 'entropy', random_state = 0, n_estimators = 10, max_depth=10, max_leaf_nodes=20)
