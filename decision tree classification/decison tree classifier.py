import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')


X = dataset.iloc[:, 2:-1]
y = dataset.iloc[:, -1]

print(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)


y_predict = pd.DataFrame(classifier.predict(X_test))
