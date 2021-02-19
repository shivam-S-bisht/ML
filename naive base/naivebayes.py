import pandas as pd
dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [3, 4]]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
 
#predictions
y_predict = classifier.predict(X_test)
print(y_predict)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_score(y_test, y_predict)
confusion_matrix(y_test, y_predict)
print(classification_report(y_test, y_predict))