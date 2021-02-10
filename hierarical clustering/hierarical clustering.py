import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Mall_Customers.csv')

print(dataset)

X = dataset.iloc[:, [3, 4]].values
print(X)

# import scipy.cluster.hierarchy as sch
# dendrogram = sch.dendrogram(sch.linkage(X,method = 'ward'))  
# plt.show()

from sklearn.cluster import AgglomerativeClustering
classifier = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_predict = classifier.fit_predict(X)

# plotting the data
plt.scatter(X[y_predict == 0, 0], X[y_predict == 0, 1], color = 'blue', label = 'careful')
plt.scatter(X[y_predict == 1, 0], X[y_predict == 1, 1], color = 'black', label = 'Standard')
plt.scatter(X[y_predict == 2, 0], X[y_predict == 2, 1], color = 'yellow', label = 'Target')
plt.scatter(X[y_predict == 3, 0], X[y_predict == 3, 1], color = 'red', label = 'Careless')
plt.scatter(X[y_predict == 4, 0], X[y_predict == 4, 1], color = 'green', label = 'Sensible')
plt.legend()
plt.show()