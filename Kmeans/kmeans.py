import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')


X = dataset.iloc[:, [3, 4]].values

from sklearn.cluster import KMeans
wcss = []

# for i in range(1, 11):
#     classifier = KMeans(n_clusters = i, init = 'k-means++', random_state = 0, max_iter = 300, n_init = 10)
#     classifier.fit(X)
#     wcss.append(classifier.inertia_)

# import matplotlib.pyplot as plt
# plt.plot(range(1, 11), wcss)
# plt.title('elbow method')
# plt.xlabel('no. o fclusters')
# plt.ylabel('WCSS') 
# plt.show()  

classifier = KMeans(n_clusters = 5, random_state =0, max_iter = 300, init = 'k-means++', n_init = 10)
y_cluster = classifier.fit_predict(X)
print(classifier.cluster_centers_)

import matplotlib.pyplot as plt
plt.scatter(X[y_cluster == 0, 0], X[y_cluster == 0, 1], color = 'red', label = 'Careful')
plt.scatter(X[y_cluster == 1, 0], X[y_cluster == 1, 1], color = 'blue', label = 'Statndard')
plt.scatter(X[y_cluster == 2, 0], X[y_cluster == 2, 1], color = 'black', label = 'target')
plt.scatter(X[y_cluster == 3, 0], X[y_cluster == 3, 1], color = 'yellow', label = 'careless')
plt.scatter(X[y_cluster == 4, 0], X[y_cluster == 4, 1], color = 'green', label = 'sensible')
plt.scatter(classifier.cluster_centers_[:, 0], classifier.cluster_centers_[:, 1], color = 'purple')
