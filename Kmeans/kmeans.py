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
