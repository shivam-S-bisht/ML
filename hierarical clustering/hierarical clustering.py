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
