# Hierarchical Clustering

#%% Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values   # using square brackets to choose the columns we want to use
# there is no dependent variable so no need to create y.

#%% Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch                          # using library scipy to paint the dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))   # use the minimum variance method (ward) to determine the clusters
plt.title('Dendrogram')                                        # by checking the dendrogram we can determine the number of clusters (in this case 5)
plt.xlabel('Customers')
plt.ylabel('Euclidian distances')
plt.show()

#%% Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
aggloclust = AgglomerativeClustering(n_clusters = 5, affinity='euclidean' linkage='ward')
y_agglocust = aggloclust.fit_predict(X)


#%% Visualising the clusters
plt.scatter(X[np.where(y_agglocust == 0)[0], 0], X[np.where(y_agglocust == 0), 1], c='blue', label='Cluster 1')
plt.scatter(X[np.where(y_agglocust == 1)[0], 0], X[np.where(y_agglocust == 1), 1], c='red', label='Cluster 2')
plt.scatter(X[np.where(y_agglocust == 2)[0], 0], X[np.where(y_agglocust == 2), 1], c='yellow', label='Cluster 3')
plt.scatter(X[np.where(y_agglocust == 3)[0], 0], X[np.where(y_agglocust == 3), 1], c='green', label='Cluster 4')
plt.scatter(X[np.where(y_agglocust == 4)[0], 0], X[np.where(y_agglocust == 4), 1], c='orange', label='Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()

#%% A more optimal way to plot
plt.scatter(X[:, 0], X[:, 1], c=y_agglocust, cmap='gist_rainbow')
plt.title('Clusters of customers')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.show()