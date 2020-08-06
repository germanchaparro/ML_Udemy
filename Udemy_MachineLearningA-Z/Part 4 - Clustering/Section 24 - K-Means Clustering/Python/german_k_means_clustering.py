# K-Means Clustering

#%% Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values   # using square brackets to choose the columns we want to use
# there is no dependent variable so no need to create y.

#%% Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()             # choose 5 as best fit for number of clusters

#%% Training the K-Means model on the dataset
kmeans = KMeans(5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)       # this method creates the label for each record creating the dependent variable


#%% Visualising the clusters
plt.scatter(X[np.where(y_kmeans == 0)[0], 0], X[np.where(y_kmeans == 0), 1], c='blue', label='Cluster 1')
plt.scatter(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], c='blue', marker='x')

plt.scatter(X[np.where(y_kmeans == 1)[0], 0], X[np.where(y_kmeans == 1), 1], c='red', label='Cluster 2')
plt.scatter(kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[1][1], c='red', marker='x')

plt.scatter(X[np.where(y_kmeans == 2)[0], 0], X[np.where(y_kmeans == 2), 1], c='yellow', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[2][0], kmeans.cluster_centers_[2][1], c='yellow', marker='x')

plt.scatter(X[np.where(y_kmeans == 3)[0], 0], X[np.where(y_kmeans == 3), 1], c='green', label='Cluster 4')
plt.scatter(kmeans.cluster_centers_[3][0], kmeans.cluster_centers_[3][1], c='green', marker='x')

plt.scatter(X[np.where(y_kmeans == 4)[0], 0], X[np.where(y_kmeans == 4), 1], c='orange', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[4][0], kmeans.cluster_centers_[4][1], c='orange', marker='x')

plt.title('Clusters of customers')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()

#%% A more optimal way to plot
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='gist_rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c=range(0, 5), cmap='gist_rainbow', marker='x')
plt.title('Clusters of customers')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()
