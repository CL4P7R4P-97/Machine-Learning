#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import datset
dataset = pd.read_csv('cars.csv')
X = dataset.iloc[:,[3,5]].values


#using the elbow methoad to find the optimum no of clusters
from sklearn.cluster import KMeans
wcss = []
for i  in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('the elbow method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

#applying the k-means to the mall dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter  = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Visulizing the clusters
plt.scatter(X[y_kmeans ==0,0], X[y_kmeans ==0, 1], s = 100, c = 'red', label = 'Racing')
plt.scatter(X[y_kmeans ==1,0], X[y_kmeans ==1, 1], s = 100, c = 'blue', label = 'Sports')
plt.scatter(X[y_kmeans ==2,0], X[y_kmeans ==2, 1], s = 100, c = 'green', label = 'Standard')
plt.scatter(X[y_kmeans ==3,0], X[y_kmeans ==3, 1], s = 100, c = 'cyan', label = 'Slowest')
#plt.scatter(X[y_kmeans ==4,0], X[y_kmeans ==4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c='yellow', label = 'Centroids')
plt.title('Clusters of Cars')
plt.xlabel('HP')
plt.ylabel('Time -to-60')
plt.legend()