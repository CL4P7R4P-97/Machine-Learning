#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('cars.csv')
X = dataset.iloc[:,[3,5]].values

#using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram  = sch.dendrogram(sch.linkage(X, method= 'ward'))
plt.title('Dendrogram')
plt.xlabel('customers')
plt.ylabel('Euclidean distance')
plt.show()

#fitting the hierarchy clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#Visualizing the hierarchy clusters
plt.scatter(X[y_hc ==0,0], X[y_hc ==0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc ==1,0], X[y_hc ==1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc ==2,0], X[y_hc ==2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_hc ==3,0], X[y_hc ==3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc ==4,0], X[y_hc ==4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of cars')
plt.xlabel('HP')
plt.ylabel('Time-to-60')
plt.legend()
plt.show()