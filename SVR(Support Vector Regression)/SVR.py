#importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset 
dataset = pd.read_excel('best.xlsx')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
y = y[:,np.newaxis]

#feature scaling due to rare usage of SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#fitting the svr to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#predicting the values
y_pred = regressor.predict(X)

#plotting the graph
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.show()