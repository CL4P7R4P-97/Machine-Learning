#importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_excel('best.xlsx')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#fitting the dataset with randomForest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators  = 800,random_state = 0)
regressor.fit(X, y)

#predicting the values
y_pred = regressor.predict(11.5)

#plottting the graph
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid) , color = 'blue')
plt.title('temp v/s pressure')
plt.xlabel('temp')
plt.ylabel('pressure')
plt.show()