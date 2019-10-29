#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_excel('best.xlsx')
X  = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


#fitting the regressor to the datset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict(11.5) #predicting for 11.5 as example

#visualizing the decisionTreeRegression
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('DecisionTreeRegression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()
