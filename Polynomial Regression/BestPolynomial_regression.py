#importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('aab.txt', sep=',', header=None)
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#we will not split the dataset because we have little information

#fitting the dataset 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

y_pred = lin_reg.predict(X)

#now importing for polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 8)
X_poly = poly_reg.fit_transform(X)
 
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#plotting linear regression
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.xlabel('temperature')
plt.ylabel('Pressure')
plt.show()

#plotting the polynomial regerssion graph
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color = 'red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.xlabel('temperature')
plt.ylabel('pressure')
plt.show()