#Multiple Linear Regression

#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,[0,1]].values #now we will have 3 columns as 0,1,2 as index
y = dataset.iloc[:,4].values

#Encoding the categorical variables
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:,3] = labelencoder_X.fit_transform(X[:,3])#we have categorical column as 2
#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()


#Avoiding the dummy variable trap
#X = X[:,1:]


#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = .20,random_state = 0 )


#Fitting the Multiple Linear Regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)  #regressor is fitted to training set

#Predicting the Test set results

y_pred = regressor.predict(X_test)

#using the backward elimination techinique to get best predictor
#import statsmodels.formula.api as sm
#def backwardElimination(x, sl):
#    numVars = len(x[0])
#    for i in range(0, numVars):
#        regressor_OLS = sm.OLS(y, x).fit()
#        maxVar = max(regressor_OLS.pvalues).astype(float)
#        if maxVar > sl:
#            for j in range(0, numVars - i):
#                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                    x = np.delete(x, j, 1)
#    print(regressor_OLS.summary())
#    return x
# 
#SL = 0.05
#X_opt = X[:, [0, 1, 2,3]]
#X_Modeled = backwardElimination(X_opt, SL)

#Backward elimination told us that we need R&D and administration so we will put in X set
#I am commenting this backward elimination portion





