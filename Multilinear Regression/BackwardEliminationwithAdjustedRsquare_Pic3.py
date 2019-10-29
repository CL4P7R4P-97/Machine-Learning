#Data Processing

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset =pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,[0,1,2]].values
y = dataset.iloc[:,4].values


#Encoding the categorical variables
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:,3] = labelencoder_X.fit_transform(X[:,3])
#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()


#Avoiding the dummy variable trap
#X = X[:,1:] 


#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)

#Fitting the model 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)


#Backward Elimination with p-values and Adjusted R Squared:
#import statsmodels.formula.api as sm
#def backwardElimination(x, SL):
#    numVars = len(x[0])
#    temp = np.zeros((50,6)).astype(int) #50 rows and 5 columns but we put 6 as it will take -1
#    for i in range(0, numVars):
#        regressor_OLS = sm.OLS(y, x).fit()
#        maxVar = max(regressor_OLS.pvalues).astype(float)
#        adjR_before = regressor_OLS.rsquared_adj.astype(float)
#        if maxVar > SL:
#            for j in range(0, numVars - i):
#                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                    temp[:,j] = x[:, j]
#                    x = np.delete(x, j, 1)
#                    tmp_regressor = sm.OLS(y, x).fit()
#                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
#                    if (adjR_before >= adjR_after):
#                        x_rollback = np.hstack((x, temp[:,[0,j]]))
#                        x_rollback = np.delete(x_rollback, j, 1)
#                        print (regressor_OLS.summary())
#                        return x_rollback
#                    else:
#                        continue
#    print(regressor_OLS.summary())
#    return x
# 
#SL = 0.05
#X_opt = X[:, [0, 1,2,3,4]] #from X we have to put index of independent variable
#X_Modeled = backwardElimination(X_opt, SL)

#from summary table we get that x3 and x4 are best predictors therefore we will take them