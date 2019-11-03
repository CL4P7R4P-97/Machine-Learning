#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
corpus = []
for i in range(0,1000):
    
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features = 1500)
X  = cv.fit_transform(corpus).toarray()
y  = dataset.iloc[:, 1:2].values

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0 )

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0 )

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fitting the classifier to the dataset
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 30, random_state = 0, criterion = 'entropy')
classifier.fit(X_train, y_train)

#prediction the test results
y_pred  = classifier.predict(X_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm =  confusion_matrix(y_test, y_pred)
print('Accuracy', end = '  : ')
accuracy = (cm[0,0]+cm[1,1])/200
print(accuracy)
print('Precision', end = ' : ')
precision = (cm[1,1])/(cm[1,1]+cm[1,1])
print(precision)
print('Recall', end = '    : ')
recall = cm[1,1]/(cm[1,1]+cm[0,1])
print(recall)
print('F1 Score', end = '  : ')
f1_score = 2*precision*recall/(precision+recall)
print(f1_score)    