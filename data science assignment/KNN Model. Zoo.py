# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:16:53 2023

@author: USER
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

zoo=pd.read_csv("C:/Users/USER/Downloads/Zoo.csv")
zoo
zoo.head()
zoo.isnull().sum()
zoo.duplicated().sum()
zoo.shape
zoo.info()
zoo1=zoo.drop(['animal name'],axis=1)
zoo1.head()
zoo1.shape


#Train test split

x=zoo1.iloc[:,0:16]
y=zoo1.iloc[:,16]

from sklearn.model_selection import train_test_split

# Testing data in to training and testing 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

neigh=np.array(range(1,30))
param_grid=dict(n_neighbors=neigh)

from sklearn.model_selection import train_test_split,GridSearchCV
# Creating the model
model=KNeighborsClassifier()
grid=GridSearchCV(estimator=model,param_grid=param_grid)
grid.fit(x_train,y_train)

# predicting the values
y_pred=grid.predict(x_test)
y_pred

# calculating accuracy
np.mean(y_pred==y_test)

#KNN (K Neighrest Neighbour Classifier)

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

#Visualizing the CV results

k_range = range(1, 33)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    train_scores = cross_val_score(knn, x_train, y_train, cv=5)
    k_scores.append(train_scores.mean())
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

model = KNeighborsClassifier(n_neighbors=2)
model.fit(x_train, y_train)
model.score(x_test, y_test)
k_scores

#Grid Search for Algorithm Tuning

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

n_neighbors = np.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors)

model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(x_train, y_train)

print(grid.best_score_)
print(grid.best_params_)









