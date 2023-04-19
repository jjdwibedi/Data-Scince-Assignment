# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:38:33 2023

@author: USER
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

glass = pd.read_csv("C:/Users/USER/Downloads/glass.csv")
glass
glass.head()
glass.isnull().sum()
glass.duplicated().sum()
glass.drop_duplicates(inplace=True)
glass.duplicated().sum()
glass.Type.value_counts()
glass.shape
glass.info()

#Normalizing data
from sklearn.preprocessing import scale

glass1=glass.iloc[:,:9]
# Converting into numpy array
glass2=glass1.values
# Normalizing the  data 
glass_norm = scale(glass2)

#Train test split
from sklearn.model_selection import train_test_split

x=glass_norm
y=glass['Type']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

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


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 5))
    
plt.title("Types of glasses")
sns.countplot(data=glass, x="Type",palette = "dark")
plt.xticks(rotation = 0, size = 15)
plt.xlabel("Types of Glass", fontsize=12)
plt.ylabel("Count", fontsize=12)

sns.pairplot(glass)





















