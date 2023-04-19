# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:34:20 2023

@author: USER
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report


df= pd.read_csv("C:/Users/USER/Downloads/Company_Data.csv")
df
df.head()
df.shape
df.info()
df.describe()

df['Sales_Efficiency'] = df.Sales.map(lambda x: 'High' if x>8 else 'Low')
df

df1= pd.get_dummies(df,columns=['ShelveLoc','Urban','US'])
df1

feature_cols=['CompPrice','Income','Advertising','Population','Price','Education','ShelveLoc_Bad','ShelveLoc_Good','ShelveLoc_Medium','Urban_No','Urban_Yes','US_No','US_Yes']
X = df1[feature_cols]
Y = df1['Sales_Efficiency']
Y.value_counts()

X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.2,random_state=0)

#Building Decision Tree Classifier using Entropy Criteria

model = DecisionTreeClassifier(criterion = 'entropy',max_depth=5)
model.fit(X_train,Y_train)

model.get_n_leaves()

#PLot the decision tree
tree.plot_tree(model);

preds = model.predict(X_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 

preds

pd.crosstab(Y_test,preds) # getting the 2 way table to understand the correct and wrong predictions
Y_test.value_counts()

# Accuracy 
np.mean(preds==Y_test)

model.score(X_train,Y_train)

from sklearn.metrics import accuracy_score#importing metrics for accuracy calculation (confusion matrix)
print("Accuracy", accuracy_score(Y_test,preds)*100)

print(classification_report(preds,Y_test))


#Building Decision Tree Classifier (CART) using Gini Criteria

from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=6)

model_gini.fit(X_train,Y_train)

#prediction and computing the accuracy
pred=model.predict(X_test)
np.mean(preds==Y_test)

from sklearn.metrics import accuracy_score#importing metrics for accuracy calculation (confusion matrix)
print("Accuracy", accuracy_score(Y_test,preds)*100)


















