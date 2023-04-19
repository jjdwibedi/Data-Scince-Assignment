# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:26:12 2023

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing

df= pd.read_csv("C:/Users/USER/Downloads/Fraud_check.csv")
df
df.head()
df.shape
df.info()
df.describe()
label_encoder = preprocessing.LabelEncoder()
df['Undergrad'] = label_encoder.fit_transform(df['Undergrad'])
df['Marital.Status'] = label_encoder.fit_transform(df['Marital.Status'])
df['Urban'] = label_encoder.fit_transform(df['Urban'])
df.info()

df.rename(columns = {'Marital.Status' : 'mar_status', 'Taxable.Income': 'tax_inc', 'City.Population': 'city_pl',
                    'Work.Experience': 'work_exp' }, inplace= True )

#treating those who have taxable_income <= 30000 as "Risky" and others are "Good"

df['status'] = df['tax_inc'].apply(lambda tax_inc: 'Risky' if tax_inc <= 30000 else 'Good')
df.head()
df.drop(['tax_inc'], axis=1, inplace=True)
df['status'] = label_encoder.fit_transform(df['status'])
x = df.iloc[:,0:5]
y = df['status']

x
y
y.value_counts()

#Splitting data into training and test data set
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state=40)

#Building Decision Tree Classifier using Entropy Criteria

model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(x_train, y_train)

#Plot the Decision tree
tree.plot_tree(model);

fn = ['Undergrad', 'Marital.Status', 'Taxable.Income', 'City.Population', 'Work.Experiance', 'Urban']
cn = ['Risky', 'Good']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize= (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn,
               class_names = cn,
               filled = True);

model.get_n_leaves()
#Predicting on test data
preds = model.predict(x_test)
pd.Series(preds).value_counts()

preds
x_test
pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions
# Accuracy 
np.mean(preds==y_test)


from sklearn.metrics import classification_report
print(classification_report(preds,y_test))

#Building Decision Tree Classifier (CART) using Gini Criteria
from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)

model_gini.fit(x_train, y_train)

#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)

#Decision Tree Regression Example
# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

array = df.values
X = array[:,0:3]
y = array[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

#Find the accuracy
model.score(X_test,y_test)




































