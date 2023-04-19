# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:49:35 2023

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

salary_test=pd.read_csv("C:/Users/USER/Downloads/SalaryData_Test.csv")
salary_test
salary_train=pd.read_csv("C:/Users/USER/Downloads/SalaryData_Train.csv")
salary_train
salary_test.head()
salary_train.head()
salary_test.columns
salary_train.columns
salary_test.info()
salary_train.info()
salary_test.info()
salary_train.info()

# Creating a list for categorical data
string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']

from sklearn.preprocessing import LabelEncoder
# from sklearn import preprocessing

number = LabelEncoder()
for i in string_columns:
    salary_train[i]= number.fit_transform(salary_train[i])
    salary_test[i]=number.fit_transform(salary_test[i])
    
salary_test.head()    
salary_train.head()    

#Exploratory Data Analysis (EDA)
salary_train.describe().T
salary_test.describe().T
salary_test.shape
salary_train.shape
salary_test.isnull().sum()
salary_train.isnull().sum()
corr = salary_train.corr()
corr
plt.figure(figsize=(7,7))
sns.heatmap(corr,annot=True)

colnames = salary_train.columns
len(colnames[0:13])
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX  = salary_test[colnames[0:13]]
testY  = salary_test[colnames[13]]

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score,confusion_matrix
sgnb = GaussianNB()
smnb = MultinomialNB()


#Gaussian Naive Bayes
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix_GNB = confusion_matrix(testY,spred_gnb)
confusion_matrix_GNB

print("Accuracy",metrics.accuracy_score(testY,spred_gnb))

#Multinomial naive Bayes
spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix_MNB = confusion_matrix(testY,spred_mnb)
confusion_matrix_MNB

print("Accuracy",metrics.accuracy_score(testY,spred_mnb))



































    