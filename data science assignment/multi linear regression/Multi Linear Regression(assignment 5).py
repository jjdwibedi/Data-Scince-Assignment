# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 20:04:20 2023

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("C:/Users/USER/Downloads/50_Startups.csv")
df
df.info()

df1 = df.rename({'R&D Spend':'RDS','Administration':'ADMS','Marketing Spend':'MKTS'},axis=1)
df1

df1.corr()

sns.set_style(style='darkgrid')
sns.pairplot(df1)

model=smf.ols("Profit~RDS+ADMS+MKTS",data=df1).fit()

#finding coefficient parameters
model.params

model.tvalues , np.round(model.pvalues,5)

#finding R squared
model.rsquared , model.rsquared_adj  

# Build SLR and MLR models for insignificant variables 'ADMS' and 'MKTS'
# their tvalues and pvalues

slr_a=smf.ols("Profit~ADMS",data=df1).fit()
slr_a.tvalues , slr_a.pvalues  # ADMS has in-significant pvalue

slr_m=smf.ols("Profit~MKTS",data=df1).fit()
slr_m.tvalues , slr_m.pvalues  # MKTS has significant pvalue

mlr_am=smf.ols("Profit~ADMS+MKTS",data=df1).fit()
mlr_am.tvalues , mlr_am.pvalues  # varaibles have significant pvalues

#collinearity check
## Calculate VIF = 1/(1-Rsquare) for all independent variables
rsq_r=smf.ols("RDS~ADMS+MKTS",data=df1).fit().rsquared
vif_r=1/(1-rsq_r)

rsq_a=smf.ols("ADMS~RDS+MKTS",data=df1).fit().rsquared
vif_a=1/(1-rsq_a)

rsq_m=smf.ols("MKTS~RDS+ADMS",data=df1).fit().rsquared
vif_m=1/(1-rsq_m)

#putting valuein data frame format
d1={'Variables':['RDS','ADMS','MKTS'],'Vif':[vif_r,vif_a,vif_m]}
Vif_df=pd.DataFrame(d1)
Vif_df

# None variable has VIF>20, No Collinearity, so consider all varaibles in Regression equation

## there is another technique. i.e Residual Analysis
# Test for Normality of Residuals (Q-Q Plot) using residual model (model.resid)

sm.qqplot(model.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()
list(np.where(model.resid<-30000))

# Test for Homoscedasticity or Heteroscedasticity (plotting model's standardized fitted values vs standardized residual values)

def standard_values(vals) : return (vals-vals.mean())/vals.std()  # User defined z = (x - mu)/sigma
plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show() 

# Test for errors or Residuals Vs Regressors or independent 'x' variables or predictors 
# using Residual Regression Plots code graphics.plot_regress_exog(model,'x',fig)    # exog = x-variable & endog = y-variable

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'RDS',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'ADMS',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'MKTS',fig=fig)
plt.show()

##Checking Outliners Or Influencer
# 1. Cook's Distance: If Cook's distance > 1, then it's an outlier
# Get influencers using cook's distance
(c,_)=model.get_influence().cooks_distance
c
 
# Plot the influencers using the stem plot
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(df1)),np.round(c,5))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()

# Index and value of influencer where C>0.5
np.argmax(c) , np.max(c)

# 2. Leverage Value using High Influence Points : Points beyond Leverage_cutoff value are influencers
influence_plot(model)
plt.show()


# Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features/columns & n = no. of datapoints
k=df1.shape[1]
n=df1.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


df1[df1.index.isin([49])] 

#IMPROVING THE MODEL
# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
df2=df1.drop(df1.index[[49]],axis=0).reset_index(drop=True)
df2


#MODEL DELETION DIOGnostics AND FINAL MODEL

while np.max(c)>0.5 :
    model=smf.ols("Profit~RDS+ADMS+MKTS",data=df2).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    df2=df2.drop(df2.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    df2
else:
    final_model=smf.ols("Profit~RDS+ADMS+MKTS",data=df2).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)

final_model.rsquared
df2

#MODEL PREDICTIONS
# say New data for prediction is
new_data=pd.DataFrame({'RDS':70000,"ADMS":90000,"MKTS":140000},index=[0])
new_data

# Manual Prediction of Price
final_model.predict(new_data)


# Automatic Prediction of Price with 90.02% accurcy
pred_y=final_model.predict(df2)
pred_y


#Table Containting R-SQUARED value for each Prepared Model
d2={'Prep_Models':['Model','Final_Model'],'Rsquared':[model.rsquared,final_model.rsquared]}
table=pd.DataFrame(d2)
table



























