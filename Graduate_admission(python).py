#!/usr/bin/env python
# coding: utf-8

# In[157]:


#downloading dataset, importing libraries

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# In[158]:


#reading the dataset, deleting missing value entries
df = pd.read_csv('Admission_Predict.csv')



df.head()

df.dropna()


# In[159]:


#QUESTIONS:
print("Is this a supervised or unsupervised learning problem?")
print("Is this a regression, classification, or clustering problem?")
print("Is the data linear or non-linear?")
print("Which regression/classification/clustering model is most suitable?")


# In[160]:


#ANSWERS:
print("This is supervised learning, as features and labels are known")
print("This is a regression problem; as the objective is to calculate probability of admission, based on given parameters")
print("This data is linear, with dependent variable(predicted value of y) and independent variables(GRE score, TOEFL score, GPA etc)")
print("Linear regression can be used, as there is a relationship between dependent and independent variables")


# In[161]:


df.columns


# In[162]:


df.drop('Serial No.',axis=1,inplace=True)


# In[163]:


df.columns


# In[164]:


df.describe()


# In[165]:


#SPLITTING VARIABLES
#independent variables
x = df[['GRE Score','TOEFL Score','SOP','CGPA']]


# In[166]:


# dependent variable
y = df[['Chance of Admit ']]


# In[167]:


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0,test_size=.20)


# In[168]:


LinReg = LinearRegression()


# In[169]:


LinReg.fit(x_train,y_train)


# In[170]:


y_predict = LinReg.predict(x_test)


# In[171]:


y_predict[:10]


# In[172]:


y_test[:10]


# In[173]:


#printing Linear Regression error score

LRScore = r2_score(y_test, y_predict)


# In[174]:


print (LRScore)


# In[175]:


#training Random Forest Regressor model


RFR = RandomForestRegressor(n_estimators = 100, random_state = 101)
RFR.fit(x_train,y_train)
y_head_rfr = RFR.predict(x_test)


# In[176]:


#printing Random Forest Regressor error
RFRscore = r2_score(y_test,y_head_rfr)


# In[177]:


y_head_rfr[:10]


# In[178]:


print(RFRscore)


# In[179]:


#training DecisionTreeRegressor model
DTR = DecisionTreeRegressor(random_state = 101)
DTR.fit(x_train,y_train)
y_head_DTR = DTR.predict(x_test) 


# In[180]:


#printing decision tree prediction
y_head_DTR[:10]


# In[181]:


#printing Decision Tree Regression score
DTRscore = r2_score(y_test, y_head_DTR)
print(DTRscore)


# In[182]:


#COMPARISON OF REGRESSOR MODELS AND THEIR ACCURACY SCORES.
y = np.array([LRScore, RFRscore, DTRscore])
x = ["LinearRegression","RandomForestReg.","DecisionTreeReg."]
plt.bar(x,y)
plt.title("Comparison of Regression Algorithms")
plt.xlabel("Regressor")
plt.ylabel("r2_score")
plt.ylim(0.42,0.725)
plt.show()


# In[183]:


# model selection:
print("as per the above visualization(plot), Linear Regression gives highest accuracy score")

