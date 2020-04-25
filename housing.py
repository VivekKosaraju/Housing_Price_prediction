# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:24:16 2020

@author: kosaraju vivek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#step1 : importing dataset
house=pd.read_csv('data.csv')
house=house.iloc[:,:14]
##viewing datasets
print("  viewing head   " )
print(house.head())
print("   information of dataset     ")
print(house.info())
print("   Statistics   ")
print(house.describe())

#Visulizing dataset
print("   Histograms    ")
house.hist(bins=50,figsize=(20,10))
plt.show()
print("  Pairplot  ")
sns.pairplot(house[:])
plt.show()

#fing correlation
print("  Correlation   ")
cor_relation=house.corr()
print(cor_relation['MEDV'].sort_values(ascending=False))
plt.figure(figsize=(10,10))
print("     Heatmap    ")
sns.heatmap(cor_relation,annot=True)
plt.show()

x=house.iloc[:,:13].values
y=house.iloc[:,13].values

#step2: Data pre-processing
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values="NaN",strategy="most_frequent")
x[:,:13]=imp.fit_transform(x[:,:13])
y=imp.fit_transform(y.reshape(506,1))


#step3: Splitting data into traing and testing data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
xtrain = sc_X.fit_transform(xtrain)
xtest = sc_X.transform(xtest)
sc_y = StandardScaler()
ytrain = sc_y.fit_transform(ytrain.reshape(379,1))
ytest = sc_y.fit_transform(ytest.reshape(127,1))

#Step3: find the model

#model1: Linear regression
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(xtrain,ytrain)
ypred_linear=regression.predict(xtest)
print(" Predicted values for Linear regression : ")
print(sc_y.inverse_transform(ypred_linear))


#model2: SVR
from sklearn.svm import SVR
sv=SVR(kernel='rbf')
sv.fit(xtrain,ytrain)
ypred_svm=sv.predict(xtest)
print(" Predicted values for SVR : ")
print(sc_y.inverse_transform(ypred_svm))


#model3: DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor()
tree.fit(xtrain,ytrain)
ypred_tree=tree.predict(xtest)
print(" Predicted values for DecisionTreeRegressor : ")
print(sc_y.inverse_transform(ypred_tree))

#model4: RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=10)
forest.fit(xtrain,ytrain)
ypred_forest=tree.predict(xtest)
print(" Predicted values for RandomForestRegressor : ")
print(sc_y.inverse_transform(ypred_forest))

#Step4: Best model
#using mean square error
from sklearn.metrics import mean_squared_error
print("Error for Linear Regression = ",mean_squared_error(ytest,ypred_linear))
print("Error for SVR = ",mean_squared_error(ytest,ypred_svm))
print("Error for DecisionTreeRegressor = ",mean_squared_error(ytest,ypred_tree))
print("Error for RandomForestRegressor = ",mean_squared_error(ytest,ypred_forest))

