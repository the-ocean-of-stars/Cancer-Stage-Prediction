
#import os
import pandas as pd
#import matplotlib.pyplot as plt we didnt use graph
#os.chdir("C:\Users\ASUS\Desktop")
df=pd.read_csv('Cancer_Data.csv')
#print(df.head(10))

X=df.iloc[:,[0,1,2,3,4,6,7,8]]
y=df.iloc[:,9]

X=X.as_matrix()
y=y.as_matrix()
#print(X)
#print(y)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt


regETR=ExtraTreesRegressor(max_depth=11, random_state=33)

regRFR=RandomForestRegressor(max_depth=10, random_state=0)

regDTR= DecisionTreeRegressor(max_depth=10)
regLR=LinearRegression()
#reg=Ridge()
#reg=Lasso()
regGBR=GradientBoostingRegressor()
kf = KFold(n_splits=5,random_state = 33,shuffle = True)
kf.get_n_splits(X,y)

accuracy=[]

for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
        
       regETR.fit(X_train, y_train)
       y_pred = regETR.predict(X_test)

       
       accuracy.append(metrics.r2_score(y_test, y_pred))

print ("Avearage r2 score for ExtraTreesRegressor : ",np.mean(accuracy))


for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
        

       regRFR.fit(X_train, y_train)
       y_pred = regRFR.predict(X_test)  

       
       accuracy.append(metrics.r2_score(y_test, y_pred))

print ("Avearage r2 score for RandomForestRegressor : ",np.mean(accuracy))


for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
        
  
       regDTR.fit(X_train, y_train)
       y_pred = regDTR.predict(X_test)

       accuracy.append(metrics.r2_score(y_test, y_pred))

print ("Avearage r2 score for DecisionTreeRegressor : ",np.mean(accuracy))


for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
        

       regLR.fit(X_train, y_train)
       y_pred = regLR.predict(X_test)

       accuracy.append(metrics.r2_score(y_test, y_pred))

print ("Avearage r2 score LinearRegression: ",np.mean(accuracy))


for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       

       regGBR.fit(X_train, y_train)
       y_pred = regGBR.predict(X_test)
       
       accuracy.append(metrics.r2_score(y_test, y_pred))

print ("Avearage r2 score for radientBoostingRegressor: ",np.mean(accuracy))