# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:29:13 2018

@author: 0
"""
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


data = pd.read_excel('Cancer_Data.xlsx')
data = pd.DataFrame(data)
print(data)

labels = data['Class']
print(labels)

data=data.drop("Clump_Thickness", 1)
data=data.drop("Uniformity_of_cell_size", 1)
data=data.drop("Uniformity_of_cell_shape", 1)
data=data.drop("Marginal_Adhesion", 1)
data=data.drop("Single_Epithelial_Cell_Size", 1)
data=data.drop("Bare_Nucleoli", 1)
data=data.drop("Bland_Chromatin", 1)
data=data.drop("Normal_Nucleoli", 1)
data=data.drop("Mitoses", 1)

print(data)

x = pd.DataFrame(data)
y = pd.DataFrame(labels)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=4)

clf = linear_model.LinearRegression()

clf.fit(x,y)
pred = clf.predict(x_test)

pred.shape
print(pred)
pd.DataFrame(pred)

print(labels)

plt.scatter(y_test, pred)
plt.xlabel("Class")
plt.ylabel("Predicted Class")
plt.title("Class vs Predicted Class:")

print('Variance score: %.2f' % r2_score(y_test, pred))



