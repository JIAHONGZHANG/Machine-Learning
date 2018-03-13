#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 11:26:39 2018

@author: zhangjiahong
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# get dataset.
dataset = pd.read_csv('/Users/zhangjiahong/Documents/Machine Learning A-Z Chinese Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv') 
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# draw graph
plt.scatter(x = X, y = y, color = 'red')
plt.title('Level VS Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
#plt.show()

# build simple linear regressor.
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y)
linear_pred = linear_reg.predict(X)

# get polynomialfeatures. Degree used to be 2.
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
linear_reg_2 = LinearRegression()
linear_reg_2.fit(X_poly, y)
linear_pred_2 = linear_reg_2.predict(X_poly)

# draw graph.
plt.plot(X, linear_pred, color = 'black')
plt.plot(X, linear_pred_2, color = 'blue')
plt.show()
