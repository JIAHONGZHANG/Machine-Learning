#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 17:21:28 2018

@author: zhangjiahong
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read csv. But be careful whether X and y are DataFrame or float64
# If DataFrame, use [name].iloc[:, :]
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

# split to train and test.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# make a regressor.
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(np.reshape(X_train, [len(X_train), 1]), y_train)

y_pred = regression.predict(np.reshape(X_test, [len(X_test), 1]))

# show the graph(training set).
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regression.predict(np.reshape(X_train, [len(X_train), 1])))
plt.title('Salary VS Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# show the graph(test set).
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regression.predict(np.reshape(X_train, [len(X_train), 1])))
plt.title('Salary VS Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
