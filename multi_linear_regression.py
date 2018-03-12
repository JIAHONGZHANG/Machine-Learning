#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 21:26:52 2018

@author: zhangjiahong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get dataset
dataset = pd.DataFrame(pd.read_csv('50_Startups.csv'))
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# preprocessing.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labeiEncoder = LabelEncoder()
X[:, -1] = labeiEncoder.fit_transform(X[:, -1])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# remove 1 column.
X = X[:, 1:]

# split to training set and test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# fit.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict.
y_pred = regressor.predict(X_test)

# OLS to get P value
import statsmodels.formula.api as sm
X_train = np.append(np.ones((40, 1)), X_train, 1)
X_opt = X_train[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

# If its p value > 0.05, remove it.
X_opt = X_train[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()