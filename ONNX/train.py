#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 01:47:56 2022

@author: ak
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()
X,y = iris.data , iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y)
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
joblib.dump(clf,"model.pkl")
