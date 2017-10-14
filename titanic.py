# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
data_tr = pd.read_csv("data/train.csv")
data_te = pd.read_csv("data/test.csv")

del data_tr['Name']
del data_tr['Ticket']
del data_tr['Cabin']


# import GridSearchCV
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':('linear','rbf','poly','sigmoid'), 'C':[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

# we are going to choose support vector machine
from sklearn.svm import SVC
svc = SVC()

# set gridsearchcv to svc
clf = GridSearchCV(svc, parameters)

# fit our data
clf.fit(data_tr, data_te)