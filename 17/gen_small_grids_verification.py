import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.neural_network import MLPClassifier
import joblib
import collections
import math
import sys
import time
import pickle
import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



X = joblib.load('vars/X.j')

#print(X)#9]])]
print(type(X), len(X))
print(type(X[0]), len(X[0]))
print(type(X[0][0]), len(X[0][0]))
#print(X.shape)
#X = X*10
#print(X)
#print(type(X[0][0][0]), len(X[0][0][0]))

#print(X[0])
print(type(X[-1]), len(X[-1]))