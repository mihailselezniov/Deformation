import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import collections
import math
import sys
import time
import pickle
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

source_f = 'ml_threads/6_1.txt'

X = joblib.load('border_vars/X.j')
Y = joblib.load('border_vars/Y.j')
print(X.shape, Y.shape)

n_cors = 6
model = KNeighborsClassifier(n_neighbors=6, n_jobs=n_cors)

with open(source_f, 'r') as f:
    threads = f.readlines()

x_train, y_train = [], []
for t in threads:
    tr = list(map(int, t.replace('\n', '').split(',')))
    x_train.append(tr[:-1])
    y_train.append(tr[-1])
#print(x_train, y_train)



cut = 343
x, y = np.array(x_train[:cut]), np.array(y_train[:cut])
#print(x, y)
model.fit(x, y)
y_pred = model.predict(X)

row_metrics = []

roc = roc_auc_score(Y, y_pred)
row_metrics.append(roc)
print(roc)

pr = average_precision_score(Y, y_pred)
row_metrics.append(pr)
print(pr)

f1 = f1_score(Y, y_pred, average=None)
row_metrics.append(f1[0])
print(f1[0])

row_metrics = ','.join(list(map(lambda x: str(round(float(x), 4)), row_metrics)))
print(row_metrics)



'''
(72900000, 8) (72900000,)
0.594854006137005
0.8951520083067506
0.2675444045542221
'''
