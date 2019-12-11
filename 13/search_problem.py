# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import collections


par = {}
parts = 10
par['pressure_time'] = {'Min': 0.0, 'Max': 100.0}
par['pressure_radius'] = {'Min': 0.0, 'Max': 5.0}
par['pressure_amplitude'] = {'Min': 0.0, 'Max': 200.0}
par['length'] = {'Min': 10.0, 'Max': 100.0}
par['diameter'] = {'Min': 0.01, 'Max': 0.5}
par['young'] = {'Min': 60.0, 'Max': 300.0}
par['density'] = {'Min': 1000.0, 'Max': 2000.0}
par['strength'] = {'Min': 0.2, 'Max': 10.0}

def get_list(Min, Max):
    return list(map(lambda x: round(x, 2), np.arange(Min, Max+0.01, (Max-Min)/(parts-1))))


print(get_list(**par['length']))
print(get_list(**par['diameter']))
print(get_list(**par['young']))
print(get_list(**par['density']))
print(get_list(**par['pressure_time']))
print(get_list(**par['pressure_radius']))
print(get_list(**par['pressure_amplitude']))
print(get_list(**par['strength']))


with open('../11/fib_all_data.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))

def make_str(data):
    return ''.join(map(str, data))

X, Y = [], []
for i, val in enumerate(data_is_broken):
    Y.extend([i%2]*val)

n = tuple(map(float, range(10)))
i = 0
y_ = []
a = np.empty((0,8), dtype=np.float64)
for i0 in n:
    for i1 in n:
        for i2 in n:
            for i3 in n:
                for i4 in n:
                    for i5 in n:
                        for i6 in n:
                            for i7 in n:
                                if 0 not in [i4, i5, i6]:
                                    X.append([i0, i1, i2, i3, i4, i5, i6, i7])
                                    y_.append(Y[i])
                                i += 1
    a = np.append(a, np.array(X), axis=0)
    X = []
    print(i0)

print('\n', '-'*10, 'index', '-'*10)
for i in range(20):
    print(X[i])

X = preprocessing.normalize(X)
print('\n', '-'*10, 'index + normalize', '-'*10)
for i in range(20):
    print(X[i])




X, Y = [], []
for i, val in enumerate(data_is_broken):
    Y.extend([i%2]*val)

n = tuple(map(float, range(10)))
i = 0
y_ = []
a = np.empty((0,8), dtype=np.float64)
for i0, l in enumerate(get_list(**par['length'])):
    for i1, di in enumerate(get_list(**par['diameter'])):
        for i2, y in enumerate(get_list(**par['young'])):
            for i3, de in enumerate(get_list(**par['density'])):
                for i4, pt in enumerate(get_list(**par['pressure_time'])):
                    for i5, pr in enumerate(get_list(**par['pressure_radius'])):
                        for i6, pa in enumerate(get_list(**par['pressure_amplitude'])):
                            for i7, s in enumerate(get_list(**par['strength'])):
                                if 0 not in [i4, i5, i6]:
                                    X.append([l, di, y, de, pt, pr, pa, s])
                                    y_.append(Y[i])
                                i += 1
    a = np.append(a, np.array(X), axis=0)
    X = []
    print(i0)
    #break


print('\n', '-'*10, 'data', '-'*10)
for i in range(20):
    print(X[i])

X = preprocessing.normalize(X)
print('\n', '-'*10, 'data + normalize', '-'*10)
for i in range(20):
    print(X[i])







'''

'''


