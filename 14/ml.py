# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import collections


from numpy import genfromtxt


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

e0 = tuple(enumerate(get_list(**par['length'])))
e1 = tuple(enumerate(get_list(**par['diameter'])))
e2 = tuple(enumerate(get_list(**par['young'])))
e3 = tuple(enumerate(get_list(**par['density'])))
e4 = tuple(enumerate(get_list(**par['pressure_time'])))
e5 = tuple(enumerate(get_list(**par['pressure_radius'])))
e6 = tuple(enumerate(get_list(**par['pressure_amplitude'])))
e7 = tuple(enumerate(get_list(**par['strength'])))

def make_str(data):
    return ''.join(map(str, data))
def make_set(data):
    return {make_str(i) for i in data}

source_f = '../12/ml_threads/6_1.txt'
with open(source_f, 'r') as f:
    threads = f.readlines()

cut = 500
x_train_dict = {}
for t in threads[:cut]:
    tr = list(map(int, t.replace('\n', '').split(',')))
    x_train_dict[make_str(tr[:-1])] = tr[-1]

i = 0
x_train, y_train = [], []
for i0, l in e0:
    for i1, di in e1:
        for i2, y in e2:
            for i3, de in e3:
                for i4, pt in e4:
                    for i5, pr in e5:
                        for i6, pa in e6:
                            for i7, s in e7:
                                if 0 not in [i4, i5, i6]:
                                    key = make_str([i0, i1, i2, i3, i4, i5, i6, i7])
                                    if key in x_train_dict:
                                        x_train.append([l, di, y, de, pt, pr, pa, s])
                                        y_train.append(x_train_dict[key])
                                i += 1
    print(i0)
x_train, y_train = np.array(x_train), np.array(y_train)


mi = list(map(float, x_train.min(axis=0)))
ma = list(map(float, x_train.max(axis=0)))
print('length', '{} - {}'.format(mi[0], ma[0]))
print('diameter', '{} - {}'.format(mi[1], ma[1]))
print('young', '{} - {}'.format(mi[2], ma[2]))
print('density', '{} - {}'.format(mi[3], ma[3]))
print('pressure_time', '{} - {}'.format(mi[4], ma[4]))
print('pressure_radius', '{} - {}'.format(mi[5], ma[5]))
print('pressure_amplitude', '{} - {}'.format(mi[6], ma[6]))
print('strength', '{} - {}'.format(mi[7], ma[7]))


extreme_values = [[
    par['length']['Min'],
    par['diameter']['Min'],
    par['young']['Min'],
    par['density']['Min'],
    par['pressure_time']['Min'],#get_list(**par['pressure_time'])[1],
    par['pressure_radius']['Min'],#get_list(**par['pressure_radius'])[1],
    par['pressure_amplitude']['Min'],#get_list(**par['pressure_amplitude'])[1],
    par['strength']['Min'],
    ],
    [
    par['length']['Max'],
    par['diameter']['Max'],
    par['young']['Max'],
    par['density']['Max'],
    par['pressure_time']['Max'],
    par['pressure_radius']['Max'],
    par['pressure_amplitude']['Max'],
    par['strength']['Max'],
    ]
]
extreme_values = np.array(extreme_values)

x_train = (x_train - extreme_values.min(axis=0)) / (extreme_values.max(axis=0) - extreme_values.min(axis=0))
print(x_train)

def make_test(num):
    data = genfromtxt('data0{}.csv'.format(num), delimiter=';', skip_header=True)
    x_test, y_test = [], []
    for d in data:
        #pressure_time;pressure_radius;pressure_amplitude;young;density;strength;length;diameter;is_broken
        pt, pr, pa, y, de, s, l, di, b = list(map(float, d))
        x_test.append([l, di, y, de, pt, pr, pa, s])
        y_test.append(b)
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = (x_test - extreme_values.min(axis=0)) / (extreme_values.max(axis=0) - extreme_values.min(axis=0))
    print(x_test)
    return x_test, y_test


def fit_model(model):
    print('\n', '-'*10, model.__class__.__name__, '-'*10)
    print(x_test.shape, y_test.shape)
    print('y_test', dict(collections.Counter(y_test)), 'y_train', dict(collections.Counter(y_train)))
    # fit model on training data
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print('y_pred', dict(collections.Counter(y_pred)))
    # make predictions for test data
    y_pred = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {}'.format(accuracy))

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:\n{}'.format(cm))

    print('Precision, recall and f1-score:')
    print(classification_report(y_test, y_pred))

    try:
        roc = roc_auc_score(y_test, y_pred)
        print('ROC AUC: {}'.format(roc))

        pr = average_precision_score(y_test, y_pred)
        print('PR AUC: {}'.format(pr))
    except Exception as e:
        print(e)

    print('-'*10, 'End',  model.__class__.__name__, '-'*10)

x_test, y_test = [], []
for num in range(1, 6):
    print('#'*5, num, '#'*5)
    x_test, y_test = make_test(num)
    fit_model(LogisticRegression())



'''
length 10.0 - 100.0
diameter 0.01 - 0.5
young 60.0 - 300.0
density 1000.0 - 2000.0
pressure_time 11.11 - 100.0
pressure_radius 0.56 - 5.0
pressure_amplitude 22.22 - 200.0
strength 0.2 - 10.0

[[0.         0.         0.         ... 0.112      0.1111     0.        ]
 [0.         0.         1.         ... 0.112      0.66665    1.        ]
 [0.         0.10204082 0.         ... 0.888      0.1111     1.        ]
 ...
 [1.         1.         1.         ... 0.888      0.1111     0.44489796]
 [1.         1.         1.         ... 0.666      0.1111     1.        ]
 [1.         1.         1.         ... 0.666      0.55555    0.55510204]]
##### 1 #####
[[0.41661313 0.34972041 0.27083333 ... 0.1213312  0.04254404 0.3877551 ]
 [0.57605707 0.19227755 0.27083333 ... 0.3301616  0.01910729 0.3877551 ]
 [0.74866349 0.32477347 0.27083333 ... 0.2336482  0.02240349 0.3877551 ]
 ...
 [0.43170003 0.33058571 0.27083333 ... 0.2167462  0.04962558 0.3877551 ]
 [0.80196049 0.36279796 0.27083333 ... 0.1799146  0.03195621 0.3877551 ]
 [0.43399338 0.23097143 0.27083333 ... 0.3719348  0.04006526 0.3877551 ]]

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {0.0: 1000} y_train {1: 167, 0: 333}
y_pred {0: 975, 1: 25}
Accuracy: 0.975
Confusion matrix:
[[975  25]
 [  0   0]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.97      0.99      1000
         1.0       0.00      0.00      0.00         0

   micro avg       0.97      0.97      0.97      1000
   macro avg       0.50      0.49      0.49      1000
weighted avg       1.00      0.97      0.99      1000

Only one class present in y_true. ROC AUC score is not defined in that case.
---------- End LogisticRegression ----------
##### 2 #####
[[0.86006079 0.3688     0.27083333 ... 0.9010802  0.77410619 0.3877551 ]
 [0.412436   0.25312041 0.27083333 ... 0.5306736  0.63991398 0.3877551 ]
 [0.70398518 0.38496122 0.27083333 ... 0.9713008  0.995086   0.3877551 ]
 ...
 [0.63879143 0.37690816 0.27083333 ... 0.9213186  0.5983255  0.3877551 ]
 [0.90587416 0.32686735 0.27083333 ... 0.7410606  0.74469634 0.3877551 ]
 [0.85916642 0.22810204 0.27083333 ... 0.1840332  0.64268082 0.3877551 ]]

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {1.0: 933, 0.0: 67} y_train {1: 167, 0: 333}
y_pred {1: 1000}
Accuracy: 0.933
Confusion matrix:
[[  0  67]
 [  0 933]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.00      0.00      0.00        67
         1.0       0.93      1.00      0.97       933

   micro avg       0.93      0.93      0.93      1000
   macro avg       0.47      0.50      0.48      1000
weighted avg       0.87      0.93      0.90      1000

ROC AUC: 0.5
PR AUC: 0.933
---------- End LogisticRegression ----------
##### 3 #####
[[0.77777778 0.18367347 0.13089808 ... 0.4417336  0.02884731 0.45169316]
 [0.77777778 0.18367347 0.12982491 ... 0.4683048  0.01767465 0.35498429]
 [0.77777778 0.18367347 0.38731981 ... 0.0652462  0.01956984 0.42728602]
 ...
 [0.77777778 0.18367347 0.41885878 ... 0.3294428  0.02901887 0.3391651 ]
 [0.77777778 0.18367347 0.43280211 ... 0.3577386  0.02303064 0.56715122]
 [0.77777778 0.18367347 0.11038333 ... 0.4620626  0.01335538 0.20540082]]

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {0.0: 1000} y_train {1: 167, 0: 333}
y_pred {1: 111, 0: 889}
Accuracy: 0.889
Confusion matrix:
[[889 111]
 [  0   0]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.89      0.94      1000
         1.0       0.00      0.00      0.00         0

   micro avg       0.89      0.89      0.89      1000
   macro avg       0.50      0.44      0.47      1000
weighted avg       1.00      0.89      0.94      1000

Only one class present in y_true. ROC AUC score is not defined in that case.
---------- End LogisticRegression ----------
##### 4 #####
[[0.77777778 0.18367347 0.16061921 ... 0.514294   0.99917638 0.41441398]
 [0.77777778 0.18367347 0.16716738 ... 0.964829   0.73893759 0.56106153]
 [0.77777778 0.18367347 0.14672082 ... 0.484006   0.52571555 0.35866898]
 ...
 [0.77777778 0.18367347 0.09346547 ... 0.8558858  0.60636981 0.26314908]
 [0.77777778 0.18367347 0.17718471 ... 0.6814964  0.61852563 0.34730755]
 [0.77777778 0.18367347 0.47493854 ... 0.8926358  0.83066915 0.30616735]]

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {1.0: 976, 0.0: 24} y_train {1: 167, 0: 333}
y_pred {1: 1000}
Accuracy: 0.976
Confusion matrix:
[[  0  24]
 [  0 976]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.00      0.00      0.00        24
         1.0       0.98      1.00      0.99       976

   micro avg       0.98      0.98      0.98      1000
   macro avg       0.49      0.50      0.49      1000
weighted avg       0.95      0.98      0.96      1000

ROC AUC: 0.5
PR AUC: 0.976
---------- End LogisticRegression ----------
##### 5 #####
[[0.87105536 0.24185102 0.15263574 ... 0.5978706  0.29545299 0.55628204]
 [0.84498792 0.31454694 0.42383385 ... 0.5234424  0.7918293  0.48468327]
 [0.90224796 0.34024286 0.43849    ... 0.5167472  0.36995751 0.28285878]
 ...
 [0.40627506 0.30915918 0.2585728  ... 0.5887886  0.54006586 0.53269837]
 [0.62903276 0.30045714 0.10703687 ... 0.7384694  0.99269049 0.35152643]
 [0.67751964 0.34665918 0.43012797 ... 0.7289876  0.09283124 0.18412449]]

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 167, 0: 333}
y_pred {1: 902, 0: 98}
Accuracy: 0.733
Confusion matrix:
[[ 98 267]
 [  0 635]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.27      0.42       365
         1.0       0.70      1.00      0.83       635

   micro avg       0.73      0.73      0.73      1000
   macro avg       0.85      0.63      0.62      1000
weighted avg       0.81      0.73      0.68      1000

ROC AUC: 0.6342465753424658
PR AUC: 0.7039911308203991
---------- End LogisticRegression ----------
'''


