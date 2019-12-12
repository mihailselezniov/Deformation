# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import collections

def make_str(data):
    return ''.join(map(str, data))
def make_set(data):
    return {make_str(i) for i in data}

source_f = '../12/ml_threads/6_1.txt'
with open(source_f, 'r') as f:
    threads = f.readlines()
x_train_, y_train_ = [], []
cut = 200
x_train_dict = {}
for t in threads[:cut]:
    tr = list(map(int, t.replace('\n', '').split(',')))
    x_train_.append(tr[:-1])
    y_train_.append(tr[-1])
    x_train_dict[make_str(tr[:-1])] = tr[-1]

#x_train_set = make_set(x_train_)
#print(x_train_set)



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


with open('../11/fib_all_data.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))


X, Y = [], []
for i, val in enumerate(data_is_broken):
    Y.extend([i%2]*val)

n = tuple(map(float, range(10)))
i = 0
y_ = []
x_train, y_train = [], []
a = np.empty((0,8), dtype=np.float64)

e0 = tuple(enumerate(get_list(**par['length'])))
e1 = tuple(enumerate(get_list(**par['diameter'])))
e2 = tuple(enumerate(get_list(**par['young'])))
e3 = tuple(enumerate(get_list(**par['density'])))
e4 = tuple(enumerate(get_list(**par['pressure_time'])))
e5 = tuple(enumerate(get_list(**par['pressure_radius'])))
e6 = tuple(enumerate(get_list(**par['pressure_amplitude'])))
e7 = tuple(enumerate(get_list(**par['strength'])))

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
                                    X.append([l, di, y, de, pt, pr, pa, s])
                                    y_.append(Y[i])
                                i += 1
    a = np.append(a, np.array(X), axis=0)
    X = []
    print(i0)
    #break

print('!!!')
X, Y = a, np.array(y_)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
x_train, y_train = np.array(x_train), np.array(y_train)
print('x_train =', x_train)

x_train_ = (x_train - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
print('x_train =', x_train_)
print(X.shape, Y.shape)
print('all', dict(collections.Counter(Y)))


def fit_model(model):
    print('\n', '-'*10, model.__class__.__name__, '-'*10)
    # fit model on training data
    model.fit(x_train, y_train)

    joblib.dump(model, 'dump_models/LogReg.j')

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

    roc = roc_auc_score(y_test, y_pred)
    print('ROC AUC: {}'.format(roc))

    pr = average_precision_score(y_test, y_pred)
    print('PR AUC: {}'.format(pr))

    print('-'*10, 'End',  model.__class__.__name__, '-'*10)


x_test, y_test = X, Y
print(y_train.shape, y_test.shape)
print('y_test', dict(collections.Counter(y_test)), 'y_train', dict(collections.Counter(y_train)))
fit_model(LogisticRegression())




'''
(72900000, 8) (72900000,)
all {1: 63518691, 0: 9381309}
(200,) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1: 69, 0: 131}

 ---------- LogisticRegression ----------
y_pred {1: 56870504, 0: 16029496}
Accuracy: 0.8785569410150892
Confusion matrix:
[[ 8278803  1102506]
 [ 7750693 55767998]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.52      0.88      0.65   9381309
           1       0.98      0.88      0.93  63518691

    accuracy                           0.88  72900000
   macro avg       0.75      0.88      0.79  72900000
weighted avg       0.92      0.88      0.89  72900000

ROC AUC: 0.8802281033189825
PR AUC: 0.967276579925729
---------- End LogisticRegression ----------
'''


