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
for i0, l in enumerate(get_list(**par['length'])):
    for i1, di in enumerate(get_list(**par['diameter'])):
        for i2, y in enumerate(get_list(**par['young'])):
            for i3, de in enumerate(get_list(**par['density'])):
                for i4, pt in enumerate(get_list(**par['pressure_time'])):
                    for i5, pr in enumerate(get_list(**par['pressure_radius'])):
                        for i6, pa in enumerate(get_list(**par['pressure_amplitude'])):
                            for i7, s in enumerate(get_list(**par['strength'])):
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
    break

print('!!!')
X, Y = a, np.array(y_)
x_train, y_train = np.array(x_train), np.array(y_train)
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

'''


