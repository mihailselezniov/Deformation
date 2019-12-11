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

print('!!!')
X, Y = a, np.array(y_)
print(X.shape, Y.shape)
print('all', dict(collections.Counter(Y)))


# normalize the data attributes
X = preprocessing.normalize(X)


def fit_model(model):
    print('\n', '-'*10, model.__class__.__name__, '-'*10)
    # fit model on training data
    model.fit(x_train, y_train)

    #joblib.dump(model, 'dump_models/KNN7.j')

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


#print('!!! {} !!!'.format(i))
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.01, random_state=42)
print(y_train.shape, y_test.shape)
print('y_test', dict(collections.Counter(y_test)), 'y_train', dict(collections.Counter(y_train)))
fit_model(KNeighborsClassifier(n_neighbors=7, n_jobs=-1))



'''
(72900000, 8) (72900000,)
all {1: 63518691, 0: 9381309}
(729000,) (72171000,)
y_test {1: 62883803, 0: 9287197} y_train {0: 94112, 1: 634888}

 ---------- KNeighborsClassifier ----------
y_pred {1: 66368482, 0: 5802518}
Accuracy: 0.8809584874811213
Confusion matrix:
[[ 3249185  6038012]
 [ 2553333 60330470]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.56      0.35      0.43   9287197
           1       0.91      0.96      0.93  62883803

    accuracy                           0.88  72171000
   macro avg       0.73      0.65      0.68  72171000
weighted avg       0.86      0.88      0.87  72171000

ROC AUC: 0.6546261908408766
PR AUC: 0.9074918814235522
---------- End KNeighborsClassifier ----------
'''


