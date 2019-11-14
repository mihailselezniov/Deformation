# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import collections

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
y = []
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
                                    y.append(Y[i])
                                i += 1
    a = np.append(a, np.array(X), axis=0)
    X = []
    print(i0)
    #break

print('!!!')
X, Y = a, np.array(y)
print(X.shape, Y.shape)
print('all', dict(collections.Counter(Y)))

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def fit_model(model):
    # fit model on training data
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print('y_pred', dict(collections.Counter(y_pred)))
    # make predictions for test data
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    #print(predictions)
    accuracy = accuracy_score(y_test, predictions)
    print("%.2f%% %s" % (accuracy * 100.0, model.__class__.__name__))

for i in [0.000001, 0.00001, 0.0001]:
    print('!!! {} !!!'.format(i))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=i, random_state=42)
    print(y_train.shape, y_test.shape)
    print('y_test', dict(collections.Counter(y_test)), 'y_train', dict(collections.Counter(y_train)))
    fit_model(XGBClassifier())
    fit_model(LogisticRegression())
    fit_model(LinearSVC(random_state=0, tol=1e-5))
    fit_model(KNeighborsClassifier(n_neighbors=6))

'''
(72900000, 8) (72900000,)
all {1: 63518691, 0: 9381309}
!!! 1e-06 !!!
(72,) (72899928,)
y_test {1: 63518632, 0: 9381296} y_train {1: 59, 0: 13}
y_pred {1: 65650697, 0: 7249231}
91.12% XGBClassifier
y_pred {1: 62405843, 0: 10494085}
92.13% LogisticRegression
y_pred {1: 61598678, 0: 11301250}
91.47% LinearSVC
y_pred {1: 66508158, 0: 6391770}
88.66% KNeighborsClassifier
!!! 1e-05 !!!
(729,) (72899271,)
y_test {1: 63518072, 0: 9381199} y_train {1: 619, 0: 110}
y_pred {1: 65221280, 0: 7677991}
94.95% XGBClassifier
y_pred {1: 63970581, 0: 8928690}
94.16% LogisticRegression
y_pred {1: 63750700, 0: 9148571}
94.11% LinearSVC
y_pred {1: 64579208, 0: 8320063}
91.72% KNeighborsClassifier
!!! 0.0001 !!!
(7290,) (72892710,)
y_test {1: 63512321, 0: 9380389} y_train {1: 6370, 0: 920}
y_pred {1: 65136249, 0: 7756461}
96.50% XGBClassifier
y_pred {1: 64608218, 0: 8284492}
94.23% LogisticRegression
y_pred {1: 61605547, 0: 11287163}
93.23% LinearSVC
y_pred {1: 64491701, 0: 8401009}
93.98% KNeighborsClassifier
'''


