# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

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

with open('data_45k/fib_all_data1.3.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))

X, Y = [], []

for i, val in enumerate(data_is_broken):
    Y.extend([i%2]*val)

a = np.empty((0,8), dtype=np.float64)
for i, l in enumerate(get_list(**par['length'])):
    for di in get_list(**par['diameter']):
        for y in get_list(**par['young']):
            for de in get_list(**par['density']):
                for pt in get_list(**par['pressure_time']):
                    for pr in get_list(**par['pressure_radius']):
                        for pa in get_list(**par['pressure_amplitude']):
                            for s in get_list(**par['strength']):
                                #a = np.append(a, np.array([[l, di, y, de, pt, pr, pa, s]]), axis=0)
                                X.append([l, di, y, de, pt, pr, pa, s])
            #break
        #break
    a = np.append(a, np.array(X), axis=0)
    X = []
    print(i)
    #break

print('!!!', len(Y))
X, y = a, np.array(Y)
X = (X - X.mean()) / X.std()
y = y[:X.shape[0]]
print(X.shape, y.shape)

# split data into train and test sets

def fit_model(model):
    # fit model on training data
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # make predictions for test data
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    #print(predictions)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.20f%%" % (accuracy * 100.0))

for i in [0.000001, 0.00001, 0.0001, 0.001, 0.01]:
    print('!!! {} !!!'.format(i))
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=i, random_state=42)
    print(y_train.shape, y_test.shape)
    fit_model(XGBClassifier())
    fit_model(LogisticRegression())
    fit_model(LinearSVC(random_state=0, tol=1e-5))
    fit_model(KNeighborsClassifier(n_neighbors=6))







