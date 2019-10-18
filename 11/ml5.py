# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
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

with open('data45k.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))

X, Y, new_Y = [], [], []

for i, val in enumerate(data_is_broken):
    Y.extend([i%2]*val)

i = 0
for i0, l in enumerate(get_list(**par['length'])):
    for i1, di in enumerate(get_list(**par['diameter'])):
        for i2, y in enumerate(get_list(**par['young'])):
            for i3, de in enumerate(get_list(**par['density'])):
                for i4, pt in enumerate(get_list(**par['pressure_time'])):
                    for i5, pr in enumerate(get_list(**par['pressure_radius'])):
                        for i6, pa in enumerate(get_list(**par['pressure_amplitude'])):
                            for i7, s in enumerate(get_list(**par['strength'])):
                                if 0 not in [i4, i5, i6]:
                                    if 1 not in [i4, i5, i6]:
                                        if [2, 2, 2] == [i5, i6, i7]:
                                            X.append([l, di, y, de, pt])
                                            new_Y.append(Y[i])
                                i += 1
    print(i0)


Y = new_Y
zero = Y.count(0)
print(zero)
print(zero*100/len(Y))
print('!!!', len(Y))
X, y = np.array(X), np.array(Y)
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

for i in [0.0001, 0.001, 0.01, 0.1, 0.3]:
    print('!!! {} !!!'.format(i))
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=i, random_state=42)
    print(y_train.shape, y_test.shape)
    fit_model(XGBClassifier())


'''
158
0.1975
!!! 80000
(80000, 5) (80000,)
!!! 0.0001 !!!
(8,) (79992,)
Accuracy: 99.80248024802480699691%
!!! 0.001 !!!
(80,) (79920,)
Accuracy: 99.80230230230230858979%
!!! 0.01 !!!
(800,) (79200,)
Accuracy: 99.80429292929292728331%
!!! 0.1 !!!
(8000,) (72000,)
Accuracy: 99.79722222222223138033%
!!! 0.3 !!!
(24000,) (56000,)
Accuracy: 99.79285714285714448124%
[Finished in 719.5s]
'''





