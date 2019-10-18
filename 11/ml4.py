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
#a = np.empty((0,8), dtype=np.float64)
for i0, l in enumerate(get_list(**par['length'])):
    for i1, di in enumerate(get_list(**par['diameter'])):
        for i2, y in enumerate(get_list(**par['young'])):
            for i3, de in enumerate(get_list(**par['density'])):
                for i4, pt in enumerate(get_list(**par['pressure_time'])):
                    for i5, pr in enumerate(get_list(**par['pressure_radius'])):
                        for i6, pa in enumerate(get_list(**par['pressure_amplitude'])):
                            for i7, s in enumerate(get_list(**par['strength'])):
                                if 0 not in [i0, i1, i2, i3, i4, i5, i6, i7]:
                                    X.append([l, di, y, de, pt, pr, pa, s])
                                    new_Y.append(Y[i])
                                i += 1
            #break
        #break
    #a = np.append(a, np.array(X), axis=0)
    #X = []
    print(i)
    #break

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

for i in [0.000001, 0.00001, 0.0001, 0.001, 0.01]:
    print('!!! {} !!!'.format(i))
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=i, random_state=42)
    print(y_train.shape, y_test.shape)
    fit_model(XGBClassifier())


'''
164168
0.38137167288537493
!!! 43046721
(43046721, 8) (43046721,)
!!! 1e-06 !!!
(43,) (43046678,)
Accuracy: 99.61862794615649363550%
!!! 1e-05 !!!
(430,) (43046291,)
Accuracy: 99.61862916366011688751%
!!! 0.0001 !!!
(4304,) (43042417,)
Accuracy: 99.62719101020744005837%
!!! 0.001 !!!
(43046,) (43003675,)
Accuracy: 99.63602413049582651183%
!!! 0.01 !!!
(430467,) (42616254,)
Accuracy: 99.64230314564954937850%
[Finished in 2967.0s]
'''





