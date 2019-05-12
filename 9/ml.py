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

with open('fib_all_data.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))

X, Y = [], []
for i, val in enumerate(data_is_broken):
    Y.extend([i%2]*val)

for l in get_list(**par['length']):
    for di in get_list(**par['diameter']):
        for y in get_list(**par['young']):
            for de in get_list(**par['density']):
                for pt in get_list(**par['pressure_time']):
                    for pr in get_list(**par['pressure_radius']):
                        for pa in get_list(**par['pressure_amplitude']):
                            for s in get_list(**par['strength']):
                                X.append([l, di, y, de, pt, pr, pa, s])
            #break
        #break
    break

print('!!!', len(Y))
X, y = np.array(X), np.array(Y)
y = y[:X.shape[0]]
print(X.shape, y.shape)

# split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.00001, random_state=42)
print(y_train.shape, y_test.shape)
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


### XGBoost 0.1% train = 100% score.    dataset = 100 000 ###
### XGBoost 0.1% train = 98% score.  dataset = 10 000 000 ###
### XGBoost 0.01% train = 97% score. dataset = 10 000 000 ###
### XGBoost 0.001% train = 91% score. dataset = 10 000 000 ###
#x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.001, random_state=42)
fit_model(XGBClassifier())

### LogisticRegression 90% train = 78% score.      dataset = 100 000 ###
### LogisticRegression 0.1% train = 79% score.  dataset = 10 000 000 ###
### LogisticRegression 0.01% train = 79% score. dataset = 10 000 000 ###
### LogisticRegression 0.001% train = 78% score. dataset = 10 000 000 ###
#x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)
fit_model(LogisticRegression())


### LinearSVC 90% train = 73% score.      dataset = 100 000 ###
### LinearSVC 0.1% train = 74% score.  dataset = 10 000 000 ###
### LinearSVC 0.01% train = 69% score. dataset = 10 000 000 ###
### LinearSVC 0.001% train = 50% score. dataset = 10 000 000 ###
#x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)
fit_model(LinearSVC(random_state=0, tol=1e-5))

### KNeighborsClassifier ###
# 10%  train for 91.8% score.    dataset = 100 000  #
# 1%   train for 82.7% score.    dataset = 100 000  #
# 0.1% train for 73.3% score.    dataset = 100 000  #
# 1%   train for 86.3% score. dataset = 10 000 000  #
# 0.1% train for 80.8% score. dataset = 10 000 000  (6)#
# 0.01% train for 74% score.  dataset = 10 000 000  (6)#
# 0.001% train for 70% score. dataset = 10 000 000  (6)#
#x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.1, random_state=42)
#x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.01, random_state=42)
#x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.001, random_state=42)
# fit_model(KNeighborsClassifier(n_neighbors=2))
# fit_model(KNeighborsClassifier(n_neighbors=3))
# fit_model(KNeighborsClassifier(n_neighbors=4))
# fit_model(KNeighborsClassifier(n_neighbors=5))
fit_model(KNeighborsClassifier(n_neighbors=6))
# fit_model(KNeighborsClassifier(n_neighbors=7))
# fit_model(KNeighborsClassifier(n_neighbors=8))
# fit_model(KNeighborsClassifier(n_neighbors=9))







