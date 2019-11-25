# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import collections
from sklearn.externals import joblib


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

X = joblib.load('border_vars/X.j')
print('X')
Y = joblib.load('border_vars/Y.j')
print('Y')

print('!!!')
print(X.shape, Y.shape)
print('all', dict(collections.Counter(Y)))

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def fit_model(model):
    print('\n', '-'*10, model.__class__.__name__, '-'*10)
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

    roc = roc_auc_score(y_test, y_pred)
    print('ROC AUC: {}'.format(roc))

    pr = average_precision_score(y_test, y_pred)
    print('Precision-recall: {}'.format(pr))

    print('-'*10, 'End',  model.__class__.__name__, '-'*10)

ml_type = 1
threads_f = 'threads/threads_{}'.format(ml_type)
threads_f_name = '{}.j'.format(threads_f)
threads = joblib.load(threads_f_name)
len_threads = len(threads[0]) + len(threads[1])
print('threads', len_threads)

x_train, y_train = [], []
for i in threads:
    for thread in threads[i]:
        x_train.append(thread)
        y_train.append(i)
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = X, Y

print(y_train.shape, y_test.shape)
print('y_test', dict(collections.Counter(y_test)), 'y_train', dict(collections.Counter(y_train)))
fit_model(XGBClassifier())
    #fit_model(LogisticRegression())
    #fit_model(LinearSVC(random_state=0, tol=1e-5))
    #fit_model(KNeighborsClassifier(n_neighbors=6))

'''

'''


