# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
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

for i in [0.00000471]:
    print('!!! {} !!!'.format(i))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=i, random_state=42)
    print(y_train.shape, y_test.shape)
    print('y_test', dict(collections.Counter(y_test)), 'y_train', dict(collections.Counter(y_train)))
    fit_model(XGBClassifier())
    #fit_model(LogisticRegression())
    #fit_model(LinearSVC(random_state=0, tol=1e-5))
    #fit_model(KNeighborsClassifier(n_neighbors=6))

'''
(72900000, 8) (72900000,)
all {1: 63518691, 0: 9381309}
!!! 4.6e-06 !!!
(335,) (72899665,)
y_test {1: 63518398, 0: 9381267} y_train {1: 293, 0: 42}

 ---------- XGBClassifier ----------
y_pred {1: 66520927, 0: 6378738}
Accuracy: 0.9233680291946472
Confusion matrix:
[[ 5086780  4294487]
 [ 1291958 62226440]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.80      0.54      0.65   9381267
           1       0.94      0.98      0.96  63518398

    accuracy                           0.92  72899665
   macro avg       0.87      0.76      0.80  72899665
weighted avg       0.92      0.92      0.92  72899665

ROC AUC: 0.7609437481730126
Precision-recall: 0.9341371876474559
---------- End XGBClassifier ----------

(72900000, 8) (72900000,)
all {1: 63518691, 0: 9381309}
!!! 4.71e-06 !!!
(343,) (72899657,)
y_test {1: 63518391, 0: 9381266} y_train {1: 300, 0: 43}

 ---------- XGBClassifier ----------
y_pred {1: 66524543, 0: 6375114}
Accuracy: 0.9241617556581919
Confusion matrix:
[[ 5113899  4267367]
 [ 1261215 62257176]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.80      0.55      0.65   9381266
           1       0.94      0.98      0.96  63518391

    accuracy                           0.92  72899657
   macro avg       0.87      0.76      0.80  72899657
weighted avg       0.92      0.92      0.92  72899657

ROC AUC: 0.7626311573419088
Precision-recall: 0.934571239912142
---------- End XGBClassifier ----------
'''


