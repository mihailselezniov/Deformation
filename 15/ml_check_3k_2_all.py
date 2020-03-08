# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from xgboost import XGBClassifier, XGBRegressor
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

e0 = tuple(enumerate(get_list(**par['length'])))
e1 = tuple(enumerate(get_list(**par['diameter'])))
e2 = tuple(enumerate(get_list(**par['young'])))
e3 = tuple(enumerate(get_list(**par['density'])))
e4 = tuple(enumerate(get_list(**par['pressure_time'])))
e5 = tuple(enumerate(get_list(**par['pressure_radius'])))
e6 = tuple(enumerate(get_list(**par['pressure_amplitude'])))
e7 = tuple(enumerate(get_list(**par['strength'])))


extreme_values = [[
        par['length']['Min'],
        par['diameter']['Min'],
        par['young']['Min'],
        par['density']['Min'],
        par['pressure_time']['Min'],#get_list(**par['pressure_time'])[1],
        par['pressure_radius']['Min'],#get_list(**par['pressure_radius'])[1],
        par['pressure_amplitude']['Min'],#get_list(**par['pressure_amplitude'])[1],
        par['strength']['Min'],
        ],
        [
        par['length']['Max'],
        par['diameter']['Max'],
        par['young']['Max'],
        par['density']['Max'],
        par['pressure_time']['Max'],
        par['pressure_radius']['Max'],
        par['pressure_amplitude']['Max'],
        par['strength']['Max'],
        ]
    ]
extreme_values = np.array(extreme_values)
#x_train = (x_train - extreme_values.min(axis=0)) / (extreme_values.max(axis=0) - extreme_values.min(axis=0))


with open('data3k_2.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))

y_test = []
for i, val in enumerate(data_is_broken):
    y_test.extend([i%2]*val)

new_parts = 19
def get_new_list(Min, Max):
    return list(map(lambda x: round(x, 2), np.arange(Min, Max+0.01, (Max-Min)/(new_parts-1))))[1::2]

e2_0 = tuple(enumerate(get_new_list(**par['length'])))
e2_1 = tuple(enumerate(get_new_list(**par['diameter'])))
e2_2 = tuple(enumerate(get_new_list(**par['young'])))
e2_3 = tuple(enumerate(get_new_list(**par['density'])))
e2_4 = tuple(enumerate(get_new_list(**par['pressure_time'])))
e2_5 = tuple(enumerate(get_new_list(**par['pressure_radius'])))
e2_6 = tuple(enumerate(get_new_list(**par['pressure_amplitude'])))
e2_7 = tuple(enumerate(get_new_list(**par['strength'])))


x_test = []
for i0, l in e2_0:
    for i1, di in e2_1:
        for i2, y in e2_2:
            for i3, de in e2_3:
                for i4, pt in e2_4:
                    for i5, pr in e2_5:
                        for i6, pa in e2_6:
                            for i7, s in e2_7:
                                #if 0 not in [i4, i5, i6]:
                                x_test.append([l, di, y, de, pt, pr, pa, s])
    print(i0)
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = (x_test - extreme_values.min(axis=0)) / (extreme_values.max(axis=0) - extreme_values.min(axis=0))



def make_str(data):
    return ''.join(map(str, data))
def make_set(data):
    return {make_str(i) for i in data}

source_f = '../12/ml_threads/6_1.txt'
with open(source_f, 'r') as f:
    threads = f.readlines()

roc_metrics, pr_metrics, f1_metrics = [], [], []
roc_metric, pr_metric, f1_metric = [], [], []

for cut in [100, 200, 300, 400, 500]:
    #cut = 200#100
    print('\n\n\n', '#'*10, cut, '#'*10)
    x_train_dict = {}
    for t in threads[:cut]:
        tr = list(map(int, t.replace('\n', '').split(',')))
        x_train_dict[make_str(tr[:-1])] = tr[-1]

    i = 0
    x_train, y_train = [], []
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
                                    i += 1
        print(i0)
    x_train, y_train = np.array(x_train), np.array(y_train)


    x_train = (x_train - extreme_values.min(axis=0)) / (extreme_values.max(axis=0) - extreme_values.min(axis=0))
    #print(x_train)


    def fit_model(model):
        global roc_metric
        global pr_metric
        global f1_metric
        print('\n', '-'*10, model.__class__.__name__, '-'*10)
        print(x_test.shape, y_test.shape)
        print('y_test', dict(collections.Counter(y_test)), 'y_train', dict(collections.Counter(y_train)))
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

        try:
            roc = roc_auc_score(y_test, y_pred)
            print('ROC AUC: {}'.format(roc))

            pr = average_precision_score(y_test, y_pred)
            print('PR AUC: {}'.format(pr))

            roc_metric.append(round(float(roc), 4))
            pr_metric.append(round(float(pr), 4))

            f1 = f1_score(y_test, y_pred, average=None)
            f1_metric.append(round(float(f1[0]), 4))

        except Exception as e:
            print(e)

        print('-'*10, 'End',  model.__class__.__name__, '-'*10)


    roc_metric, pr_metric, f1_metric = [], [], []


    fit_model(XGBClassifier(random_state=42))
    fit_model(LogisticRegression())
    fit_model(LinearSVC(random_state=42, tol=1e-5))
    fit_model(KNeighborsClassifier(n_neighbors=5))
    fit_model(SGDClassifier(random_state=42))
    fit_model(BernoulliNB())
    fit_model(RandomForestClassifier(random_state=42))
    fit_model(MLPClassifier())
    fit_model(SVC(random_state=42))

    roc_metrics.append(roc_metric[:])
    pr_metrics.append(pr_metric[:])
    f1_metrics.append(f1_metric[:])


print('roc_metrics')
print(roc_metrics)
print('pr_metrics')
print(pr_metrics)
print('f1_metrics')
print(f1_metrics)
print()



'''
 ########## 100 ##########
0
1
2
3
4
5
6
7
8
9

 ---------- XGBClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 33, 0: 67}
y_pred {1: 25496173, 0: 17550548}
Accuracy: 0.7734472969497491
Confusion matrix:
[[ 8772387   974190]
 [ 8778161 24521983]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.50      0.90      0.64   9746577
           1       0.96      0.74      0.83  33300144

    accuracy                           0.77  43046721
   macro avg       0.73      0.82      0.74  43046721
weighted avg       0.86      0.77      0.79  43046721

ROC AUC: 0.818220343781294
PR AUC: 0.9121773895121484
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 23505259, 0: 19541462}
Accuracy: 0.7564580819059366
Confusion matrix:
[[ 9402179   344398]
 [10139283 23160861]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.48      0.96      0.64   9746577
           1       0.99      0.70      0.82  33300144

    accuracy                           0.76  43046721
   macro avg       0.73      0.83      0.73  43046721
weighted avg       0.87      0.76      0.78  43046721

ROC AUC: 0.8300915332545971
PR AUC: 0.9208689953167597
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 33, 0: 67}
y_pred {1: 32987585, 0: 10059136}
Accuracy: 0.8921962255847548
Confusion matrix:
[[ 7582557  2164020]
 [ 2476579 30823565]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.75      0.78      0.77   9746577
           1       0.93      0.93      0.93  33300144

    accuracy                           0.89  43046721
   macro avg       0.84      0.85      0.85  43046721
weighted avg       0.89      0.89      0.89  43046721

ROC AUC: 0.8517999307371715
PR AUC: 0.9224387286514195
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 33, 0: 67}
y_pred {0: 34212441, 1: 8834280}
Accuracy: 0.4062420689371439
Confusion matrix:
[[ 9199843   546734]
 [25012598  8287546]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.27      0.94      0.42   9746577
           1       0.94      0.25      0.39  33300144

    accuracy                           0.41  43046721
   macro avg       0.60      0.60      0.41  43046721
weighted avg       0.79      0.41      0.40  43046721

ROC AUC: 0.5963896016699186
PR AUC: 0.8145288948528536
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 33, 0: 67}
y_pred {1: 33786598, 0: 9260123}
Accuracy: 0.8926810244153092
Confusion matrix:
[[ 7193485  2553092]
 [ 2066638 31233506]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.78      0.74      0.76   9746577
           1       0.92      0.94      0.93  33300144

    accuracy                           0.89  43046721
   macro avg       0.85      0.84      0.84  43046721
weighted avg       0.89      0.89      0.89  43046721

ROC AUC: 0.8379957568428628
PR AUC: 0.9150726723708281
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 33, 0: 67}
y_pred {0: 43046721}
Accuracy: 0.22641856972102475
Confusion matrix:
[[ 9746577        0]
 [33300144        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.23      1.00      0.37   9746577
           1       0.00      0.00      0.00  33300144

    accuracy                           0.23  43046721
   macro avg       0.11      0.50      0.18  43046721
weighted avg       0.05      0.23      0.08  43046721

ROC AUC: 0.5
PR AUC: 0.7735814302789753
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 8444934, 0: 34601787}
Accuracy: 0.40341253866932164
Confusion matrix:
[[ 9333615   412962]
 [25268172  8031972]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.27      0.96      0.42   9746577
           1       0.95      0.24      0.38  33300144

    accuracy                           0.40  43046721
   macro avg       0.61      0.60      0.40  43046721
weighted avg       0.80      0.40      0.39  43046721

ROC AUC: 0.5994146826504084
PR AUC: 0.8163986530889374
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 27678271, 0: 15368450}
Accuracy: 0.8309363214912466
Confusion matrix:
[[ 8918695   827882]
 [ 6449755 26850389]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.58      0.92      0.71   9746577
           1       0.97      0.81      0.88  33300144

    accuracy                           0.83  43046721
   macro avg       0.78      0.86      0.80  43046721
weighted avg       0.88      0.83      0.84  43046721

ROC AUC: 0.8606868530553057
PR AUC: 0.9320284138615378
---------- End MLPClassifier ----------

 ---------- SVC ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 43046721}
Accuracy: 0.22641856972102475
Confusion matrix:
[[ 9746577        0]
 [33300144        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.23      1.00      0.37   9746577
           1       0.00      0.00      0.00  33300144

    accuracy                           0.23  43046721
   macro avg       0.11      0.50      0.18  43046721
weighted avg       0.05      0.23      0.08  43046721

ROC AUC: 0.5
PR AUC: 0.7735814302789753
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End SVC ----------



 ########## 200 ##########
0
1
2
3
4
5
6
7
8
9

 ---------- XGBClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 69, 0: 131}
y_pred {1: 28065218, 0: 14981503}
Accuracy: 0.8096448275351797
Confusion matrix:
[[ 8266957  1479620]
 [ 6714546 26585598]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.55      0.85      0.67   9746577
           1       0.95      0.80      0.87  33300144

    accuracy                           0.81  43046721
   macro avg       0.75      0.82      0.77  43046721
weighted avg       0.86      0.81      0.82  43046721

ROC AUC: 0.8232768298928631
PR AUC: 0.912255298630718
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 29193885, 0: 13852836}
Accuracy: 0.8566421121831789
Confusion matrix:
[[ 8714163  1032414]
 [ 5138673 28161471]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.63      0.89      0.74   9746577
           1       0.96      0.85      0.90  33300144

    accuracy                           0.86  43046721
   macro avg       0.80      0.87      0.82  43046721
weighted avg       0.89      0.86      0.86  43046721

ROC AUC: 0.8698801796543981
PR AUC: 0.9351535942030833
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 69, 0: 131}
y_pred {1: 33910849, 0: 9135872}
Accuracy: 0.9009962919126872
Confusion matrix:
[[ 7310332  2436245]
 [ 1825540 31474604]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.80      0.75      0.77   9746577
           1       0.93      0.95      0.94  33300144

    accuracy                           0.90  43046721
   macro avg       0.86      0.85      0.86  43046721
weighted avg       0.90      0.90      0.90  43046721

ROC AUC: 0.8476100895704881
PR AUC: 0.9196833840326784
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 69, 0: 131}
y_pred {0: 31698967, 1: 11347754}
Accuracy: 0.44769558638391993
Confusion matrix:
[[ 8835325   911252]
 [22863642 10436502]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.28      0.91      0.43   9746577
           1       0.92      0.31      0.47  33300144

    accuracy                           0.45  43046721
   macro avg       0.60      0.61      0.45  43046721
weighted avg       0.77      0.45      0.46  43046721

ROC AUC: 0.6099562729098419
PR AUC: 0.8193752739304228
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 69, 0: 131}
y_pred {1: 32302649, 0: 10744072}
Accuracy: 0.881783911020772
Confusion matrix:
[[ 7700917  2045660]
 [ 3043155 30256989]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.72      0.79      0.75   9746577
           1       0.94      0.91      0.92  33300144

    accuracy                           0.88  43046721
   macro avg       0.83      0.85      0.84  43046721
weighted avg       0.89      0.88      0.88  43046721

ROC AUC: 0.8493646956625647
PR AUC: 0.9217679244093868
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 69, 0: 131}
y_pred {0: 43046721}
Accuracy: 0.22641856972102475
Confusion matrix:
[[ 9746577        0]
 [33300144        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.23      1.00      0.37   9746577
           1       0.00      0.00      0.00  33300144

    accuracy                           0.23  43046721
   macro avg       0.11      0.50      0.18  43046721
weighted avg       0.05      0.23      0.08  43046721

ROC AUC: 0.5
PR AUC: 0.7735814302789753
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 13927635, 0: 29119086}
Accuracy: 0.5108699917004131
Confusion matrix:
[[ 8905110   841467]
 [20213976 13086168]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.31      0.91      0.46   9746577
           1       0.94      0.39      0.55  33300144

    accuracy                           0.51  43046721
   macro avg       0.62      0.65      0.51  43046721
weighted avg       0.80      0.51      0.53  43046721

ROC AUC: 0.653320850438345
PR AUC: 0.8388160856741352
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 28647210, 0: 14399511}
Accuracy: 0.8517054063188693
Confusion matrix:
[[ 8881246   865331]
 [ 5518265 27781879]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.62      0.91      0.74   9746577
           1       0.97      0.83      0.90  33300144

    accuracy                           0.85  43046721
   macro avg       0.79      0.87      0.82  43046721
weighted avg       0.89      0.85      0.86  43046721

ROC AUC: 0.8727519929054701
PR AUC: 0.9372786473099952
---------- End MLPClassifier ----------

 ---------- SVC ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 43046721}
Accuracy: 0.22641856972102475
Confusion matrix:
[[ 9746577        0]
 [33300144        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.23      1.00      0.37   9746577
           1       0.00      0.00      0.00  33300144

    accuracy                           0.23  43046721
   macro avg       0.11      0.50      0.18  43046721
weighted avg       0.05      0.23      0.08  43046721

ROC AUC: 0.5
PR AUC: 0.7735814302789753
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End SVC ----------



 ########## 300 ##########
0
1
2
3
4
5
6
7
8
9

 ---------- XGBClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 98, 0: 202}
y_pred {1: 28994824, 0: 14051897}
Accuracy: 0.8301576094495096
Confusion matrix:
[[ 8243658  1502919]
 [ 5808239 27491905]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.59      0.85      0.69   9746577
           1       0.95      0.83      0.88  33300144

    accuracy                           0.83  43046721
   macro avg       0.77      0.84      0.79  43046721
weighted avg       0.87      0.83      0.84  43046721

ROC AUC: 0.8356897441259543
PR AUC: 0.9177147763528695
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 29485628, 0: 13561093}
Accuracy: 0.860278277641635
Confusion matrix:
[[ 8646554  1100023]
 [ 4914539 28385605]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.64      0.89      0.74   9746577
           1       0.96      0.85      0.90  33300144

    accuracy                           0.86  43046721
   macro avg       0.80      0.87      0.82  43046721
weighted avg       0.89      0.86      0.87  43046721

ROC AUC: 0.8697771946155405
PR AUC: 0.9347832479848792
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 98, 0: 202}
y_pred {1: 33483810, 0: 9562911}
Accuracy: 0.9017058697688031
Confusion matrix:
[[ 7539124  2207453]
 [ 2023787 31276357]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.79      0.77      0.78   9746577
           1       0.93      0.94      0.94  33300144

    accuracy                           0.90  43046721
   macro avg       0.86      0.86      0.86  43046721
weighted avg       0.90      0.90      0.90  43046721

ROC AUC: 0.8563704644429927
PR AUC: 0.924320223514191
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 98, 0: 202}
y_pred {0: 31314642, 1: 11732079}
Accuracy: 0.45144112138065057
Confusion matrix:
[[ 8723779  1022798]
 [22590863 10709281]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.28      0.90      0.42   9746577
           1       0.91      0.32      0.48  33300144

    accuracy                           0.45  43046721
   macro avg       0.60      0.61      0.45  43046721
weighted avg       0.77      0.45      0.46  43046721

ROC AUC: 0.6083297193135792
PR AUC: 0.8183604922331142
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 98, 0: 202}
y_pred {1: 32592733, 0: 10453988}
Accuracy: 0.8948827949055632
Confusion matrix:
[[ 7837807  1908770]
 [ 2616181 30683963]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.75      0.80      0.78   9746577
           1       0.94      0.92      0.93  33300144

    accuracy                           0.89  43046721
   macro avg       0.85      0.86      0.85  43046721
weighted avg       0.90      0.89      0.90  43046721

ROC AUC: 0.8627981543859253
PR AUC: 0.9282484719590847
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 98, 0: 202}
y_pred {0: 43046721}
Accuracy: 0.22641856972102475
Confusion matrix:
[[ 9746577        0]
 [33300144        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.23      1.00      0.37   9746577
           1       0.00      0.00      0.00  33300144

    accuracy                           0.23  43046721
   macro avg       0.11      0.50      0.18  43046721
weighted avg       0.05      0.23      0.08  43046721

ROC AUC: 0.5
PR AUC: 0.7735814302789753
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 23643288, 0: 19403433}
Accuracy: 0.706768280910409
Confusion matrix:
[[ 8263673  1482904]
 [11139760 22160384]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.43      0.85      0.57   9746577
           1       0.94      0.67      0.78  33300144

    accuracy                           0.71  43046721
   macro avg       0.68      0.76      0.67  43046721
weighted avg       0.82      0.71      0.73  43046721

ROC AUC: 0.7566639931630728
PR AUC: 0.8825186768867955
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 31419675, 0: 11627046}
Accuracy: 0.8939978959140698
Confusion matrix:
[[ 8405290  1341287]
 [ 3221756 30078388]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.72      0.86      0.79   9746577
           1       0.96      0.90      0.93  33300144

    accuracy                           0.89  43046721
   macro avg       0.84      0.88      0.86  43046721
weighted avg       0.90      0.89      0.90  43046721

ROC AUC: 0.8828173888163711
PR AUC: 0.939534986653241
---------- End MLPClassifier ----------

 ---------- SVC ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 43046721}
Accuracy: 0.22641856972102475
Confusion matrix:
[[ 9746577        0]
 [33300144        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.23      1.00      0.37   9746577
           1       0.00      0.00      0.00  33300144

    accuracy                           0.23  43046721
   macro avg       0.11      0.50      0.18  43046721
weighted avg       0.05      0.23      0.08  43046721

ROC AUC: 0.5
PR AUC: 0.7735814302789753
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End SVC ----------



 ########## 400 ##########
0
1
2
3
4
5
6
7
8
9

 ---------- XGBClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 132, 0: 268}
y_pred {1: 29729369, 0: 13317352}
Accuracy: 0.8368437168536019
Confusion matrix:
[[ 8020293  1726284]
 [ 5297059 28003085]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.60      0.82      0.70   9746577
           1       0.94      0.84      0.89  33300144

    accuracy                           0.84  43046721
   macro avg       0.77      0.83      0.79  43046721
weighted avg       0.87      0.84      0.84  43046721

ROC AUC: 0.8319064480625866
PR AUC: 0.9151536088290004
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 30373476, 0: 12673245}
Accuracy: 0.8718517027115724
Confusion matrix:
[[ 8451729  1294848]
 [ 4221516 29078628]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.67      0.87      0.75   9746577
           1       0.96      0.87      0.91  33300144

    accuracy                           0.87  43046721
   macro avg       0.81      0.87      0.83  43046721
weighted avg       0.89      0.87      0.88  43046721

ROC AUC: 0.8701883658772022
PR AUC: 0.9340700416935896
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 132, 0: 268}
y_pred {1: 33456269, 0: 9590452}
Accuracy: 0.9011993735829495
Confusion matrix:
[[ 7541993  2204584]
 [ 2048459 31251685]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.79      0.77      0.78   9746577
           1       0.93      0.94      0.94  33300144

    accuracy                           0.90  43046721
   macro avg       0.86      0.86      0.86  43046721
weighted avg       0.90      0.90      0.90  43046721

ROC AUC: 0.8561471954710697
PR AUC: 0.9242308620564913
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 132, 0: 268}
y_pred {0: 30650146, 1: 12396575}
Accuracy: 0.4620584689830382
Confusion matrix:
[[ 8620052  1126525]
 [22030094 11270050]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.28      0.88      0.43   9746577
           1       0.91      0.34      0.49  33300144

    accuracy                           0.46  43046721
   macro avg       0.60      0.61      0.46  43046721
weighted avg       0.77      0.46      0.48  43046721

ROC AUC: 0.6114284363736979
PR AUC: 0.819454964977788
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 132, 0: 268}
y_pred {1: 36850746, 0: 6195975}
Accuracy: 0.8931975329781797
Confusion matrix:
[[ 5672528  4074049]
 [  523447 32776697]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.92      0.58      0.71   9746577
           1       0.89      0.98      0.93  33300144

    accuracy                           0.89  43046721
   macro avg       0.90      0.78      0.82  43046721
weighted avg       0.90      0.89      0.88  43046721

ROC AUC: 0.7831415049656372
PR AUC: 0.887623338796779
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 132, 0: 268}
y_pred {0: 43046721}
Accuracy: 0.22641856972102475
Confusion matrix:
[[ 9746577        0]
 [33300144        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.23      1.00      0.37   9746577
           1       0.00      0.00      0.00  33300144

    accuracy                           0.23  43046721
   macro avg       0.11      0.50      0.18  43046721
weighted avg       0.05      0.23      0.08  43046721

ROC AUC: 0.5
PR AUC: 0.7735814302789753
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 15178035, 0: 27868686}
Accuracy: 0.5361483863079838
Confusion matrix:
[[ 8823986   922591]
 [19044700 14255444]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.32      0.91      0.47   9746577
           1       0.94      0.43      0.59  33300144

    accuracy                           0.54  43046721
   macro avg       0.63      0.67      0.53  43046721
weighted avg       0.80      0.54      0.56  43046721

ROC AUC: 0.6667158050213523
PR AUC: 0.8444875975048229
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 32553592, 0: 10493129}
Accuracy: 0.908833474215144
Confusion matrix:
[[ 8157643  1588934]
 [ 2335486 30964658]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.78      0.84      0.81   9746577
           1       0.95      0.93      0.94  33300144

    accuracy                           0.91  43046721
   macro avg       0.86      0.88      0.87  43046721
weighted avg       0.91      0.91      0.91  43046721

ROC AUC: 0.8834203822860214
PR AUC: 0.9387337224627114
---------- End MLPClassifier ----------

 ---------- SVC ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 43046684, 1: 37}
Accuracy: 0.22641942925223038
Confusion matrix:
[[ 9746577        0]
 [33300107       37]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.23      1.00      0.37   9746577
           1       1.00      0.00      0.00  33300144

    accuracy                           0.23  43046721
   macro avg       0.61      0.50      0.18  43046721
weighted avg       0.82      0.23      0.08  43046721

ROC AUC: 0.5000005555531531
PR AUC: 0.7735816818540759
---------- End SVC ----------



 ########## 500 ##########
0
1
2
3
4
5
6
7
8
9

 ---------- XGBClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 167, 0: 333}
y_pred {1: 29335103, 0: 13711618}
Accuracy: 0.8479496498699634
Confusion matrix:
[[ 8456463  1290114]
 [ 5255155 28044989]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.62      0.87      0.72   9746577
           1       0.96      0.84      0.90  33300144

    accuracy                           0.85  43046721
   macro avg       0.79      0.85      0.81  43046721
weighted avg       0.88      0.85      0.86  43046721

ROC AUC: 0.8549111823771518
PR AUC: 0.9272303028894541
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 31937209, 0: 11109512}
Accuracy: 0.8888713730367523
Confusion matrix:
[[ 8036183  1710394]
 [ 3073329 30226815]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.72      0.82      0.77   9746577
           1       0.95      0.91      0.93  33300144

    accuracy                           0.89  43046721
   macro avg       0.83      0.87      0.85  43046721
weighted avg       0.90      0.89      0.89  43046721

ROC AUC: 0.8661108009942445
PR AUC: 0.9304912081260941
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 167, 0: 333}
y_pred {1: 34204383, 0: 8842338}
Accuracy: 0.903486795196317
Confusion matrix:
[[ 7217169  2529408]
 [ 1625169 31674975]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.82      0.74      0.78   9746577
           1       0.93      0.95      0.94  33300144

    accuracy                           0.90  43046721
   macro avg       0.87      0.85      0.86  43046721
weighted avg       0.90      0.90      0.90  43046721

ROC AUC: 0.8458393824999046
PR AUC: 0.9186091456025353
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 167, 0: 333}
y_pred {0: 29770462, 1: 13276259}
Accuracy: 0.4819365451784353
Confusion matrix:
[[ 8608053  1138524]
 [21162409 12137735]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.29      0.88      0.44   9746577
           1       0.91      0.36      0.52  33300144

    accuracy                           0.48  43046721
   macro avg       0.60      0.62      0.48  43046721
weighted avg       0.77      0.48      0.50  43046721

ROC AUC: 0.6238411339000999
PR AUC: 0.8248520885290166
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 167, 0: 333}
y_pred {1: 34493031, 0: 8553690}
Accuracy: 0.9027155401685532
Confusion matrix:
[[ 7056245  2690332]
 [ 1497445 31802699]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.82      0.72      0.77   9746577
           1       0.92      0.96      0.94  33300144

    accuracy                           0.90  43046721
   macro avg       0.87      0.84      0.85  43046721
weighted avg       0.90      0.90      0.90  43046721

ROC AUC: 0.839501740879783
PR AUC: 0.9153293402187355
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 167, 0: 333}
y_pred {0: 43046721}
Accuracy: 0.22641856972102475
Confusion matrix:
[[ 9746577        0]
 [33300144        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.23      1.00      0.37   9746577
           1       0.00      0.00      0.00  33300144

    accuracy                           0.23  43046721
   macro avg       0.11      0.50      0.18  43046721
weighted avg       0.05      0.23      0.08  43046721

ROC AUC: 0.5
PR AUC: 0.7735814302789753
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 23438313, 0: 19608408}
Accuracy: 0.7180791308123097
Confusion matrix:
[[ 8609608  1136969]
 [10998800 22301344]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.44      0.88      0.59   9746577
           1       0.95      0.67      0.79  33300144

    accuracy                           0.72  43046721
   macro avg       0.70      0.78      0.69  43046721
weighted avg       0.84      0.72      0.74  43046721

ROC AUC: 0.7765269873193598
PR AUC: 0.8927287419085752
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 33727493, 0: 9319228}
Accuracy: 0.917807189077189
Confusion matrix:
[[ 7763837  1982740]
 [ 1555391 31744753]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.83      0.80      0.81   9746577
           1       0.94      0.95      0.95  33300144

    accuracy                           0.92  43046721
   macro avg       0.89      0.87      0.88  43046721
weighted avg       0.92      0.92      0.92  43046721

ROC AUC: 0.8749311977342844
PR AUC: 0.9333831721771708
---------- End MLPClassifier ----------

 ---------- SVC ----------
(43046721, 8) (43046721,)
y_test {1: 33300144, 0: 9746577} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 43015997, 1: 30724}
Accuracy: 0.2271323058497301
Confusion matrix:
[[ 9746577        0]
 [33269420    30724]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.23      1.00      0.37   9746577
           1       1.00      0.00      0.00  33300144

    accuracy                           0.23  43046721
   macro avg       0.61      0.50      0.19  43046721
weighted avg       0.82      0.23      0.09  43046721

ROC AUC: 0.5004613193264269
PR AUC: 0.7737903328031238
---------- End SVC ----------
roc_metrics
[[0.8182, 0.8301, 0.8518, 0.5964, 0.838, 0.5, 0.5994, 0.8607, 0.5], [0.8233, 0.8699, 0.8476, 0.61, 0.8494, 0.5, 0.6533, 0.8728, 0.5], [0.8357, 0.8698, 0.8564, 0.6083, 0.8628, 0.5, 0.7567, 0.8828, 0.5], [0.8319, 0.8702, 0.8561, 0.6114, 0.7831, 0.5, 0.6667, 0.8834, 0.5], [0.8549, 0.8661, 0.8458, 0.6238, 0.8395, 0.5, 0.7765, 0.8749, 0.5005]]
pr_metrics
[[0.9122, 0.9209, 0.9224, 0.8145, 0.9151, 0.7736, 0.8164, 0.932, 0.7736], [0.9123, 0.9352, 0.9197, 0.8194, 0.9218, 0.7736, 0.8388, 0.9373, 0.7736], [0.9177, 0.9348, 0.9243, 0.8184, 0.9282, 0.7736, 0.8825, 0.9395, 0.7736], [0.9152, 0.9341, 0.9242, 0.8195, 0.8876, 0.7736, 0.8445, 0.9387, 0.7736], [0.9272, 0.9305, 0.9186, 0.8249, 0.9153, 0.7736, 0.8927, 0.9334, 0.7738]]
f1_metrics
[[0.6427, 0.642, 0.7657, 0.4186, 0.7569, 0.3692, 0.4209, 0.7102, 0.3692], [0.6686, 0.7385, 0.7743, 0.4264, 0.7517, 0.3692, 0.4583, 0.7356, 0.3692], [0.6928, 0.7419, 0.7809, 0.4249, 0.776, 0.3692, 0.567, 0.7865, 0.3692], [0.6955, 0.754, 0.7801, 0.4268, 0.7116, 0.3692, 0.4692, 0.8061, 0.3692], [0.721, 0.7706, 0.7765, 0.4357, 0.7712, 0.3692, 0.5866, 0.8144, 0.3695]]


'''









