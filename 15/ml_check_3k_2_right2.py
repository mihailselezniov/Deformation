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

Y_test = []
for i, val in enumerate(data_is_broken):
    Y_test.extend([i%2]*val)

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


i = 0
x_test, y_test = [], []
for i0, l in e2_0:
    for i1, di in e2_1:
        for i2, y in e2_2:
            for i3, de in e2_3:
                for i4, pt in e2_4:
                    for i5, pr in e2_5:
                        for i6, pa in e2_6:
                            for i7, s in e2_7:
                                if 0 not in [i0, i1, i2, i3, i4, i5, i6, i7]:
                                    x_test.append([l, di, y, de, pt, pr, pa, s])
                                    y_test.append(Y_test[i])
                                i += 1
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
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 33, 0: 67}
y_pred {1: 10320303, 0: 6456913}
Accuracy: 0.7356359958648682
Confusion matrix:
[[ 2159371   137750]
 [ 4297542 10182553]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.33      0.94      0.49   2297121
           1       0.99      0.70      0.82  14480095

    accuracy                           0.74  16777216
   macro avg       0.66      0.82      0.66  16777216
weighted avg       0.90      0.74      0.78  16777216

ROC AUC: 0.8216220029942389
PR AUC: 0.9499777530988741
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 9238624, 0: 7538592}
Accuracy: 0.6871145367622375
Confusion matrix:
[[2293183    3938]
 [5245409 9234686]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.30      1.00      0.47   2297121
           1       1.00      0.64      0.78  14480095

    accuracy                           0.69  16777216
   macro avg       0.65      0.82      0.62  16777216
weighted avg       0.90      0.69      0.74  16777216

ROC AUC: 0.8180180270405486
PR AUC: 0.9501292704040744
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 33, 0: 67}
y_pred {1: 13518601, 0: 3258615}
Accuracy: 0.907004714012146
Confusion matrix:
[[ 1997767   299354]
 [ 1260848 13219247]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.61      0.87      0.72   2297121
           1       0.98      0.91      0.94  14480095

    accuracy                           0.91  16777216
   macro avg       0.80      0.89      0.83  16777216
weighted avg       0.93      0.91      0.91  16777216

ROC AUC: 0.8913041985890946
PR AUC: 0.967862137864688
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 33, 0: 67}
y_pred {0: 13143396, 1: 3633820}
Accuracy: 0.3425009846687317
Confusion matrix:
[[ 2204757    92364]
 [10938639  3541456]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.17      0.96      0.29   2297121
           1       0.97      0.24      0.39  14480095

    accuracy                           0.34  16777216
   macro avg       0.57      0.60      0.34  16777216
weighted avg       0.86      0.34      0.38  16777216

ROC AUC: 0.6021827472377926
PR AUC: 0.8903512232116357
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 33, 0: 67}
y_pred {1: 13745223, 0: 3031993}
Accuracy: 0.9129761457443237
Confusion matrix:
[[ 1934548   362573]
 [ 1097445 13382650]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.64      0.84      0.73   2297121
           1       0.97      0.92      0.95  14480095

    accuracy                           0.91  16777216
   macro avg       0.81      0.88      0.84  16777216
weighted avg       0.93      0.91      0.92  16777216

ROC AUC: 0.8831860449479608
PR AUC: 0.9652440007668962
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 33, 0: 67}
y_pred {0: 16777216}
Accuracy: 0.1369190812110901
Confusion matrix:
[[ 2297121        0]
 [14480095        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.14      1.00      0.24   2297121
           1       0.00      0.00      0.00  14480095

    accuracy                           0.14  16777216
   macro avg       0.07      0.50      0.12  16777216
weighted avg       0.02      0.14      0.03  16777216

ROC AUC: 0.5
PR AUC: 0.8630809187889099
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {0: 15240488, 1: 1536728}
Accuracy: 0.2271992564201355
Confusion matrix:
[[ 2286082    11039]
 [12954406  1525689]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.15      1.00      0.26   2297121
           1       0.99      0.11      0.19  14480095

    accuracy                           0.23  16777216
   macro avg       0.57      0.55      0.23  16777216
weighted avg       0.88      0.23      0.20  16777216

ROC AUC: 0.5502794953036221
PR AUC: 0.8767504585354068
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 9904033, 0: 6873183}
Accuracy: 0.7255830764770508
Confusion matrix:
[[2283176   13945]
 [4590007 9890088]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.33      0.99      0.50   2297121
           1       1.00      0.68      0.81  14480095

    accuracy                           0.73  16777216
   macro avg       0.67      0.84      0.65  16777216
weighted avg       0.91      0.73      0.77  16777216

ROC AUC: 0.8384710018208726
PR AUC: 0.955636692545087
---------- End MLPClassifier ----------

 ---------- SVC ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 16777216}
Accuracy: 0.1369190812110901
Confusion matrix:
[[ 2297121        0]
 [14480095        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.14      1.00      0.24   2297121
           1       0.00      0.00      0.00  14480095

    accuracy                           0.14  16777216
   macro avg       0.07      0.50      0.12  16777216
weighted avg       0.02      0.14      0.03  16777216

ROC AUC: 0.5
PR AUC: 0.8630809187889099
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
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 69, 0: 131}
y_pred {0: 4403987, 1: 12373229}
Accuracy: 0.8207904100418091
Confusion matrix:
[[ 1847235   449886]
 [ 2556752 11923343]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.42      0.80      0.55   2297121
           1       0.96      0.82      0.89  14480095

    accuracy                           0.82  16777216
   macro avg       0.69      0.81      0.72  16777216
weighted avg       0.89      0.82      0.84  16777216

ROC AUC: 0.8137910643595393
PR AUC: 0.9458845782950277
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 11926298, 0: 4850918}
Accuracy: 0.8367909789085388
Confusion matrix:
[[ 2204923    92198]
 [ 2645995 11834100]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.45      0.96      0.62   2297121
           1       0.99      0.82      0.90  14480095

    accuracy                           0.84  16777216
   macro avg       0.72      0.89      0.76  16777216
weighted avg       0.92      0.84      0.86  16777216

ROC AUC: 0.8885652051888929
PR AUC: 0.9686623289352791
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 69, 0: 131}
y_pred {1: 13923592, 0: 2853624}
Accuracy: 0.9256812930107117
Confusion matrix:
[[ 1951942   345179]
 [  901682 13578413]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.68      0.85      0.76   2297121
           1       0.98      0.94      0.96  14480095

    accuracy                           0.93  16777216
   macro avg       0.83      0.89      0.86  16777216
weighted avg       0.94      0.93      0.93  16777216

ROC AUC: 0.8937318161849735
PR AUC: 0.9682267859798876
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 69, 0: 131}
y_pred {0: 11951635, 1: 4825581}
Accuracy: 0.4044630527496338
Confusion matrix:
[[2128652  168469]
 [9822983 4657112]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.18      0.93      0.30   2297121
           1       0.97      0.32      0.48  14480095

    accuracy                           0.40  16777216
   macro avg       0.57      0.62      0.39  16777216
weighted avg       0.86      0.40      0.46  16777216

ROC AUC: 0.6241412269887419
PR AUC: 0.8958887164641594
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 69, 0: 131}
y_pred {1: 13200962, 0: 3576254}
Accuracy: 0.890839159488678
Confusion matrix:
[[ 2020980   276141]
 [ 1555274 12924821]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.57      0.88      0.69   2297121
           1       0.98      0.89      0.93  14480095

    accuracy                           0.89  16777216
   macro avg       0.77      0.89      0.81  16777216
weighted avg       0.92      0.89      0.90  16777216

ROC AUC: 0.8861902505644524
PR AUC: 0.9666223666953273
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 69, 0: 131}
y_pred {0: 16777216}
Accuracy: 0.1369190812110901
Confusion matrix:
[[ 2297121        0]
 [14480095        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.14      1.00      0.24   2297121
           1       0.00      0.00      0.00  14480095

    accuracy                           0.14  16777216
   macro avg       0.07      0.50      0.12  16777216
weighted avg       0.02      0.14      0.03  16777216

ROC AUC: 0.5
PR AUC: 0.8630809187889099
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {0: 12272646, 1: 4504570}
Accuracy: 0.39166468381881714
Confusion matrix:
[[ 2181797   115324]
 [10090849  4389246]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.18      0.95      0.30   2297121
           1       0.97      0.30      0.46  14480095

    accuracy                           0.39  16777216
   macro avg       0.58      0.63      0.38  16777216
weighted avg       0.87      0.39      0.44  16777216

ROC AUC: 0.6264595117609452
PR AUC: 0.896823791563236
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 11548964, 0: 5228252}
Accuracy: 0.8173250555992126
Confusion matrix:
[[ 2230298    66823]
 [ 2997954 11482141]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.43      0.97      0.59   2297121
           1       0.99      0.79      0.88  14480095

    accuracy                           0.82  16777216
   macro avg       0.71      0.88      0.74  16777216
weighted avg       0.92      0.82      0.84  16777216

ROC AUC: 0.8819352225124416
PR AUC: 0.9670642032874871
---------- End MLPClassifier ----------

 ---------- SVC ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 16777216}
Accuracy: 0.1369190812110901
Confusion matrix:
[[ 2297121        0]
 [14480095        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.14      1.00      0.24   2297121
           1       0.00      0.00      0.00  14480095

    accuracy                           0.14  16777216
   macro avg       0.07      0.50      0.12  16777216
weighted avg       0.02      0.14      0.03  16777216

ROC AUC: 0.5
PR AUC: 0.8630809187889099
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
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 98, 0: 202}
y_pred {1: 12918307, 0: 3858909}
Accuracy: 0.8537349700927734
Confusion matrix:
[[ 1851055   446066]
 [ 2007854 12472241]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.48      0.81      0.60   2297121
           1       0.97      0.86      0.91  14480095

    accuracy                           0.85  16777216
   macro avg       0.72      0.83      0.76  16777216
weighted avg       0.90      0.85      0.87  16777216

ROC AUC: 0.8335760758037978
PR AUC: 0.9512726280902186
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 12137889, 0: 4639327}
Accuracy: 0.8505421876907349
Confusion matrix:
[[ 2214481    82640]
 [ 2424846 12055249]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.48      0.96      0.64   2297121
           1       0.99      0.83      0.91  14480095

    accuracy                           0.85  16777216
   macro avg       0.74      0.90      0.77  16777216
weighted avg       0.92      0.85      0.87  16777216

ROC AUC: 0.8982819458289274
PR AUC: 0.971403154639343
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 98, 0: 202}
y_pred {1: 13817553, 0: 2959663}
Accuracy: 0.9283089637756348
Confusion matrix:
[[ 2027004   270117]
 [  932659 13547436]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.68      0.88      0.77   2297121
           1       0.98      0.94      0.96  14480095

    accuracy                           0.93  16777216
   macro avg       0.83      0.91      0.86  16777216
weighted avg       0.94      0.93      0.93  16777216

ROC AUC: 0.909000452760626
PR AUC: 0.9728913835202854
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 98, 0: 202}
y_pred {0: 11798455, 1: 4978761}
Accuracy: 0.41220784187316895
Confusion matrix:
[[2117030  180091]
 [9681425 4798670]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.18      0.92      0.30   2297121
           1       0.96      0.33      0.49  14480095

    accuracy                           0.41  16777216
   macro avg       0.57      0.63      0.40  16777216
weighted avg       0.86      0.41      0.47  16777216

ROC AUC: 0.6264995591568355
PR AUC: 0.8964683179886697
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 98, 0: 202}
y_pred {1: 13394918, 0: 3382298}
Accuracy: 0.911088764667511
Confusion matrix:
[[ 2093868   203253]
 [ 1288430 13191665]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.62      0.91      0.74   2297121
           1       0.98      0.91      0.95  14480095

    accuracy                           0.91  16777216
   macro avg       0.80      0.91      0.84  16777216
weighted avg       0.93      0.91      0.92  16777216

ROC AUC: 0.9112694929598848
PR AUC: 0.9739932981567361
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 98, 0: 202}
y_pred {0: 16777216}
Accuracy: 0.1369190812110901
Confusion matrix:
[[ 2297121        0]
 [14480095        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.14      1.00      0.24   2297121
           1       0.00      0.00      0.00  14480095

    accuracy                           0.14  16777216
   macro avg       0.07      0.50      0.12  16777216
weighted avg       0.02      0.14      0.03  16777216

ROC AUC: 0.5
PR AUC: 0.8630809187889099
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 9971077, 0: 6806139}
Accuracy: 0.7073688507080078
Confusion matrix:
[[2096862  200259]
 [4709277 9770818]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.31      0.91      0.46   2297121
           1       0.98      0.67      0.80  14480095

    accuracy                           0.71  16777216
   macro avg       0.64      0.79      0.63  16777216
weighted avg       0.89      0.71      0.75  16777216

ROC AUC: 0.7937987833498727
PR AUC: 0.9419184141767607
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 13714636, 0: 3062580}
Accuracy: 0.929169237613678
Confusion matrix:
[[ 2085679   211442]
 [  976901 13503194]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.68      0.91      0.78   2297121
           1       0.98      0.93      0.96  14480095

    accuracy                           0.93  16777216
   macro avg       0.83      0.92      0.87  16777216
weighted avg       0.94      0.93      0.93  16777216

ROC AUC: 0.9202441906836089
PR AUC: 0.9763856153743451
---------- End MLPClassifier ----------

 ---------- SVC ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 16777216}
Accuracy: 0.1369190812110901
Confusion matrix:
[[ 2297121        0]
 [14480095        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.14      1.00      0.24   2297121
           1       0.00      0.00      0.00  14480095

    accuracy                           0.14  16777216
   macro avg       0.07      0.50      0.12  16777216
weighted avg       0.02      0.14      0.03  16777216

ROC AUC: 0.5
PR AUC: 0.8630809187889099
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
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 132, 0: 268}
y_pred {0: 3955227, 1: 12821989}
Accuracy: 0.8419500589370728
Confusion matrix:
[[ 1800355   496766]
 [ 2154872 12325223]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.46      0.78      0.58   2297121
           1       0.96      0.85      0.90  14480095

    accuracy                           0.84  16777216
   macro avg       0.71      0.82      0.74  16777216
weighted avg       0.89      0.84      0.86  16777216

ROC AUC: 0.8174639679484806
PR AUC: 0.9466465691939516
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 12371886, 0: 4405330}
Accuracy: 0.8644232153892517
Confusion matrix:
[[ 2213925    83196]
 [ 2191405 12288690]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.50      0.96      0.66   2297121
           1       0.99      0.85      0.92  14480095

    accuracy                           0.86  16777216
   macro avg       0.75      0.91      0.79  16777216
weighted avg       0.93      0.86      0.88  16777216

ROC AUC: 0.9062216799031849
PR AUC: 0.9735718792427673
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 132, 0: 268}
y_pred {1: 13705745, 0: 3071471}
Accuracy: 0.925989031791687
Confusion matrix:
[[ 2063447   233674]
 [ 1008024 13472071]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.67      0.90      0.77   2297121
           1       0.98      0.93      0.96  14480095

    accuracy                           0.93  16777216
   macro avg       0.83      0.91      0.86  16777216
weighted avg       0.94      0.93      0.93  16777216

ROC AUC: 0.9143304077377848
PR AUC: 0.974605986297138
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 132, 0: 268}
y_pred {0: 11940205, 1: 4837011}
Accuracy: 0.4058626890182495
Confusion matrix:
[[2134678  162443]
 [9805527 4674568]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.18      0.93      0.30   2297121
           1       0.97      0.32      0.48  14480095

    accuracy                           0.41  16777216
   macro avg       0.57      0.63      0.39  16777216
weighted avg       0.86      0.41      0.46  16777216

ROC AUC: 0.6260556272985273
PR AUC: 0.8964405020390366
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 132, 0: 268}
y_pred {1: 15096878, 0: 1680338}
Accuracy: 0.9434977173805237
Confusion matrix:
[[ 1514754   782367]
 [  165584 14314511]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.90      0.66      0.76   2297121
           1       0.95      0.99      0.97  14480095

    accuracy                           0.94  16777216
   macro avg       0.92      0.82      0.86  16777216
weighted avg       0.94      0.94      0.94  16777216

ROC AUC: 0.8239894130026147
PR AUC: 0.947203804476906
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 132, 0: 268}
y_pred {0: 16777216}
Accuracy: 0.1369190812110901
Confusion matrix:
[[ 2297121        0]
 [14480095        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.14      1.00      0.24   2297121
           1       0.00      0.00      0.00  14480095

    accuracy                           0.14  16777216
   macro avg       0.07      0.50      0.12  16777216
weighted avg       0.02      0.14      0.03  16777216

ROC AUC: 0.5
PR AUC: 0.8630809187889099
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 5236527, 0: 11540689}
Accuracy: 0.4322168827056885
Confusion matrix:
[[2155995  141126]
 [9384694 5095401]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.19      0.94      0.31   2297121
           1       0.97      0.35      0.52  14480095

    accuracy                           0.43  16777216
   macro avg       0.58      0.65      0.41  16777216
weighted avg       0.87      0.43      0.49  16777216

ROC AUC: 0.6452269966314694
PR AUC: 0.9017778342491427
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 13331299, 0: 3445917}
Accuracy: 0.9163366556167603
Confusion matrix:
[[ 2169700   127421]
 [ 1276217 13203878]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.63      0.94      0.76   2297121
           1       0.99      0.91      0.95  14480095

    accuracy                           0.92  16777216
   macro avg       0.81      0.93      0.85  16777216
weighted avg       0.94      0.92      0.92  16777216

ROC AUC: 0.9281970882515922
PR AUC: 0.979216879108687
---------- End MLPClassifier ----------

 ---------- SVC ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 16777216}
Accuracy: 0.1369190812110901
Confusion matrix:
[[ 2297121        0]
 [14480095        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.14      1.00      0.24   2297121
           1       0.00      0.00      0.00  14480095

    accuracy                           0.14  16777216
   macro avg       0.07      0.50      0.12  16777216
weighted avg       0.02      0.14      0.03  16777216

ROC AUC: 0.5
PR AUC: 0.8630809187889099
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
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
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 167, 0: 333}
y_pred {1: 12002714, 0: 4774502}
Accuracy: 0.8262931704521179
Confusion matrix:
[[ 2078653   218468]
 [ 2695849 11784246]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.44      0.90      0.59   2297121
           1       0.98      0.81      0.89  14480095

    accuracy                           0.83  16777216
   macro avg       0.71      0.86      0.74  16777216
weighted avg       0.91      0.83      0.85  16777216

ROC AUC: 0.8593593352794018
PR AUC: 0.9596960716127653
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 13007825, 0: 3769391}
Accuracy: 0.8954943418502808
Confusion matrix:
[[ 2156599   140522]
 [ 1612792 12867303]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.57      0.94      0.71   2297121
           1       0.99      0.89      0.94  14480095

    accuracy                           0.90  16777216
   macro avg       0.78      0.91      0.82  16777216
weighted avg       0.93      0.90      0.91  16777216

ROC AUC: 0.9137234866299201
PR AUC: 0.975150304177939
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 167, 0: 333}
y_pred {1: 13983134, 0: 2794082}
Accuracy: 0.9345173239707947
Confusion matrix:
[[ 1996293   300828]
 [  797789 13682306]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.71      0.87      0.78   2297121
           1       0.98      0.94      0.96  14480095

    accuracy                           0.93  16777216
   macro avg       0.85      0.91      0.87  16777216
weighted avg       0.94      0.93      0.94  16777216

ROC AUC: 0.906972863653121
PR AUC: 0.9721280396314192
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 167, 0: 333}
y_pred {0: 11907876, 1: 4869340}
Accuracy: 0.410145103931427
Confusion matrix:
[[2154437  142684]
 [9753439 4726656]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.18      0.94      0.30   2297121
           1       0.97      0.33      0.49  14480095

    accuracy                           0.41  16777216
   macro avg       0.58      0.63      0.40  16777216
weighted avg       0.86      0.41      0.46  16777216

ROC AUC: 0.6321550525679537
PR AUC: 0.8982095833077233
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 167, 0: 333}
y_pred {1: 14048151, 0: 2729065}
Accuracy: 0.9353750944137573
Confusion matrix:
[[ 1970980   326141]
 [  758085 13722010]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.72      0.86      0.78   2297121
           1       0.98      0.95      0.96  14480095

    accuracy                           0.94  16777216
   macro avg       0.85      0.90      0.87  16777216
weighted avg       0.94      0.94      0.94  16777216

ROC AUC: 0.9028341263109226
PR AUC: 0.9708312940967774
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 167, 0: 333}
y_pred {0: 16777216}
Accuracy: 0.1369190812110901
Confusion matrix:
[[ 2297121        0]
 [14480095        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.14      1.00      0.24   2297121
           1       0.00      0.00      0.00  14480095

    accuracy                           0.14  16777216
   macro avg       0.07      0.50      0.12  16777216
weighted avg       0.02      0.14      0.03  16777216

ROC AUC: 0.5
PR AUC: 0.8630809187889099
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 8859280, 0: 7917936}
Accuracy: 0.64217609167099
Confusion matrix:
[[2105884  191237]
 [5812052 8668043]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.27      0.92      0.41   2297121
           1       0.98      0.60      0.74  14480095

    accuracy                           0.64  16777216
   macro avg       0.62      0.76      0.58  16777216
weighted avg       0.88      0.64      0.70  16777216

ROC AUC: 0.7576835484061278
PR AUC: 0.9321213178955169
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 13898402, 0: 2878814}
Accuracy: 0.9413538575172424
Confusion matrix:
[[ 2096008   201113]
 [  782806 13697289]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.73      0.91      0.81   2297121
           1       0.99      0.95      0.97  14480095

    accuracy                           0.94  16777216
   macro avg       0.86      0.93      0.89  16777216
weighted avg       0.95      0.94      0.94  16777216

ROC AUC: 0.929194571130669
PR AUC: 0.9789100880909173
---------- End MLPClassifier ----------

 ---------- SVC ----------
(16777216, 8) (16777216,)
y_test {1: 14480095, 0: 2297121} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 16775476, 1: 1740}
Accuracy: 0.13702279329299927
Confusion matrix:
[[ 2297121        0]
 [14478355     1740]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.14      1.00      0.24   2297121
           1       1.00      0.00      0.00  14480095

    accuracy                           0.14  16777216
   macro avg       0.57      0.50      0.12  16777216
weighted avg       0.88      0.14      0.03  16777216

ROC AUC: 0.5000600824787407
PR AUC: 0.863097371664482
---------- End SVC ----------
roc_metrics
[[0.8216, 0.818, 0.8913, 0.6022, 0.8832, 0.5, 0.5503, 0.8385, 0.5], [0.8138, 0.8886, 0.8937, 0.6241, 0.8862, 0.5, 0.6265, 0.8819, 0.5], [0.8336, 0.8983, 0.909, 0.6265, 0.9113, 0.5, 0.7938, 0.9202, 0.5], [0.8175, 0.9062, 0.9143, 0.6261, 0.824, 0.5, 0.6452, 0.9282, 0.5], [0.8594, 0.9137, 0.907, 0.6322, 0.9028, 0.5, 0.7577, 0.9292, 0.5001]]
pr_metrics
[[0.95, 0.9501, 0.9679, 0.8904, 0.9652, 0.8631, 0.8768, 0.9556, 0.8631], [0.9459, 0.9687, 0.9682, 0.8959, 0.9666, 0.8631, 0.8968, 0.9671, 0.8631], [0.9513, 0.9714, 0.9729, 0.8965, 0.974, 0.8631, 0.9419, 0.9764, 0.8631], [0.9466, 0.9736, 0.9746, 0.8964, 0.9472, 0.8631, 0.9018, 0.9792, 0.8631], [0.9597, 0.9752, 0.9721, 0.8982, 0.9708, 0.8631, 0.9321, 0.9789, 0.8631]]
f1_metrics
[[0.4933, 0.4663, 0.7192, 0.2856, 0.726, 0.2409, 0.2607, 0.4979, 0.2409], [0.5513, 0.6169, 0.7579, 0.2988, 0.6882, 0.2409, 0.2995, 0.5927, 0.2409], [0.6014, 0.6385, 0.7712, 0.3004, 0.7374, 0.2409, 0.4607, 0.7783, 0.2409], [0.5759, 0.6606, 0.7687, 0.2999, 0.7617, 0.2409, 0.3116, 0.7556, 0.2409], [0.5879, 0.711, 0.7842, 0.3033, 0.7843, 0.2409, 0.4123, 0.8099, 0.2409]]

'''









