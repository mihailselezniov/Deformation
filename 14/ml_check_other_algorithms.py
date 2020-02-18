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


from numpy import genfromtxt


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


    mi = list(map(float, x_train.min(axis=0)))
    ma = list(map(float, x_train.max(axis=0)))
    print('length', '{} - {}'.format(mi[0], ma[0]))
    print('diameter', '{} - {}'.format(mi[1], ma[1]))
    print('young', '{} - {}'.format(mi[2], ma[2]))
    print('density', '{} - {}'.format(mi[3], ma[3]))
    print('pressure_time', '{} - {}'.format(mi[4], ma[4]))
    print('pressure_radius', '{} - {}'.format(mi[5], ma[5]))
    print('pressure_amplitude', '{} - {}'.format(mi[6], ma[6]))
    print('strength', '{} - {}'.format(mi[7], ma[7]))


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

    x_train = (x_train - extreme_values.min(axis=0)) / (extreme_values.max(axis=0) - extreme_values.min(axis=0))
    #print(x_train)

    def make_test(num):
        data = genfromtxt('data0{}.csv'.format(num), delimiter=';', skip_header=True)
        x_test, y_test = [], []
        for d in data:
            #pressure_time;pressure_radius;pressure_amplitude;young;density;strength;length;diameter;is_broken
            pt, pr, pa, y, de, s, l, di, b = list(map(float, d))
            x_test.append([l, di, y, de, pt, pr, pa, s])
            y_test.append(b)
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = (x_test - extreme_values.min(axis=0)) / (extreme_values.max(axis=0) - extreme_values.min(axis=0))
        #print(x_test)
        return x_test, y_test


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

    x_test, y_test = [], []
    x_test, y_test = make_test(5)


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
print('roc')
print(list(zip(*roc_metrics)))
print('pr')
print(list(zip(*pr_metrics)))
print('f1')
print(list(zip(*f1_metrics)))

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
length 10.0 - 100.0
diameter 0.01 - 0.5
young 60.0 - 300.0
density 1000.0 - 2000.0
pressure_time 11.11 - 100.0
pressure_radius 0.56 - 5.0
pressure_amplitude 22.22 - 200.0
strength 0.2 - 10.0

 ---------- XGBClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 33, 0: 67}
y_pred {1: 828, 0: 172}
Accuracy: 0.755
Confusion matrix:
[[146 219]
 [ 26 609]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.85      0.40      0.54       365
         1.0       0.74      0.96      0.83       635

   micro avg       0.76      0.76      0.76      1000
   macro avg       0.79      0.68      0.69      1000
weighted avg       0.78      0.76      0.73      1000

ROC AUC: 0.6795275590551182
PR AUC: 0.7313919890448477
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 33, 0: 67}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 691, 0: 309}
Accuracy: 0.836
Confusion matrix:
[[255 110]
 [ 54 581]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.83      0.70      0.76       365
         1.0       0.84      0.91      0.88       635

   micro avg       0.84      0.84      0.84      1000
   macro avg       0.83      0.81      0.82      1000
weighted avg       0.84      0.84      0.83      1000

ROC AUC: 0.8067953834537805
PR AUC: 0.8233084312362547
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 33, 0: 67}
y_pred {1: 904, 0: 96}
Accuracy: 0.731
Confusion matrix:
[[ 96 269]
 [  0 635]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.26      0.42       365
         1.0       0.70      1.00      0.83       635

   micro avg       0.73      0.73      0.73      1000
   macro avg       0.85      0.63      0.62      1000
weighted avg       0.81      0.73      0.68      1000

ROC AUC: 0.6315068493150685
PR AUC: 0.702433628318584
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 33, 0: 67}
y_pred {0: 748, 1: 252}
Accuracy: 0.467
Confusion matrix:
[[290  75]
 [458 177]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.39      0.79      0.52       365
         1.0       0.70      0.28      0.40       635

   micro avg       0.47      0.47      0.47      1000
   macro avg       0.55      0.54      0.46      1000
weighted avg       0.59      0.47      0.44      1000

ROC AUC: 0.5366303527127603
PR AUC: 0.6537817772778403
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 33, 0: 67}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDClassifier in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
  FutureWarning)
y_pred {0: 331, 1: 669}
Accuracy: 0.842
Confusion matrix:
[[269  96]
 [ 62 573]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.81      0.74      0.77       365
         1.0       0.86      0.90      0.88       635

   micro avg       0.84      0.84      0.84      1000
   macro avg       0.83      0.82      0.83      1000
weighted avg       0.84      0.84      0.84      1000

ROC AUC: 0.8196742530471361
PR AUC: 0.8348752515800997
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 33, 0: 67}
y_pred {0: 1000}
Accuracy: 0.365
Confusion matrix:
[[365   0]
 [635   0]]
Precision, recall and f1-score:
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.36      1.00      0.53       365
         1.0       0.00      0.00      0.00       635

   micro avg       0.36      0.36      0.36      1000
   macro avg       0.18      0.50      0.27      1000
weighted avg       0.13      0.36      0.20      1000

ROC AUC: 0.5
PR AUC: 0.635
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 33, 0: 67}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {0: 926, 1: 74}
Accuracy: 0.401
Confusion matrix:
[[346  19]
 [580  55]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.37      0.95      0.54       365
         1.0       0.74      0.09      0.16       635

   micro avg       0.40      0.40      0.40      1000
   macro avg       0.56      0.52      0.35      1000
weighted avg       0.61      0.40      0.29      1000

ROC AUC: 0.5172796893538993
PR AUC: 0.6443753990210683
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 33, 0: 67}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:562: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 721, 0: 279}
Accuracy: 0.858
Confusion matrix:
[[251 114]
 [ 28 607]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.90      0.69      0.78       365
         1.0       0.84      0.96      0.90       635

   micro avg       0.86      0.86      0.86      1000
   macro avg       0.87      0.82      0.84      1000
weighted avg       0.86      0.86      0.85      1000

ROC AUC: 0.8217883723438679
PR AUC: 0.8327637249227341
---------- End MLPClassifier ----------

 ---------- SVC ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 33, 0: 67}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 1000}
Accuracy: 0.365
Confusion matrix:
[[365   0]
 [635   0]]
Precision, recall and f1-score:
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.36      1.00      0.53       365
         1.0       0.00      0.00      0.00       635

   micro avg       0.36      0.36      0.36      1000
   macro avg       0.18      0.50      0.27      1000
weighted avg       0.13      0.36      0.20      1000

ROC AUC: 0.5
PR AUC: 0.635
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
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
length 10.0 - 100.0
diameter 0.01 - 0.5
young 60.0 - 300.0
density 1000.0 - 2000.0
pressure_time 11.11 - 100.0
pressure_radius 0.56 - 5.0
pressure_amplitude 22.22 - 200.0
strength 0.2 - 10.0

 ---------- XGBClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 69, 0: 131}
y_pred {1: 713, 0: 287}
Accuracy: 0.766
Confusion matrix:
[[209 156]
 [ 78 557]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.73      0.57      0.64       365
         1.0       0.78      0.88      0.83       635

   micro avg       0.77      0.77      0.77      1000
   macro avg       0.75      0.72      0.73      1000
weighted avg       0.76      0.77      0.76      1000

ROC AUC: 0.724884047028368
PR AUC: 0.7632469878852801
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 69, 0: 131}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 803, 0: 197}
Accuracy: 0.818
Confusion matrix:
[[190 175]
 [  7 628]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.96      0.52      0.68       365
         1.0       0.78      0.99      0.87       635

   micro avg       0.82      0.82      0.82      1000
   macro avg       0.87      0.75      0.77      1000
weighted avg       0.85      0.82      0.80      1000

ROC AUC: 0.7547621615791178
PR AUC: 0.7804460340651691
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 69, 0: 131}
y_pred {1: 911, 0: 89}
Accuracy: 0.724
Confusion matrix:
[[ 89 276]
 [  0 635]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.24      0.39       365
         1.0       0.70      1.00      0.82       635

   micro avg       0.72      0.72      0.72      1000
   macro avg       0.85      0.62      0.61      1000
weighted avg       0.81      0.72      0.66      1000

ROC AUC: 0.6219178082191781
PR AUC: 0.6970362239297475
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 69, 0: 131}
y_pred {0: 715, 1: 285}
Accuracy: 0.474
Confusion matrix:
[[277  88]
 [438 197]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.39      0.76      0.51       365
         1.0       0.69      0.31      0.43       635

   micro avg       0.47      0.47      0.47      1000
   macro avg       0.54      0.53      0.47      1000
weighted avg       0.58      0.47      0.46      1000

ROC AUC: 0.534570165030741
PR AUC: 0.6524439839756873
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 69, 0: 131}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDClassifier in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
  FutureWarning)
y_pred {1: 818, 0: 182}
Accuracy: 0.805
Confusion matrix:
[[176 189]
 [  6 629]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.97      0.48      0.64       365
         1.0       0.77      0.99      0.87       635

   micro avg       0.81      0.81      0.81      1000
   macro avg       0.87      0.74      0.75      1000
weighted avg       0.84      0.81      0.78      1000

ROC AUC: 0.73637148096214
PR AUC: 0.7676829986716208
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 69, 0: 131}
y_pred {0: 1000}
Accuracy: 0.365
Confusion matrix:
[[365   0]
 [635   0]]
Precision, recall and f1-score:
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.36      1.00      0.53       365
         1.0       0.00      0.00      0.00       635

   micro avg       0.36      0.36      0.36      1000
   macro avg       0.18      0.50      0.27      1000
weighted avg       0.13      0.36      0.20      1000

ROC AUC: 0.5
PR AUC: 0.635
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 69, 0: 131}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {0: 896, 1: 104}
Accuracy: 0.393
Confusion matrix:
[[327  38]
 [569  66]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.36      0.90      0.52       365
         1.0       0.63      0.10      0.18       635

   micro avg       0.39      0.39      0.39      1000
   macro avg       0.50      0.50      0.35      1000
weighted avg       0.54      0.39      0.30      1000

ROC AUC: 0.49991370941645996
PR AUC: 0.6349600242277408
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 69, 0: 131}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:562: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 837, 0: 163}
Accuracy: 0.798
Confusion matrix:
[[163 202]
 [  0 635]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.45      0.62       365
         1.0       0.76      1.00      0.86       635

   micro avg       0.80      0.80      0.80      1000
   macro avg       0.88      0.72      0.74      1000
weighted avg       0.85      0.80      0.77      1000

ROC AUC: 0.7232876712328767
PR AUC: 0.7586618876941458
---------- End MLPClassifier ----------

 ---------- SVC ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 69, 0: 131}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 1000}
Accuracy: 0.365
Confusion matrix:
[[365   0]
 [635   0]]
Precision, recall and f1-score:
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.36      1.00      0.53       365
         1.0       0.00      0.00      0.00       635

   micro avg       0.36      0.36      0.36      1000
   macro avg       0.18      0.50      0.27      1000
weighted avg       0.13      0.36      0.20      1000

ROC AUC: 0.5
PR AUC: 0.635
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
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
length 10.0 - 100.0
diameter 0.01 - 0.5
young 60.0 - 300.0
density 1000.0 - 2000.0
pressure_time 11.11 - 100.0
pressure_radius 0.56 - 5.0
pressure_amplitude 22.22 - 200.0
strength 0.2 - 10.0

 ---------- XGBClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 98, 0: 202}
y_pred {1: 749, 0: 251}
Accuracy: 0.762
Confusion matrix:
[[189 176]
 [ 62 573]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.75      0.52      0.61       365
         1.0       0.77      0.90      0.83       635

   micro avg       0.76      0.76      0.76      1000
   macro avg       0.76      0.71      0.72      1000
weighted avg       0.76      0.76      0.75      1000

ROC AUC: 0.7100852119512459
PR AUC: 0.7523251579533866
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 98, 0: 202}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 847, 0: 153}
Accuracy: 0.782
Confusion matrix:
[[150 215]
 [  3 632]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.98      0.41      0.58       365
         1.0       0.75      1.00      0.85       635

   micro avg       0.78      0.78      0.78      1000
   macro avg       0.86      0.70      0.72      1000
weighted avg       0.83      0.78      0.75      1000

ROC AUC: 0.703117247330385
PR AUC: 0.7456377487937975
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 98, 0: 202}
y_pred {1: 915, 0: 85}
Accuracy: 0.72
Confusion matrix:
[[ 85 280]
 [  0 635]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.23      0.38       365
         1.0       0.69      1.00      0.82       635

   micro avg       0.72      0.72      0.72      1000
   macro avg       0.85      0.62      0.60      1000
weighted avg       0.81      0.72      0.66      1000

ROC AUC: 0.6164383561643836
PR AUC: 0.6939890710382514
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 98, 0: 202}
y_pred {0: 751, 1: 249}
Accuracy: 0.454
Confusion matrix:
[[285  80]
 [466 169]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.38      0.78      0.51       365
         1.0       0.68      0.27      0.38       635

   micro avg       0.45      0.45      0.45      1000
   macro avg       0.53      0.52      0.45      1000
weighted avg       0.57      0.45      0.43      1000

ROC AUC: 0.5234818250458418
PR AUC: 0.6466343484172912
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 98, 0: 202}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDClassifier in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
  FutureWarning)
y_pred {1: 948, 0: 52}
Accuracy: 0.687
Confusion matrix:
[[ 52 313]
 [  0 635]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.14      0.25       365
         1.0       0.67      1.00      0.80       635

   micro avg       0.69      0.69      0.69      1000
   macro avg       0.83      0.57      0.53      1000
weighted avg       0.79      0.69      0.60      1000

ROC AUC: 0.5712328767123287
PR AUC: 0.669831223628692
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 98, 0: 202}
y_pred {0: 1000}
Accuracy: 0.365
Confusion matrix:
[[365   0]
 [635   0]]
Precision, recall and f1-score:
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.36      1.00      0.53       365
         1.0       0.00      0.00      0.00       635

   micro avg       0.36      0.36      0.36      1000
   macro avg       0.18      0.50      0.27      1000
weighted avg       0.13      0.36      0.20      1000

ROC AUC: 0.5
PR AUC: 0.635
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 98, 0: 202}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 210, 0: 790}
Accuracy: 0.459
Confusion matrix:
[[307  58]
 [483 152]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.39      0.84      0.53       365
         1.0       0.72      0.24      0.36       635

   micro avg       0.46      0.46      0.46      1000
   macro avg       0.56      0.54      0.45      1000
weighted avg       0.60      0.46      0.42      1000

ROC AUC: 0.5402329845755582
PR AUC: 0.6562583427071615
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 98, 0: 202}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:562: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 909, 0: 91}
Accuracy: 0.726
Confusion matrix:
[[ 91 274]
 [  0 635]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.25      0.40       365
         1.0       0.70      1.00      0.82       635

   micro avg       0.73      0.73      0.73      1000
   macro avg       0.85      0.62      0.61      1000
weighted avg       0.81      0.73      0.67      1000

ROC AUC: 0.6246575342465753
PR AUC: 0.6985698569856986
---------- End MLPClassifier ----------

 ---------- SVC ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 98, 0: 202}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 1000}
Accuracy: 0.365
Confusion matrix:
[[365   0]
 [635   0]]
Precision, recall and f1-score:
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.36      1.00      0.53       365
         1.0       0.00      0.00      0.00       635

   micro avg       0.36      0.36      0.36      1000
   macro avg       0.18      0.50      0.27      1000
weighted avg       0.13      0.36      0.20      1000

ROC AUC: 0.5
PR AUC: 0.635
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
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
length 10.0 - 100.0
diameter 0.01 - 0.5
young 60.0 - 300.0
density 1000.0 - 2000.0
pressure_time 11.11 - 100.0
pressure_radius 0.56 - 5.0
pressure_amplitude 22.22 - 200.0
strength 0.2 - 10.0

 ---------- XGBClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 132, 0: 268}
y_pred {1: 661, 0: 339}
Accuracy: 0.77
Confusion matrix:
[[237 128]
 [102 533]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.70      0.65      0.67       365
         1.0       0.81      0.84      0.82       635

   micro avg       0.77      0.77      0.77      1000
   macro avg       0.75      0.74      0.75      1000
weighted avg       0.77      0.77      0.77      1000

ROC AUC: 0.744342573616654
PR AUC: 0.7788294280915339
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 132, 0: 268}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 854, 0: 146}
Accuracy: 0.775
Confusion matrix:
[[143 222]
 [  3 632]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.98      0.39      0.56       365
         1.0       0.74      1.00      0.85       635

   micro avg       0.78      0.78      0.78      1000
   macro avg       0.86      0.69      0.70      1000
weighted avg       0.83      0.78      0.74      1000

ROC AUC: 0.6935282062344946
PR AUC: 0.7395505541315532
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 132, 0: 268}
y_pred {1: 910, 0: 90}
Accuracy: 0.725
Confusion matrix:
[[ 90 275]
 [  0 635]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.25      0.40       365
         1.0       0.70      1.00      0.82       635

   micro avg       0.72      0.72      0.73      1000
   macro avg       0.85      0.62      0.61      1000
weighted avg       0.81      0.72      0.67      1000

ROC AUC: 0.6232876712328768
PR AUC: 0.6978021978021978
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 132, 0: 268}
y_pred {0: 848, 1: 152}
Accuracy: 0.433
Confusion matrix:
[[323  42]
 [525 110]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.38      0.88      0.53       365
         1.0       0.72      0.17      0.28       635

   micro avg       0.43      0.43      0.43      1000
   macro avg       0.55      0.53      0.41      1000
weighted avg       0.60      0.43      0.37      1000

ROC AUC: 0.5290799266530041
PR AUC: 0.650362619146291
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 132, 0: 268}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDClassifier in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
  FutureWarning)
y_pred {1: 914, 0: 86}
Accuracy: 0.719
Confusion matrix:
[[ 85 280]
 [  1 634]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.99      0.23      0.38       365
         1.0       0.69      1.00      0.82       635

   micro avg       0.72      0.72      0.72      1000
   macro avg       0.84      0.62      0.60      1000
weighted avg       0.80      0.72      0.66      1000

ROC AUC: 0.6156509545895804
PR AUC: 0.6935618980340804
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 132, 0: 268}
y_pred {0: 1000}
Accuracy: 0.365
Confusion matrix:
[[365   0]
 [635   0]]
Precision, recall and f1-score:
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.36      1.00      0.53       365
         1.0       0.00      0.00      0.00       635

   micro avg       0.36      0.36      0.36      1000
   macro avg       0.18      0.50      0.27      1000
weighted avg       0.13      0.36      0.20      1000

ROC AUC: 0.5
PR AUC: 0.635
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 132, 0: 268}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {0: 864, 1: 136}
Accuracy: 0.461
Confusion matrix:
[[345  20]
 [519 116]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.40      0.95      0.56       365
         1.0       0.85      0.18      0.30       635

   micro avg       0.46      0.46      0.46      1000
   macro avg       0.63      0.56      0.43      1000
weighted avg       0.69      0.46      0.40      1000

ROC AUC: 0.5639413224031927
PR AUC: 0.674812876331635
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 132, 0: 268}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:562: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 900, 0: 100}
Accuracy: 0.735
Confusion matrix:
[[100 265]
 [  0 635]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.27      0.43       365
         1.0       0.71      1.00      0.83       635

   micro avg       0.73      0.73      0.73      1000
   macro avg       0.85      0.64      0.63      1000
weighted avg       0.81      0.73      0.68      1000

ROC AUC: 0.636986301369863
PR AUC: 0.7055555555555556
---------- End MLPClassifier ----------

 ---------- SVC ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 132, 0: 268}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 1000}
Accuracy: 0.365
Confusion matrix:
[[365   0]
 [635   0]]
Precision, recall and f1-score:
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.36      1.00      0.53       365
         1.0       0.00      0.00      0.00       635

   micro avg       0.36      0.36      0.36      1000
   macro avg       0.18      0.50      0.27      1000
weighted avg       0.13      0.36      0.20      1000

ROC AUC: 0.5
PR AUC: 0.635
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
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
length 10.0 - 100.0
diameter 0.01 - 0.5
young 60.0 - 300.0
density 1000.0 - 2000.0
pressure_time 11.11 - 100.0
pressure_radius 0.56 - 5.0
pressure_amplitude 22.22 - 200.0
strength 0.2 - 10.0

 ---------- XGBClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 167, 0: 333}
y_pred {1: 681, 0: 319}
Accuracy: 0.798
Confusion matrix:
[[241 124]
 [ 78 557]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.76      0.66      0.70       365
         1.0       0.82      0.88      0.85       635

   micro avg       0.80      0.80      0.80      1000
   macro avg       0.79      0.77      0.78      1000
weighted avg       0.80      0.80      0.79      1000

ROC AUC: 0.7687196634667242
PR AUC: 0.7954465526610937
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 167, 0: 333}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 902, 0: 98}
Accuracy: 0.733
Confusion matrix:
[[ 98 267]
 [  0 635]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.27      0.42       365
         1.0       0.70      1.00      0.83       635

   micro avg       0.73      0.73      0.73      1000
   macro avg       0.85      0.63      0.62      1000
weighted avg       0.81      0.73      0.68      1000

ROC AUC: 0.6342465753424658
PR AUC: 0.7039911308203991
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 167, 0: 333}
y_pred {1: 926, 0: 74}
Accuracy: 0.709
Confusion matrix:
[[ 74 291]
 [  0 635]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.20      0.34       365
         1.0       0.69      1.00      0.81       635

   micro avg       0.71      0.71      0.71      1000
   macro avg       0.84      0.60      0.58      1000
weighted avg       0.80      0.71      0.64      1000

ROC AUC: 0.6013698630136987
PR AUC: 0.6857451403887689
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 167, 0: 333}
y_pred {0: 831, 1: 169}
Accuracy: 0.458
Confusion matrix:
[[327  38]
 [504 131]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.39      0.90      0.55       365
         1.0       0.78      0.21      0.33       635

   micro avg       0.46      0.46      0.46      1000
   macro avg       0.58      0.55      0.44      1000
weighted avg       0.64      0.46      0.41      1000

ROC AUC: 0.5510948117786647
PR AUC: 0.6639124073987793
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 167, 0: 333}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDClassifier in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
  FutureWarning)
y_pred {1: 994, 0: 6}
Accuracy: 0.641
Confusion matrix:
[[  6 359]
 [  0 635]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.02      0.03       365
         1.0       0.64      1.00      0.78       635

   micro avg       0.64      0.64      0.64      1000
   macro avg       0.82      0.51      0.41      1000
weighted avg       0.77      0.64      0.51      1000

ROC AUC: 0.5082191780821919
PR AUC: 0.6388329979879276
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 167, 0: 333}
y_pred {0: 1000}
Accuracy: 0.365
Confusion matrix:
[[365   0]
 [635   0]]
Precision, recall and f1-score:
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.36      1.00      0.53       365
         1.0       0.00      0.00      0.00       635

   micro avg       0.36      0.36      0.36      1000
   macro avg       0.18      0.50      0.27      1000
weighted avg       0.13      0.36      0.20      1000

ROC AUC: 0.5
PR AUC: 0.635
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 167, 0: 333}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 277, 0: 723}
Accuracy: 0.57
Confusion matrix:
[[329  36]
 [394 241]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.46      0.90      0.60       365
         1.0       0.87      0.38      0.53       635

   micro avg       0.57      0.57      0.57      1000
   macro avg       0.66      0.64      0.57      1000
weighted avg       0.72      0.57      0.56      1000

ROC AUC: 0.6404487110344084
PR AUC: 0.7242026777338753
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 167, 0: 333}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:562: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 936, 0: 64}
Accuracy: 0.699
Confusion matrix:
[[ 64 301]
 [  0 635]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       1.00      0.18      0.30       365
         1.0       0.68      1.00      0.81       635

   micro avg       0.70      0.70      0.70      1000
   macro avg       0.84      0.59      0.55      1000
weighted avg       0.80      0.70      0.62      1000

ROC AUC: 0.5876712328767123
PR AUC: 0.6784188034188035
---------- End MLPClassifier ----------

 ---------- SVC ----------
(1000, 8) (1000,)
y_test {1.0: 635, 0.0: 365} y_train {1: 167, 0: 333}
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 1000}
Accuracy: 0.365
Confusion matrix:
[[365   0]
 [635   0]]
Precision, recall and f1-score:
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.36      1.00      0.53       365
         1.0       0.00      0.00      0.00       635

   micro avg       0.36      0.36      0.36      1000
   macro avg       0.18      0.50      0.27      1000
weighted avg       0.13      0.36      0.20      1000

ROC AUC: 0.5
PR AUC: 0.635
/Users/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End SVC ----------
roc_metrics
[[0.6795, 0.8068, 0.6315, 0.5366, 0.8197, 0.5, 0.5173, 0.8218, 0.5], [0.7249, 0.7548, 0.6219, 0.5346, 0.7364, 0.5, 0.4999, 0.7233, 0.5], [0.7101, 0.7031, 0.6164, 0.5235, 0.5712, 0.5, 0.5402, 0.6247, 0.5], [0.7443, 0.6935, 0.6233, 0.5291, 0.6157, 0.5, 0.5639, 0.637, 0.5], [0.7687, 0.6342, 0.6014, 0.5511, 0.5082, 0.5, 0.6404, 0.5877, 0.5]]
pr_metrics
[[0.7314, 0.8233, 0.7024, 0.6538, 0.8349, 0.635, 0.6444, 0.8328, 0.635], [0.7632, 0.7804, 0.697, 0.6524, 0.7677, 0.635, 0.635, 0.7587, 0.635], [0.7523, 0.7456, 0.694, 0.6466, 0.6698, 0.635, 0.6563, 0.6986, 0.635], [0.7788, 0.7396, 0.6978, 0.6504, 0.6936, 0.635, 0.6748, 0.7056, 0.635], [0.7954, 0.704, 0.6857, 0.6639, 0.6388, 0.635, 0.7242, 0.6784, 0.635]]
f1_metrics
[[0.5438, 0.7567, 0.4165, 0.5211, 0.773, 0.5348, 0.536, 0.7795, 0.5348], [0.6411, 0.6762, 0.3921, 0.513, 0.6435, 0.5348, 0.5186, 0.6174, 0.5348], [0.6136, 0.5792, 0.3778, 0.5108, 0.2494, 0.5348, 0.5316, 0.3991, 0.5348], [0.6733, 0.5597, 0.3956, 0.5326, 0.3769, 0.5348, 0.5614, 0.4301, 0.5348], [0.7047, 0.4233, 0.3371, 0.5468, 0.0323, 0.5348, 0.6048, 0.2984, 0.5348]]

roc
[(0.6795, 0.7249, 0.7101, 0.7443, 0.7687), (0.8068, 0.7548, 0.7031, 0.6935, 0.6342), (0.6315, 0.6219, 0.6164, 0.6233, 0.6014), (0.5366, 0.5346, 0.5235, 0.5291, 0.5511), (0.8197, 0.7364, 0.5712, 0.6157, 0.5082), (0.5, 0.5, 0.5, 0.5, 0.5), (0.5173, 0.4999, 0.5402, 0.5639, 0.6404), (0.8218, 0.7233, 0.6247, 0.637, 0.5877), (0.5, 0.5, 0.5, 0.5, 0.5)]
pr
[(0.7314, 0.7632, 0.7523, 0.7788, 0.7954), (0.8233, 0.7804, 0.7456, 0.7396, 0.704), (0.7024, 0.697, 0.694, 0.6978, 0.6857), (0.6538, 0.6524, 0.6466, 0.6504, 0.6639), (0.8349, 0.7677, 0.6698, 0.6936, 0.6388), (0.635, 0.635, 0.635, 0.635, 0.635), (0.6444, 0.635, 0.6563, 0.6748, 0.7242), (0.8328, 0.7587, 0.6986, 0.7056, 0.6784), (0.635, 0.635, 0.635, 0.635, 0.635)]
f1
[(0.5438, 0.6411, 0.6136, 0.6733, 0.7047), (0.7567, 0.6762, 0.5792, 0.5597, 0.4233), (0.4165, 0.3921, 0.3778, 0.3956, 0.3371), (0.5211, 0.513, 0.5108, 0.5326, 0.5468), (0.773, 0.6435, 0.2494, 0.3769, 0.0323), (0.5348, 0.5348, 0.5348, 0.5348, 0.5348), (0.536, 0.5186, 0.5316, 0.5614, 0.6048), (0.7795, 0.6174, 0.3991, 0.4301, 0.2984), (0.5348, 0.5348, 0.5348, 0.5348, 0.5348)]
'''


