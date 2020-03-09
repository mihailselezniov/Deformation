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


fix_rows = '''
3
18
35
39
78
117
130
149
162
178
184
187
204
206
233
242
253
262
264
279
284
297
302
305
311
313
345
354
360
372
374
388
415
426
438
466
480
481
496
514
517
521
533
547
564
585
587
592
608
613
618
624
630
632
640
660
663
680
683
685
709
715
716
719
739
744
746
747
753
759
784
790
806
813
832
837
852
857
859
869
871
880
895
903
916
918
921
940
945
947
960
967
993
'''
fix_rows = list(map(int, fix_rows.split()))


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
        data = genfromtxt('../14/data0{}.csv'.format(num), delimiter=';', skip_header=True)
        x_test, y_test = [], []
        for i, d in enumerate(data):
            #pressure_time;pressure_radius;pressure_amplitude;young;density;strength;length;diameter;is_broken
            pt, pr, pa, y, de, s, l, di, b = list(map(float, d))
            x_test.append([l, di, y, de, pt, pr, pa, s])
            if i in fix_rows:
                print(i, b)
                y_test.append(int(not b))
            else:
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


def charts(metrics):
    s = ''
    for i, val in enumerate(metrics):
        s += "['{}00', {}],\n".format(i+1, ', '.join(map(str, val)))
    print(s)
#charts(roc_metrics)

print('roc_metrics')
print(roc_metrics)
print('pr_metrics')
print(pr_metrics)
print('f1_metrics')
print(f1_metrics)
print()
print('roc')
print(charts(roc_metrics))
print('pr')
print(charts(pr_metrics))
print('f1')
print(charts(f1_metrics))



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
3 0.0
18 0.0
35 0.0
39 0.0
78 0.0
117 0.0
130 0.0
149 0.0
162 0.0
178 0.0
184 0.0
187 0.0
204 0.0
206 0.0
233 0.0
242 0.0
253 0.0
262 0.0
264 0.0
279 0.0
284 0.0
297 0.0
302 0.0
305 0.0
311 0.0
313 0.0
345 0.0
354 0.0
360 0.0
372 0.0
374 0.0
388 0.0
415 0.0
426 0.0
438 0.0
466 0.0
480 0.0
481 0.0
496 0.0
514 0.0
517 0.0
521 0.0
533 0.0
547 0.0
564 0.0
585 0.0
587 0.0
592 0.0
608 0.0
613 0.0
618 0.0
624 0.0
630 0.0
632 0.0
640 0.0
660 0.0
663 0.0
680 0.0
683 0.0
685 0.0
709 0.0
715 0.0
716 0.0
719 0.0
739 0.0
744 0.0
746 0.0
747 0.0
753 0.0
759 0.0
784 0.0
790 0.0
806 0.0
813 0.0
832 0.0
837 0.0
852 0.0
857 0.0
859 0.0
869 0.0
871 0.0
880 0.0
895 0.0
903 0.0
916 0.0
918 0.0
921 0.0
940 0.0
945 0.0
947 0.0
960 0.0
967 0.0
993 0.0

 ---------- XGBClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 33, 0: 67}
y_pred {1: 828, 0: 172}
Accuracy: 0.788
Confusion matrix:
[[116 156]
 [ 56 672]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.67      0.43      0.52       272
         1.0       0.81      0.92      0.86       728

    accuracy                           0.79      1000
   macro avg       0.74      0.67      0.69      1000
weighted avg       0.77      0.79      0.77      1000

ROC AUC: 0.6747737556561086
PR AUC: 0.8051638795986622
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 691, 0: 309}
Accuracy: 0.817
Confusion matrix:
[[199  73]
 [110 618]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.64      0.73      0.69       272
         1.0       0.89      0.85      0.87       728

    accuracy                           0.82      1000
   macro avg       0.77      0.79      0.78      1000
weighted avg       0.83      0.82      0.82      1000

ROC AUC: 0.7902593729799613
PR AUC: 0.8692197961228352
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 33, 0: 67}
y_pred {1: 904, 0: 96}
Accuracy: 0.812
Confusion matrix:
[[ 90 182]
 [  6 722]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.94      0.33      0.49       272
         1.0       0.80      0.99      0.88       728

    accuracy                           0.81      1000
   macro avg       0.87      0.66      0.69      1000
weighted avg       0.84      0.81      0.78      1000

ROC AUC: 0.6613202973497092
PR AUC: 0.7980901001653214
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 33, 0: 67}
y_pred {0: 748, 1: 252}
Accuracy: 0.41
Confusion matrix:
[[215  57]
 [533 195]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.29      0.79      0.42       272
         1.0       0.77      0.27      0.40       728

    accuracy                           0.41      1000
   macro avg       0.53      0.53      0.41      1000
weighted avg       0.64      0.41      0.40      1000

ROC AUC: 0.5291491596638656
PR AUC: 0.7402704081632654
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 33, 0: 67}
y_pred {1: 941, 0: 59}
Accuracy: 0.781
Confusion matrix:
[[ 56 216]
 [  3 725]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.95      0.21      0.34       272
         1.0       0.77      1.00      0.87       728

    accuracy                           0.78      1000
   macro avg       0.86      0.60      0.60      1000
weighted avg       0.82      0.78      0.72      1000

ROC AUC: 0.6008807369101488
PR AUC: 0.7702820006773248
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 33, 0: 67}
y_pred {0: 1000}
Accuracy: 0.272
Confusion matrix:
[[272   0]
 [728   0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.27      1.00      0.43       272
         1.0       0.00      0.00      0.00       728

    accuracy                           0.27      1000
   macro avg       0.14      0.50      0.21      1000
weighted avg       0.07      0.27      0.12      1000

ROC AUC: 0.5
PR AUC: 0.728
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {0: 926, 1: 74}
Accuracy: 0.318
Confusion matrix:
[[258  14]
 [668  60]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.28      0.95      0.43       272
         1.0       0.81      0.08      0.15       728

    accuracy                           0.32      1000
   macro avg       0.54      0.52      0.29      1000
weighted avg       0.67      0.32      0.23      1000

ROC AUC: 0.5154734970911441
PR AUC: 0.7348250668250668
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 735, 0: 265}
Accuracy: 0.855
Confusion matrix:
[[196  76]
 [ 69 659]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.74      0.72      0.73       272
         1.0       0.90      0.91      0.90       728

    accuracy                           0.85      1000
   macro avg       0.82      0.81      0.82      1000
weighted avg       0.85      0.85      0.85      1000

ROC AUC: 0.8129040077569489
PR AUC: 0.8806188233535172
---------- End MLPClassifier ----------

 ---------- SVC ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 1000}
Accuracy: 0.272
Confusion matrix:
[[272   0]
 [728   0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.27      1.00      0.43       272
         1.0       0.00      0.00      0.00       728

    accuracy                           0.27      1000
   macro avg       0.14      0.50      0.21      1000
weighted avg       0.07      0.27      0.12      1000

ROC AUC: 0.5
PR AUC: 0.728
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
length 10.0 - 100.0
diameter 0.01 - 0.5
young 60.0 - 300.0
density 1000.0 - 2000.0
pressure_time 11.11 - 100.0
pressure_radius 0.56 - 5.0
pressure_amplitude 22.22 - 200.0
strength 0.2 - 10.0
3 0.0
18 0.0
35 0.0
39 0.0
78 0.0
117 0.0
130 0.0
149 0.0
162 0.0
178 0.0
184 0.0
187 0.0
204 0.0
206 0.0
233 0.0
242 0.0
253 0.0
262 0.0
264 0.0
279 0.0
284 0.0
297 0.0
302 0.0
305 0.0
311 0.0
313 0.0
345 0.0
354 0.0
360 0.0
372 0.0
374 0.0
388 0.0
415 0.0
426 0.0
438 0.0
466 0.0
480 0.0
481 0.0
496 0.0
514 0.0
517 0.0
521 0.0
533 0.0
547 0.0
564 0.0
585 0.0
587 0.0
592 0.0
608 0.0
613 0.0
618 0.0
624 0.0
630 0.0
632 0.0
640 0.0
660 0.0
663 0.0
680 0.0
683 0.0
685 0.0
709 0.0
715 0.0
716 0.0
719 0.0
739 0.0
744 0.0
746 0.0
747 0.0
753 0.0
759 0.0
784 0.0
790 0.0
806 0.0
813 0.0
832 0.0
837 0.0
852 0.0
857 0.0
859 0.0
869 0.0
871 0.0
880 0.0
895 0.0
903 0.0
916 0.0
918 0.0
921 0.0
940 0.0
945 0.0
947 0.0
960 0.0
967 0.0
993 0.0

 ---------- XGBClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 69, 0: 131}
y_pred {1: 713, 0: 287}
Accuracy: 0.745
Confusion matrix:
[[152 120]
 [135 593]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.53      0.56      0.54       272
         1.0       0.83      0.81      0.82       728

    accuracy                           0.74      1000
   macro avg       0.68      0.69      0.68      1000
weighted avg       0.75      0.74      0.75      1000

ROC AUC: 0.6866919844861021
PR AUC: 0.8124675184562983
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 803, 0: 197}
Accuracy: 0.843
Confusion matrix:
[[156 116]
 [ 41 687]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.79      0.57      0.67       272
         1.0       0.86      0.94      0.90       728

    accuracy                           0.84      1000
   macro avg       0.82      0.76      0.78      1000
weighted avg       0.84      0.84      0.83      1000

ROC AUC: 0.7586053652230123
PR AUC: 0.8483587371532577
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 69, 0: 131}
y_pred {1: 911, 0: 89}
Accuracy: 0.805
Confusion matrix:
[[ 83 189]
 [  6 722]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.93      0.31      0.46       272
         1.0       0.79      0.99      0.88       728

    accuracy                           0.81      1000
   macro avg       0.86      0.65      0.67      1000
weighted avg       0.83      0.81      0.77      1000

ROC AUC: 0.6484526502908856
PR AUC: 0.79200378765033
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 69, 0: 131}
y_pred {0: 715, 1: 285}
Accuracy: 0.439
Confusion matrix:
[[213  59]
 [502 226]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.30      0.78      0.43       272
         1.0       0.79      0.31      0.45       728

    accuracy                           0.44      1000
   macro avg       0.55      0.55      0.44      1000
weighted avg       0.66      0.44      0.44      1000

ROC AUC: 0.5467638978668391
PR AUC: 0.7481731251204935
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 69, 0: 131}
y_pred {1: 926, 0: 74}
Accuracy: 0.792
Confusion matrix:
[[ 69 203]
 [  5 723]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.93      0.25      0.40       272
         1.0       0.78      0.99      0.87       728

    accuracy                           0.79      1000
   macro avg       0.86      0.62      0.64      1000
weighted avg       0.82      0.79      0.74      1000

ROC AUC: 0.6234041693600517
PR AUC: 0.7804150547077113
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 69, 0: 131}
y_pred {0: 1000}
Accuracy: 0.272
Confusion matrix:
[[272   0]
 [728   0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.27      1.00      0.43       272
         1.0       0.00      0.00      0.00       728

    accuracy                           0.27      1000
   macro avg       0.14      0.50      0.21      1000
weighted avg       0.07      0.27      0.12      1000

ROC AUC: 0.5
PR AUC: 0.728
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {0: 896, 1: 104}
Accuracy: 0.314
Confusion matrix:
[[241  31]
 [655  73]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.27      0.89      0.41       272
         1.0       0.70      0.10      0.18       728

    accuracy                           0.31      1000
   macro avg       0.49      0.49      0.29      1000
weighted avg       0.58      0.31      0.24      1000

ROC AUC: 0.4931520685197156
PR AUC: 0.7253851437024514
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 850, 0: 150}
Accuracy: 0.854
Confusion matrix:
[[138 134]
 [ 12 716]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.92      0.51      0.65       272
         1.0       0.84      0.98      0.91       728

    accuracy                           0.85      1000
   macro avg       0.88      0.75      0.78      1000
weighted avg       0.86      0.85      0.84      1000

ROC AUC: 0.7454347123464771
PR AUC: 0.8404680025856497
---------- End MLPClassifier ----------

 ---------- SVC ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 1000}
Accuracy: 0.272
Confusion matrix:
[[272   0]
 [728   0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.27      1.00      0.43       272
         1.0       0.00      0.00      0.00       728

    accuracy                           0.27      1000
   macro avg       0.14      0.50      0.21      1000
weighted avg       0.07      0.27      0.12      1000

ROC AUC: 0.5
PR AUC: 0.728
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
length 10.0 - 100.0
diameter 0.01 - 0.5
young 60.0 - 300.0
density 1000.0 - 2000.0
pressure_time 11.11 - 100.0
pressure_radius 0.56 - 5.0
pressure_amplitude 22.22 - 200.0
strength 0.2 - 10.0
3 0.0
18 0.0
35 0.0
39 0.0
78 0.0
117 0.0
130 0.0
149 0.0
162 0.0
178 0.0
184 0.0
187 0.0
204 0.0
206 0.0
233 0.0
242 0.0
253 0.0
262 0.0
264 0.0
279 0.0
284 0.0
297 0.0
302 0.0
305 0.0
311 0.0
313 0.0
345 0.0
354 0.0
360 0.0
372 0.0
374 0.0
388 0.0
415 0.0
426 0.0
438 0.0
466 0.0
480 0.0
481 0.0
496 0.0
514 0.0
517 0.0
521 0.0
533 0.0
547 0.0
564 0.0
585 0.0
587 0.0
592 0.0
608 0.0
613 0.0
618 0.0
624 0.0
630 0.0
632 0.0
640 0.0
660 0.0
663 0.0
680 0.0
683 0.0
685 0.0
709 0.0
715 0.0
716 0.0
719 0.0
739 0.0
744 0.0
746 0.0
747 0.0
753 0.0
759 0.0
784 0.0
790 0.0
806 0.0
813 0.0
832 0.0
837 0.0
852 0.0
857 0.0
859 0.0
869 0.0
871 0.0
880 0.0
895 0.0
903 0.0
916 0.0
918 0.0
921 0.0
940 0.0
945 0.0
947 0.0
960 0.0
967 0.0
993 0.0

 ---------- XGBClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 98, 0: 202}
y_pred {1: 749, 0: 251}
Accuracy: 0.749
Confusion matrix:
[[136 136]
 [115 613]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.54      0.50      0.52       272
         1.0       0.82      0.84      0.83       728

    accuracy                           0.75      1000
   macro avg       0.68      0.67      0.68      1000
weighted avg       0.74      0.75      0.75      1000

ROC AUC: 0.6710164835164836
PR AUC: 0.8041404656758462
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 847, 0: 153}
Accuracy: 0.819
Confusion matrix:
[[122 150]
 [ 31 697]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.80      0.45      0.57       272
         1.0       0.82      0.96      0.89       728

    accuracy                           0.82      1000
   macro avg       0.81      0.70      0.73      1000
weighted avg       0.82      0.82      0.80      1000

ROC AUC: 0.7029734970911442
PR AUC: 0.8188631109150589
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 98, 0: 202}
y_pred {1: 915, 0: 85}
Accuracy: 0.801
Confusion matrix:
[[ 79 193]
 [  6 722]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.93      0.29      0.44       272
         1.0       0.79      0.99      0.88       728

    accuracy                           0.80      1000
   macro avg       0.86      0.64      0.66      1000
weighted avg       0.83      0.80      0.76      1000

ROC AUC: 0.641099709114415
PR AUC: 0.7885677055185253
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 98, 0: 202}
y_pred {0: 751, 1: 249}
Accuracy: 0.415
Confusion matrix:
[[219  53]
 [532 196]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.29      0.81      0.43       272
         1.0       0.79      0.27      0.40       728

    accuracy                           0.41      1000
   macro avg       0.54      0.54      0.41      1000
weighted avg       0.65      0.41      0.41      1000

ROC AUC: 0.5371889140271493
PR AUC: 0.7439246215631758
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 98, 0: 202}
y_pred {1: 904, 0: 96}
Accuracy: 0.808
Confusion matrix:
[[ 88 184]
 [  8 720]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.92      0.32      0.48       272
         1.0       0.80      0.99      0.88       728

    accuracy                           0.81      1000
   macro avg       0.86      0.66      0.68      1000
weighted avg       0.83      0.81      0.77      1000

ROC AUC: 0.6562702003878474
PR AUC: 0.7957078673538851
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 98, 0: 202}
y_pred {0: 1000}
Accuracy: 0.272
Confusion matrix:
[[272   0]
 [728   0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.27      1.00      0.43       272
         1.0       0.00      0.00      0.00       728

    accuracy                           0.27      1000
   macro avg       0.14      0.50      0.21      1000
weighted avg       0.07      0.27      0.12      1000

ROC AUC: 0.5
PR AUC: 0.728
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 210, 0: 790}
Accuracy: 0.402
Confusion matrix:
[[232  40]
 [558 170]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.29      0.85      0.44       272
         1.0       0.81      0.23      0.36       728

    accuracy                           0.40      1000
   macro avg       0.55      0.54      0.40      1000
weighted avg       0.67      0.40      0.38      1000

ROC AUC: 0.5432288299935359
PR AUC: 0.7470371533228676
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 878, 0: 122}
Accuracy: 0.834
Confusion matrix:
[[114 158]
 [  8 720]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.93      0.42      0.58       272
         1.0       0.82      0.99      0.90       728

    accuracy                           0.83      1000
   macro avg       0.88      0.70      0.74      1000
weighted avg       0.85      0.83      0.81      1000

ROC AUC: 0.7040643180349062
PR AUC: 0.8190340684372575
---------- End MLPClassifier ----------

 ---------- SVC ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 1000}
Accuracy: 0.272
Confusion matrix:
[[272   0]
 [728   0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.27      1.00      0.43       272
         1.0       0.00      0.00      0.00       728

    accuracy                           0.27      1000
   macro avg       0.14      0.50      0.21      1000
weighted avg       0.07      0.27      0.12      1000

ROC AUC: 0.5
PR AUC: 0.728
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
length 10.0 - 100.0
diameter 0.01 - 0.5
young 60.0 - 300.0
density 1000.0 - 2000.0
pressure_time 11.11 - 100.0
pressure_radius 0.56 - 5.0
pressure_amplitude 22.22 - 200.0
strength 0.2 - 10.0
3 0.0
18 0.0
35 0.0
39 0.0
78 0.0
117 0.0
130 0.0
149 0.0
162 0.0
178 0.0
184 0.0
187 0.0
204 0.0
206 0.0
233 0.0
242 0.0
253 0.0
262 0.0
264 0.0
279 0.0
284 0.0
297 0.0
302 0.0
305 0.0
311 0.0
313 0.0
345 0.0
354 0.0
360 0.0
372 0.0
374 0.0
388 0.0
415 0.0
426 0.0
438 0.0
466 0.0
480 0.0
481 0.0
496 0.0
514 0.0
517 0.0
521 0.0
533 0.0
547 0.0
564 0.0
585 0.0
587 0.0
592 0.0
608 0.0
613 0.0
618 0.0
624 0.0
630 0.0
632 0.0
640 0.0
660 0.0
663 0.0
680 0.0
683 0.0
685 0.0
709 0.0
715 0.0
716 0.0
719 0.0
739 0.0
744 0.0
746 0.0
747 0.0
753 0.0
759 0.0
784 0.0
790 0.0
806 0.0
813 0.0
832 0.0
837 0.0
852 0.0
857 0.0
859 0.0
869 0.0
871 0.0
880 0.0
895 0.0
903 0.0
916 0.0
918 0.0
921 0.0
940 0.0
945 0.0
947 0.0
960 0.0
967 0.0
993 0.0

 ---------- XGBClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 132, 0: 268}
y_pred {1: 661, 0: 339}
Accuracy: 0.737
Confusion matrix:
[[174  98]
 [165 563]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.51      0.64      0.57       272
         1.0       0.85      0.77      0.81       728

    accuracy                           0.74      1000
   macro avg       0.68      0.71      0.69      1000
weighted avg       0.76      0.74      0.75      1000

ROC AUC: 0.7065287653522947
PR AUC: 0.8236943691709198
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 854, 0: 146}
Accuracy: 0.816
Confusion matrix:
[[117 155]
 [ 29 699]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.80      0.43      0.56       272
         1.0       0.82      0.96      0.88       728

    accuracy                           0.82      1000
   macro avg       0.81      0.70      0.72      1000
weighted avg       0.81      0.82      0.80      1000

ROC AUC: 0.6951559469941824
PR AUC: 0.8148960418972129
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 132, 0: 268}
y_pred {1: 910, 0: 90}
Accuracy: 0.802
Confusion matrix:
[[ 82 190]
 [  8 720]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.91      0.30      0.45       272
         1.0       0.79      0.99      0.88       728

    accuracy                           0.80      1000
   macro avg       0.85      0.65      0.67      1000
weighted avg       0.82      0.80      0.76      1000

ROC AUC: 0.6452407886231416
PR AUC: 0.7905141891075957
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 132, 0: 268}
y_pred {0: 848, 1: 152}
Accuracy: 0.37
Confusion matrix:
[[245  27]
 [603 125]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.29      0.90      0.44       272
         1.0       0.82      0.17      0.28       728

    accuracy                           0.37      1000
   macro avg       0.56      0.54      0.36      1000
weighted avg       0.68      0.37      0.33      1000

ROC AUC: 0.536219295410472
PR AUC: 0.7442033689994216
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 132, 0: 268}
y_pred {1: 941, 0: 59}
Accuracy: 0.775
Confusion matrix:
[[ 53 219]
 [  6 722]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.90      0.19      0.32       272
         1.0       0.77      0.99      0.87       728

    accuracy                           0.78      1000
   macro avg       0.83      0.59      0.59      1000
weighted avg       0.80      0.78      0.72      1000

ROC AUC: 0.5933055914673562
PR AUC: 0.7669452184372483
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 132, 0: 268}
y_pred {0: 1000}
Accuracy: 0.272
Confusion matrix:
[[272   0]
 [728   0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.27      1.00      0.43       272
         1.0       0.00      0.00      0.00       728

    accuracy                           0.27      1000
   macro avg       0.14      0.50      0.21      1000
weighted avg       0.07      0.27      0.12      1000

ROC AUC: 0.5
PR AUC: 0.728
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {0: 864, 1: 136}
Accuracy: 0.386
Confusion matrix:
[[261  11]
 [603 125]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.30      0.96      0.46       272
         1.0       0.92      0.17      0.29       728

    accuracy                           0.39      1000
   macro avg       0.61      0.57      0.37      1000
weighted avg       0.75      0.39      0.34      1000

ROC AUC: 0.5656310601163542
PR AUC: 0.7608155300581771
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 883, 0: 117}
Accuracy: 0.827
Confusion matrix:
[[108 164]
 [  9 719]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.92      0.40      0.56       272
         1.0       0.81      0.99      0.89       728

    accuracy                           0.83      1000
   macro avg       0.87      0.69      0.72      1000
weighted avg       0.84      0.83      0.80      1000

ROC AUC: 0.6923480930833872
PR AUC: 0.8132030166888604
---------- End MLPClassifier ----------

 ---------- SVC ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 1000}
Accuracy: 0.272
Confusion matrix:
[[272   0]
 [728   0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.27      1.00      0.43       272
         1.0       0.00      0.00      0.00       728

    accuracy                           0.27      1000
   macro avg       0.14      0.50      0.21      1000
weighted avg       0.07      0.27      0.12      1000

ROC AUC: 0.5
PR AUC: 0.728
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
length 10.0 - 100.0
diameter 0.01 - 0.5
young 60.0 - 300.0
density 1000.0 - 2000.0
pressure_time 11.11 - 100.0
pressure_radius 0.56 - 5.0
pressure_amplitude 22.22 - 200.0
strength 0.2 - 10.0
3 0.0
18 0.0
35 0.0
39 0.0
78 0.0
117 0.0
130 0.0
149 0.0
162 0.0
178 0.0
184 0.0
187 0.0
204 0.0
206 0.0
233 0.0
242 0.0
253 0.0
262 0.0
264 0.0
279 0.0
284 0.0
297 0.0
302 0.0
305 0.0
311 0.0
313 0.0
345 0.0
354 0.0
360 0.0
372 0.0
374 0.0
388 0.0
415 0.0
426 0.0
438 0.0
466 0.0
480 0.0
481 0.0
496 0.0
514 0.0
517 0.0
521 0.0
533 0.0
547 0.0
564 0.0
585 0.0
587 0.0
592 0.0
608 0.0
613 0.0
618 0.0
624 0.0
630 0.0
632 0.0
640 0.0
660 0.0
663 0.0
680 0.0
683 0.0
685 0.0
709 0.0
715 0.0
716 0.0
719 0.0
739 0.0
744 0.0
746 0.0
747 0.0
753 0.0
759 0.0
784 0.0
790 0.0
806 0.0
813 0.0
832 0.0
837 0.0
852 0.0
857 0.0
859 0.0
869 0.0
871 0.0
880 0.0
895 0.0
903 0.0
916 0.0
918 0.0
921 0.0
940 0.0
945 0.0
947 0.0
960 0.0
967 0.0
993 0.0

 ---------- XGBClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 167, 0: 333}
y_pred {1: 681, 0: 319}
Accuracy: 0.759
Confusion matrix:
[[175  97]
 [144 584]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.55      0.64      0.59       272
         1.0       0.86      0.80      0.83       728

    accuracy                           0.76      1000
   macro avg       0.70      0.72      0.71      1000
weighted avg       0.77      0.76      0.76      1000

ROC AUC: 0.7227900775694893
PR AUC: 0.8319346791241065
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 902, 0: 98}
Accuracy: 0.804
Confusion matrix:
[[ 87 185]
 [ 11 717]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.89      0.32      0.47       272
         1.0       0.79      0.98      0.88       728

    accuracy                           0.80      1000
   macro avg       0.84      0.65      0.68      1000
weighted avg       0.82      0.80      0.77      1000

ROC AUC: 0.6523715255332903
PR AUC: 0.7938893667308302
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 167, 0: 333}
y_pred {1: 926, 0: 74}
Accuracy: 0.79
Confusion matrix:
[[ 68 204]
 [  6 722]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.92      0.25      0.39       272
         1.0       0.78      0.99      0.87       728

    accuracy                           0.79      1000
   macro avg       0.85      0.62      0.63      1000
weighted avg       0.82      0.79      0.74      1000

ROC AUC: 0.6208791208791209
PR AUC: 0.7792715448698171
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 167, 0: 333}
y_pred {0: 831, 1: 169}
Accuracy: 0.393
Confusion matrix:
[[248  24]
 [583 145]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.30      0.91      0.45       272
         1.0       0.86      0.20      0.32       728

    accuracy                           0.39      1000
   macro avg       0.58      0.56      0.39      1000
weighted avg       0.71      0.39      0.36      1000

ROC AUC: 0.5554702650290885
PR AUC: 0.7538905000325118
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 167, 0: 333}
y_pred {1: 921, 0: 79}
Accuracy: 0.793
Confusion matrix:
[[ 72 200]
 [  7 721]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.91      0.26      0.41       272
         1.0       0.78      0.99      0.87       728

    accuracy                           0.79      1000
   macro avg       0.85      0.63      0.64      1000
weighted avg       0.82      0.79      0.75      1000

ROC AUC: 0.6275452488687783
PR AUC: 0.7823173807734068
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 167, 0: 333}
y_pred {0: 1000}
Accuracy: 0.272
Confusion matrix:
[[272   0]
 [728   0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.27      1.00      0.43       272
         1.0       0.00      0.00      0.00       728

    accuracy                           0.27      1000
   macro avg       0.14      0.50      0.21      1000
weighted avg       0.07      0.27      0.12      1000

ROC AUC: 0.5
PR AUC: 0.728
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 277, 0: 723}
Accuracy: 0.503
Confusion matrix:
[[249  23]
 [474 254]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.34      0.92      0.50       272
         1.0       0.92      0.35      0.51       728

    accuracy                           0.50      1000
   macro avg       0.63      0.63      0.50      1000
weighted avg       0.76      0.50      0.50      1000

ROC AUC: 0.6321711376858437
PR AUC: 0.7939309715555203
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 934, 0: 66}
Accuracy: 0.786
Confusion matrix:
[[ 62 210]
 [  4 724]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

         0.0       0.94      0.23      0.37       272
         1.0       0.78      0.99      0.87       728

    accuracy                           0.79      1000
   macro avg       0.86      0.61      0.62      1000
weighted avg       0.82      0.79      0.73      1000

ROC AUC: 0.6112233354880414
PR AUC: 0.7749014753982634
---------- End MLPClassifier ----------

 ---------- SVC ----------
(1000, 8) (1000,)
y_test {1.0: 728, 0.0: 272} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 1000}
Accuracy: 0.272
Confusion matrix:
[[272   0]
 [728   0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

         0.0       0.27      1.00      0.43       272
         1.0       0.00      0.00      0.00       728

    accuracy                           0.27      1000
   macro avg       0.14      0.50      0.21      1000
weighted avg       0.07      0.27      0.12      1000

ROC AUC: 0.5
PR AUC: 0.728
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End SVC ----------
roc_metrics
[[0.6748, 0.7903, 0.6613, 0.5291, 0.6009, 0.5, 0.5155, 0.8129, 0.5], [0.6867, 0.7586, 0.6485, 0.5468, 0.6234, 0.5, 0.4932, 0.7454, 0.5], [0.671, 0.703, 0.6411, 0.5372, 0.6563, 0.5, 0.5432, 0.7041, 0.5], [0.7065, 0.6952, 0.6452, 0.5362, 0.5933, 0.5, 0.5656, 0.6923, 0.5], [0.7228, 0.6524, 0.6209, 0.5555, 0.6275, 0.5, 0.6322, 0.6112, 0.5]]
pr_metrics
[[0.8052, 0.8692, 0.7981, 0.7403, 0.7703, 0.728, 0.7348, 0.8806, 0.728], [0.8125, 0.8484, 0.792, 0.7482, 0.7804, 0.728, 0.7254, 0.8405, 0.728], [0.8041, 0.8189, 0.7886, 0.7439, 0.7957, 0.728, 0.747, 0.819, 0.728], [0.8237, 0.8149, 0.7905, 0.7442, 0.7669, 0.728, 0.7608, 0.8132, 0.728], [0.8319, 0.7939, 0.7793, 0.7539, 0.7823, 0.728, 0.7939, 0.7749, 0.728]]
f1_metrics
[[0.5225, 0.685, 0.4891, 0.4216, 0.3384, 0.4277, 0.4307, 0.73, 0.4277], [0.5438, 0.6652, 0.4598, 0.4316, 0.3988, 0.4277, 0.4127, 0.654, 0.4277], [0.5201, 0.5741, 0.4426, 0.4282, 0.4783, 0.4277, 0.4369, 0.5787, 0.4277], [0.5696, 0.5598, 0.453, 0.4375, 0.3202, 0.4277, 0.4595, 0.5553, 0.4277], [0.5922, 0.4703, 0.3931, 0.4497, 0.4103, 0.4277, 0.5005, 0.3669, 0.4277]]

roc
['100', 0.6748, 0.7903, 0.6613, 0.5291, 0.6009, 0.5, 0.5155, 0.8129, 0.5],
['200', 0.6867, 0.7586, 0.6485, 0.5468, 0.6234, 0.5, 0.4932, 0.7454, 0.5],
['300', 0.671, 0.703, 0.6411, 0.5372, 0.6563, 0.5, 0.5432, 0.7041, 0.5],
['400', 0.7065, 0.6952, 0.6452, 0.5362, 0.5933, 0.5, 0.5656, 0.6923, 0.5],
['500', 0.7228, 0.6524, 0.6209, 0.5555, 0.6275, 0.5, 0.6322, 0.6112, 0.5],

None
pr
['100', 0.8052, 0.8692, 0.7981, 0.7403, 0.7703, 0.728, 0.7348, 0.8806, 0.728],
['200', 0.8125, 0.8484, 0.792, 0.7482, 0.7804, 0.728, 0.7254, 0.8405, 0.728],
['300', 0.8041, 0.8189, 0.7886, 0.7439, 0.7957, 0.728, 0.747, 0.819, 0.728],
['400', 0.8237, 0.8149, 0.7905, 0.7442, 0.7669, 0.728, 0.7608, 0.8132, 0.728],
['500', 0.8319, 0.7939, 0.7793, 0.7539, 0.7823, 0.728, 0.7939, 0.7749, 0.728],

None
f1
['100', 0.5225, 0.685, 0.4891, 0.4216, 0.3384, 0.4277, 0.4307, 0.73, 0.4277],
['200', 0.5438, 0.6652, 0.4598, 0.4316, 0.3988, 0.4277, 0.4127, 0.654, 0.4277],
['300', 0.5201, 0.5741, 0.4426, 0.4282, 0.4783, 0.4277, 0.4369, 0.5787, 0.4277],
['400', 0.5696, 0.5598, 0.453, 0.4375, 0.3202, 0.4277, 0.4595, 0.5553, 0.4277],
['500', 0.5922, 0.4703, 0.3931, 0.4497, 0.4103, 0.4277, 0.5005, 0.3669, 0.4277],


'''






