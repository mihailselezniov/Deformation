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
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 33, 0: 67}
y_pred {1: 20872008, 0: 9361080}
Accuracy: 0.7713856751913665
Confusion matrix:
[[ 2612335   162972]
 [ 6748745 20709036]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.28      0.94      0.43   2775307
           1       0.99      0.75      0.86  27457781

    accuracy                           0.77  30233088
   macro avg       0.64      0.85      0.64  30233088
weighted avg       0.93      0.77      0.82  30233088

ROC AUC: 0.8477457993709515
PR AUC: 0.9715485354609068
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 19350366, 0: 10882722}
Accuracy: 0.7314627933474741
Confusion matrix:
[[ 2769660     5647]
 [ 8113062 19344719]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.25      1.00      0.41   2775307
           1       1.00      0.70      0.83  27457781

    accuracy                           0.73  30233088
   macro avg       0.63      0.85      0.62  30233088
weighted avg       0.93      0.73      0.79  30233088

ROC AUC: 0.8512456056325731
PR AUC: 0.9726707644829715
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 33, 0: 67}
y_pred {1: 25753857, 0: 4479231}
Accuracy: 0.9205157607453132
Confusion matrix:
[[ 2425742   349565]
 [ 2053489 25404292]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.54      0.87      0.67   2775307
           1       0.99      0.93      0.95  27457781

    accuracy                           0.92  30233088
   macro avg       0.76      0.90      0.81  30233088
weighted avg       0.95      0.92      0.93  30233088

ROC AUC: 0.8996287109726478
PR AUC: 0.980576566191755
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 33, 0: 67}
y_pred {0: 22963529, 1: 7269559}
Accuracy: 0.32497183218598114
Confusion matrix:
[[ 2665325   109982]
 [20298204  7159577]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.12      0.96      0.21   2775307
           1       0.98      0.26      0.41  27457781

    accuracy                           0.32  30233088
   macro avg       0.55      0.61      0.31  30233088
weighted avg       0.91      0.32      0.39  30233088

ROC AUC: 0.6105598981135527
PR AUC: 0.9281940354007876
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 33, 0: 67}
y_pred {1: 26138479, 0: 4094609}
Accuracy: 0.9268699909185592
Confusion matrix:
[[ 2329485   445822]
 [ 1765124 25692657]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.57      0.84      0.68   2775307
           1       0.98      0.94      0.96  27457781

    accuracy                           0.93  30233088
   macro avg       0.78      0.89      0.82  30233088
weighted avg       0.94      0.93      0.93  30233088

ROC AUC: 0.8875380875113489
PR AUC: 0.9781391260013497
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 33, 0: 67}
y_pred {0: 30233088}
Accuracy: 0.0917970073053735
Confusion matrix:
[[ 2775307        0]
 [27457781        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.09      1.00      0.17   2775307
           1       0.00      0.00      0.00  27457781

    accuracy                           0.09  30233088
   macro avg       0.05      0.50      0.08  30233088
weighted avg       0.01      0.09      0.02  30233088

ROC AUC: 0.5
PR AUC: 0.9082029926946265
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 6624964, 0: 23608124}
Accuracy: 0.30831124495122697
Confusion matrix:
[[ 2735772    39535]
 [20872352  6585429]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.12      0.99      0.21   2775307
           1       0.99      0.24      0.39  27457781

    accuracy                           0.31  30233088
   macro avg       0.55      0.61      0.30  30233088
weighted avg       0.91      0.31      0.37  30233088

ROC AUC: 0.6127965415667408
PR AUC: 0.928788181362327
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 22218194, 0: 8014894}
Accuracy: 0.8235247090869448
Confusion matrix:
[[ 2727404    47903]
 [ 5287490 22170291]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.34      0.98      0.51   2775307
           1       1.00      0.81      0.89  27457781

    accuracy                           0.82  30233088
   macro avg       0.67      0.90      0.70  30233088
weighted avg       0.94      0.82      0.86  30233088

ROC AUC: 0.8950857830781003
PR AUC: 0.9805819895623722
---------- End MLPClassifier ----------

 ---------- SVC ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 30233088}
Accuracy: 0.0917970073053735
Confusion matrix:
[[ 2775307        0]
 [27457781        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.09      1.00      0.17   2775307
           1       0.00      0.00      0.00  27457781

    accuracy                           0.09  30233088
   macro avg       0.05      0.50      0.08  30233088
weighted avg       0.01      0.09      0.02  30233088

ROC AUC: 0.5
PR AUC: 0.9082029926946265
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
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 69, 0: 131}
y_pred {1: 23684942, 0: 6548146}
Accuracy: 0.8407594685663602
Confusion matrix:
[[ 2254560   520747]
 [ 4293586 23164195]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.34      0.81      0.48   2775307
           1       0.98      0.84      0.91  27457781

    accuracy                           0.84  30233088
   macro avg       0.66      0.83      0.69  30233088
weighted avg       0.92      0.84      0.87  30233088

ROC AUC: 0.8279968580336526
PR AUC: 0.9670972688956141
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 23415015, 0: 6818073}
Accuracy: 0.859238593159918
Confusion matrix:
[[ 2668864   106443]
 [ 4149209 23308572]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.39      0.96      0.56   2775307
           1       1.00      0.85      0.92  27457781

    accuracy                           0.86  30233088
   macro avg       0.69      0.91      0.74  30233088
weighted avg       0.94      0.86      0.88  30233088

ROC AUC: 0.9052670424212677
PR AUC: 0.9822693579446738
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 69, 0: 131}
y_pred {1: 26365483, 0: 3867605}
Accuracy: 0.9373318398702772
Confusion matrix:
[[ 2374130   401177]
 [ 1493475 25964306]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.61      0.86      0.71   2775307
           1       0.98      0.95      0.96  27457781

    accuracy                           0.94  30233088
   macro avg       0.80      0.90      0.84  30233088
weighted avg       0.95      0.94      0.94  30233088

ROC AUC: 0.9005280088555725
PR AUC: 0.980618636694123
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 69, 0: 131}
y_pred {0: 21187644, 1: 9045444}
Accuracy: 0.3782768402619011
Confusion matrix:
[[ 2583170   192137]
 [18604474  8853307]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.12      0.93      0.22   2775307
           1       0.98      0.32      0.49  27457781

    accuracy                           0.38  30233088
   macro avg       0.55      0.63      0.35  30233088
weighted avg       0.90      0.38      0.46  30233088

ROC AUC: 0.6266012742086223
PR AUC: 0.9309525123019805
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 69, 0: 131}
y_pred {1: 25235667, 0: 4997421}
Accuracy: 0.904825798806923
Confusion matrix:
[[ 2447659   327648]
 [ 2549762 24908019]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.49      0.88      0.63   2775307
           1       0.99      0.91      0.95  27457781

    accuracy                           0.90  30233088
   macro avg       0.74      0.89      0.79  30233088
weighted avg       0.94      0.90      0.92  30233088

ROC AUC: 0.8945402637332561
PR AUC: 0.9796977587585388
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 69, 0: 131}
y_pred {0: 30233088}
Accuracy: 0.0917970073053735
Confusion matrix:
[[ 2775307        0]
 [27457781        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.09      1.00      0.17   2775307
           1       0.00      0.00      0.00  27457781

    accuracy                           0.09  30233088
   macro avg       0.05      0.50      0.08  30233088
weighted avg       0.01      0.09      0.02  30233088

ROC AUC: 0.5
PR AUC: 0.9082029926946265
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 11363026, 0: 18870062}
Accuracy: 0.4573695879163915
Confusion matrix:
[[ 2619988   155319]
 [16250074 11207707]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.14      0.94      0.24   2775307
           1       0.99      0.41      0.58  27457781

    accuracy                           0.46  30233088
   macro avg       0.56      0.68      0.41  30233088
weighted avg       0.91      0.46      0.55  30233088

ROC AUC: 0.676107507312788
PR AUC: 0.9400933332382555
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 24908587, 0: 5324501}
Accuracy: 0.906133571271317
Confusion matrix:
[[ 2630968   144339]
 [ 2693533 24764248]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.49      0.95      0.65   2775307
           1       0.99      0.90      0.95  27457781

    accuracy                           0.91  30233088
   macro avg       0.74      0.92      0.80  30233088
weighted avg       0.95      0.91      0.92  30233088

ROC AUC: 0.9249472213177019
PR AUC: 0.9857686657105307
---------- End MLPClassifier ----------

 ---------- SVC ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 30233088}
Accuracy: 0.0917970073053735
Confusion matrix:
[[ 2775307        0]
 [27457781        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.09      1.00      0.17   2775307
           1       0.00      0.00      0.00  27457781

    accuracy                           0.09  30233088
   macro avg       0.05      0.50      0.08  30233088
weighted avg       0.01      0.09      0.02  30233088

ROC AUC: 0.5
PR AUC: 0.9082029926946265
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
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 98, 0: 202}
y_pred {1: 24224987, 0: 6008101}
Accuracy: 0.8603185357711393
Confusion matrix:
[[ 2280203   495104]
 [ 3727898 23729883]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.38      0.82      0.52   2775307
           1       0.98      0.86      0.92  27457781

    accuracy                           0.86  30233088
   macro avg       0.68      0.84      0.72  30233088
weighted avg       0.92      0.86      0.88  30233088

ROC AUC: 0.842917758208755
PR AUC: 0.969873928529596
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 23574421, 0: 6658667}
Accuracy: 0.86538477313333
Confusion matrix:
[[ 2682070    93237]
 [ 3976597 23481184]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.40      0.97      0.57   2775307
           1       1.00      0.86      0.92  27457781

    accuracy                           0.87  30233088
   macro avg       0.70      0.91      0.74  30233088
weighted avg       0.94      0.87      0.89  30233088

ROC AUC: 0.910789464262977
PR AUC: 0.9833231989550733
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 98, 0: 202}
y_pred {1: 26123197, 0: 4109891}
Accuracy: 0.9355782644498637
Confusion matrix:
[[ 2468765   306542]
 [ 1641126 25816655]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.60      0.89      0.72   2775307
           1       0.99      0.94      0.96  27457781

    accuracy                           0.94  30233088
   macro avg       0.79      0.91      0.84  30233088
weighted avg       0.95      0.94      0.94  30233088

ROC AUC: 0.9148887826006498
PR AUC: 0.9834802620794294
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 98, 0: 202}
y_pred {0: 20824773, 1: 9408315}
Accuracy: 0.38924856104675776
Confusion matrix:
[[ 2567589   207718]
 [18257184  9200597]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.12      0.93      0.22   2775307
           1       0.98      0.34      0.50  27457781

    accuracy                           0.39  30233088
   macro avg       0.55      0.63      0.36  30233088
weighted avg       0.90      0.39      0.47  30233088

ROC AUC: 0.6301182699524746
PR AUC: 0.931564505228858
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 98, 0: 202}
y_pred {1: 25569924, 0: 4663164}
Accuracy: 0.9221416945566394
Confusion matrix:
[[ 2542287   233020]
 [ 2120877 25336904]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.55      0.92      0.68   2775307
           1       0.99      0.92      0.96  27457781

    accuracy                           0.92  30233088
   macro avg       0.77      0.92      0.82  30233088
weighted avg       0.95      0.92      0.93  30233088

ROC AUC: 0.9193983669018354
PR AUC: 0.9845003265735672
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 98, 0: 202}
y_pred {0: 30233088}
Accuracy: 0.0917970073053735
Confusion matrix:
[[ 2775307        0]
 [27457781        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.09      1.00      0.17   2775307
           1       0.00      0.00      0.00  27457781

    accuracy                           0.09  30233088
   macro avg       0.05      0.50      0.08  30233088
weighted avg       0.01      0.09      0.02  30233088

ROC AUC: 0.5
PR AUC: 0.9082029926946265
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 19269831, 0: 10963257}
Accuracy: 0.7119699449821335
Confusion matrix:
[[ 2515263   260044]
 [ 8447994 19009787]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.23      0.91      0.37   2775307
           1       0.99      0.69      0.81  27457781

    accuracy                           0.71  30233088
   macro avg       0.61      0.80      0.59  30233088
weighted avg       0.92      0.71      0.77  30233088

ROC AUC: 0.7993143425940217
PR AUC: 0.9624137396562463
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 25514232, 0: 4718856}
Accuracy: 0.9237808258289726
Confusion matrix:
[[ 2594911   180396]
 [ 2123945 25333836]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.55      0.93      0.69   2775307
           1       0.99      0.92      0.96  27457781

    accuracy                           0.92  30233088
   macro avg       0.77      0.93      0.82  30233088
weighted avg       0.95      0.92      0.93  30233088

ROC AUC: 0.9288232522538259
PR AUC: 0.9863757263716217
---------- End MLPClassifier ----------

 ---------- SVC ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 30233088}
Accuracy: 0.0917970073053735
Confusion matrix:
[[ 2775307        0]
 [27457781        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.09      1.00      0.17   2775307
           1       0.00      0.00      0.00  27457781

    accuracy                           0.09  30233088
   macro avg       0.05      0.50      0.08  30233088
weighted avg       0.01      0.09      0.02  30233088

ROC AUC: 0.5
PR AUC: 0.9082029926946265
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
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 132, 0: 268}
y_pred {1: 24615480, 0: 5617608}
Accuracy: 0.8673912171988518
Confusion matrix:
[[ 2191871   583436]
 [ 3425737 24032044]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.39      0.79      0.52   2775307
           1       0.98      0.88      0.92  27457781

    accuracy                           0.87  30233088
   macro avg       0.68      0.83      0.72  30233088
weighted avg       0.92      0.87      0.89  30233088

ROC AUC: 0.8325061267691527
PR AUC: 0.9678022121715352
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 24132505, 0: 6100583}
Accuracy: 0.883478525250216
Confusion matrix:
[[ 2676543    98764]
 [ 3424040 24033741]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.44      0.96      0.60   2775307
           1       1.00      0.88      0.93  27457781

    accuracy                           0.88  30233088
   macro avg       0.72      0.92      0.77  30233088
weighted avg       0.94      0.88      0.90  30233088

ROC AUC: 0.9198556569298495
PR AUC: 0.9849705106843407
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 132, 0: 268}
y_pred {1: 26072928, 0: 4160160}
Accuracy: 0.9360626674986029
Confusion matrix:
[[ 2501222   274085]
 [ 1658938 25798843]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.60      0.90      0.72   2775307
           1       0.99      0.94      0.96  27457781

    accuracy                           0.94  30233088
   macro avg       0.80      0.92      0.84  30233088
weighted avg       0.95      0.94      0.94  30233088

ROC AUC: 0.9204118912972257
PR AUC: 0.9845767108744891
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 132, 0: 268}
y_pred {0: 20316831, 1: 9916257}
Accuracy: 0.40662270423715896
Confusion matrix:
[[ 2576255   199052]
 [17740576  9717205]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.13      0.93      0.22   2775307
           1       0.98      0.35      0.52  27457781

    accuracy                           0.41  30233088
   macro avg       0.55      0.64      0.37  30233088
weighted avg       0.90      0.41      0.49  30233088

ROC AUC: 0.6410868538493939
PR AUC: 0.9335857418997253
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 132, 0: 268}
y_pred {1: 28081109, 0: 2151979}
Accuracy: 0.9563816306160985
Confusion matrix:
[[ 1804284   971023]
 [  347695 27110086]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.84      0.65      0.73   2775307
           1       0.97      0.99      0.98  27457781

    accuracy                           0.96  30233088
   macro avg       0.90      0.82      0.85  30233088
weighted avg       0.95      0.96      0.95  30233088

ROC AUC: 0.8187288070540304
PR AUC: 0.9646962317462499
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 132, 0: 268}
y_pred {0: 30233088}
Accuracy: 0.0917970073053735
Confusion matrix:
[[ 2775307        0]
 [27457781        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.09      1.00      0.17   2775307
           1       0.00      0.00      0.00  27457781

    accuracy                           0.09  30233088
   macro avg       0.05      0.50      0.08  30233088
weighted avg       0.01      0.09      0.02  30233088

ROC AUC: 0.5
PR AUC: 0.9082029926946265
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 12305957, 0: 17927131}
Accuracy: 0.4877263612635269
Confusion matrix:
[[ 2607412   167895]
 [15319719 12138062]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.15      0.94      0.25   2775307
           1       0.99      0.44      0.61  27457781

    accuracy                           0.49  30233088
   macro avg       0.57      0.69      0.43  30233088
weighted avg       0.91      0.49      0.58  30233088

ROC AUC: 0.690783366767588
PR AUC: 0.9427517941236724
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 25221782, 0: 5011306}
Accuracy: 0.9160637841559552
Confusion matrix:
[[ 2624481   150826]
 [ 2386825 25070956]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.52      0.95      0.67   2775307
           1       0.99      0.91      0.95  27457781

    accuracy                           0.92  30233088
   macro avg       0.76      0.93      0.81  30233088
weighted avg       0.95      0.92      0.93  30233088

ROC AUC: 0.9293636052970538
PR AUC: 0.9865601865637038
---------- End MLPClassifier ----------

 ---------- SVC ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 30233051, 1: 37}
Accuracy: 0.09179823113007841
Confusion matrix:
[[ 2775307        0]
 [27457744       37]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.09      1.00      0.17   2775307
           1       1.00      0.00      0.00  27457781

    accuracy                           0.09  30233088
   macro avg       0.55      0.50      0.08  30233088
weighted avg       0.92      0.09      0.02  30233088

ROC AUC: 0.5000006737616561
PR AUC: 0.9082031163932338
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
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 167, 0: 333}
y_pred {1: 24053115, 0: 6179973}
Accuracy: 0.8682784901099088
Confusion matrix:
[[ 2486466   288841]
 [ 3693507 23764274]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.40      0.90      0.56   2775307
           1       0.99      0.87      0.92  27457781

    accuracy                           0.87  30233088
   macro avg       0.70      0.88      0.74  30233088
weighted avg       0.93      0.87      0.89  30233088

ROC AUC: 0.8807044039899787
PR AUC: 0.9772587103124715
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 25090240, 0: 5142848}
Accuracy: 0.910800808703365
Confusion matrix:
[[ 2610694   164613]
 [ 2532154 24925627]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.51      0.94      0.66   2775307
           1       0.99      0.91      0.95  27457781

    accuracy                           0.91  30233088
   macro avg       0.75      0.92      0.80  30233088
weighted avg       0.95      0.91      0.92  30233088

ROC AUC: 0.9242333276281159
PR AUC: 0.9855786904261938
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 167, 0: 333}
y_pred {1: 26485239, 0: 3747849}
Accuracy: 0.9444534412098426
Confusion matrix:
[[ 2421906   353401]
 [ 1325943 26131838]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.65      0.87      0.74   2775307
           1       0.99      0.95      0.97  27457781

    accuracy                           0.94  30233088
   macro avg       0.82      0.91      0.86  30233088
weighted avg       0.96      0.94      0.95  30233088

ROC AUC: 0.9121860648486685
PR AUC: 0.9828681328297034
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 167, 0: 333}
y_pred {0: 19780691, 1: 10452397}
Accuracy: 0.4255173007798608
Confusion matrix:
[[ 2593806   181501]
 [17186885 10270896]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.13      0.93      0.23   2775307
           1       0.98      0.37      0.54  27457781

    accuracy                           0.43  30233088
   macro avg       0.56      0.65      0.39  30233088
weighted avg       0.90      0.43      0.51  30233088

ROC AUC: 0.6543314346255341
PR AUC: 0.9360453079704546
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 167, 0: 333}
y_pred {1: 26674770, 0: 3558318}
Accuracy: 0.9476769624062219
Confusion matrix:
[[ 2375869   399438]
 [ 1182449 26275332]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.67      0.86      0.75   2775307
           1       0.99      0.96      0.97  27457781

    accuracy                           0.95  30233088
   macro avg       0.83      0.91      0.86  30233088
weighted avg       0.96      0.95      0.95  30233088

ROC AUC: 0.906505020848937
PR AUC: 0.9817173155680339
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 167, 0: 333}
y_pred {0: 30233088}
Accuracy: 0.0917970073053735
Confusion matrix:
[[ 2775307        0]
 [27457781        0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.09      1.00      0.17   2775307
           1       0.00      0.00      0.00  27457781

    accuracy                           0.09  30233088
   macro avg       0.05      0.50      0.08  30233088
weighted avg       0.01      0.09      0.02  30233088

ROC AUC: 0.5
PR AUC: 0.9082029926946265
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 18905723, 0: 11327365}
Accuracy: 0.7023971219876712
Confusion matrix:
[[ 2552609   222698]
 [ 8774756 18683025]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.23      0.92      0.36   2775307
           1       0.99      0.68      0.81  27457781

    accuracy                           0.70  30233088
   macro avg       0.61      0.80      0.58  30233088
weighted avg       0.92      0.70      0.77  30233088

ROC AUC: 0.8000923449190362
PR AUC: 0.9626491633581195
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 26322057, 0: 3911031}
Accuracy: 0.9467785758437908
Confusion matrix:
[[ 2538645   236662]
 [ 1372386 26085395]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.65      0.91      0.76   2775307
           1       0.99      0.95      0.97  27457781

    accuracy                           0.95  30233088
   macro avg       0.82      0.93      0.86  30233088
weighted avg       0.96      0.95      0.95  30233088

ROC AUC: 0.9323720755928387
PR AUC: 0.9868702027975371
---------- End MLPClassifier ----------

 ---------- SVC ----------
(30233088, 8) (30233088,)
y_test {1: 27457781, 0: 2775307} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 30202364, 1: 30724}
Accuracy: 0.09281324487925283
Confusion matrix:
[[ 2775307        0]
 [27427057    30724]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.09      1.00      0.17   2775307
           1       1.00      0.00      0.00  27457781

    accuracy                           0.09  30233088
   macro avg       0.55      0.50      0.09  30233088
weighted avg       0.92      0.09      0.02  30233088

ROC AUC: 0.5005594771114242
PR AUC: 0.9083057093435957
---------- End SVC ----------
roc_metrics
[[0.8477, 0.8512, 0.8996, 0.6106, 0.8875, 0.5, 0.6128, 0.8951, 0.5], [0.828, 0.9053, 0.9005, 0.6266, 0.8945, 0.5, 0.6761, 0.9249, 0.5], [0.8429, 0.9108, 0.9149, 0.6301, 0.9194, 0.5, 0.7993, 0.9288, 0.5], [0.8325, 0.9199, 0.9204, 0.6411, 0.8187, 0.5, 0.6908, 0.9294, 0.5], [0.8807, 0.9242, 0.9122, 0.6543, 0.9065, 0.5, 0.8001, 0.9324, 0.5006]]
pr_metrics
[[0.9715, 0.9727, 0.9806, 0.9282, 0.9781, 0.9082, 0.9288, 0.9806, 0.9082], [0.9671, 0.9823, 0.9806, 0.931, 0.9797, 0.9082, 0.9401, 0.9858, 0.9082], [0.9699, 0.9833, 0.9835, 0.9316, 0.9845, 0.9082, 0.9624, 0.9864, 0.9082], [0.9678, 0.985, 0.9846, 0.9336, 0.9647, 0.9082, 0.9428, 0.9866, 0.9082], [0.9773, 0.9856, 0.9829, 0.936, 0.9817, 0.9082, 0.9626, 0.9869, 0.9083]]
f1_metrics
[[0.4305, 0.4056, 0.6688, 0.2071, 0.6782, 0.1682, 0.2074, 0.5055, 0.1682], [0.4836, 0.5564, 0.7148, 0.2156, 0.6298, 0.1682, 0.2421, 0.6496, 0.1682], [0.5192, 0.5686, 0.7171, 0.2176, 0.6836, 0.1682, 0.3662, 0.6925, 0.1682], [0.5223, 0.6031, 0.7213, 0.2231, 0.7324, 0.1682, 0.2519, 0.6741, 0.1682], [0.5553, 0.6594, 0.7426, 0.23, 0.7502, 0.1682, 0.362, 0.7594, 0.1683]]


'''









