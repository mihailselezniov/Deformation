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
                                if 0 in [i4, i5, i6]:
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
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 33, 0: 67}
y_pred {1: 4624165, 0: 8189468}
Accuracy: 0.7783115842322001
Confusion matrix:
[[6160052  811218]
 [2029416 3812947]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.75      0.88      0.81   6971270
           1       0.82      0.65      0.73   5842363

    accuracy                           0.78  12813633
   macro avg       0.79      0.77      0.77  12813633
weighted avg       0.79      0.78      0.77  12813633

ROC AUC: 0.7681359632764869
PR AUC: 0.6965248948261646
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 4154893, 0: 8658740}
Accuracy: 0.8154331406245208
Confusion matrix:
[[6632519  338751]
 [2026221 3816142]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.77      0.95      0.85   6971270
           1       0.92      0.65      0.76   5842363

    accuracy                           0.82  12813633
   macro avg       0.84      0.80      0.81  12813633
weighted avg       0.84      0.82      0.81  12813633

ROC AUC: 0.8022961208122538
PR AUC: 0.7580602221132129
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 33, 0: 67}
y_pred {1: 7233728, 0: 5579905}
Accuracy: 0.8253777831782758
Confusion matrix:
[[5156815 1814455]
 [ 423090 5419273]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.92      0.74      0.82   6971270
           1       0.75      0.93      0.83   5842363

    accuracy                           0.83  12813633
   macro avg       0.84      0.83      0.83  12813633
weighted avg       0.84      0.83      0.82  12813633

ROC AUC: 0.8336531397023643
PR AUC: 0.7279331962567532
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 33, 0: 67}
y_pred {0: 11248912, 1: 1564721}
Accuracy: 0.5979948856034819
Confusion matrix:
[[6534518  436752]
 [4714394 1127969]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.58      0.94      0.72   6971270
           1       0.72      0.19      0.30   5842363

    accuracy                           0.60  12813633
   macro avg       0.65      0.57      0.51  12813633
weighted avg       0.64      0.60      0.53  12813633

ROC AUC: 0.5652084898539557
PR AUC: 0.507097618997515
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 33, 0: 67}
y_pred {1: 7648119, 0: 5165514}
Accuracy: 0.8120139698085624
Confusion matrix:
[[4864000 2107270]
 [ 301514 5540849]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.94      0.70      0.80   6971270
           1       0.72      0.95      0.82   5842363

    accuracy                           0.81  12813633
   macro avg       0.83      0.82      0.81  12813633
weighted avg       0.84      0.81      0.81  12813633

ROC AUC: 0.8230562803114654
PR AUC: 0.7106141188089076
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 33, 0: 67}
y_pred {0: 12813633}
Accuracy: 0.5440510119183217
Confusion matrix:
[[6971270       0]
 [5842363       0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.54      1.00      0.70   6971270
           1       0.00      0.00      0.00   5842363

    accuracy                           0.54  12813633
   macro avg       0.27      0.50      0.35  12813633
weighted avg       0.30      0.54      0.38  12813633

ROC AUC: 0.5
PR AUC: 0.45594898808167833
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 1819970, 0: 10993663}
Accuracy: 0.6277990012668538
Confusion matrix:
[[6597843  373427]
 [4395820 1446543]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.60      0.95      0.73   6971270
           1       0.79      0.25      0.38   5842363

    accuracy                           0.63  12813633
   macro avg       0.70      0.60      0.56  12813633
weighted avg       0.69      0.63      0.57  12813633

ROC AUC: 0.5970144847388369
PR AUC: 0.5398511822225242
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 5230006, 0: 7583627}
Accuracy: 0.8488591799062764
Confusion matrix:
[[6309117  662153]
 [1274510 4567853]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.83      0.91      0.87   6971270
           1       0.87      0.78      0.83   5842363

    accuracy                           0.85  12813633
   macro avg       0.85      0.84      0.85  12813633
weighted avg       0.85      0.85      0.85  12813633

ROC AUC: 0.8434335655884747
PR AUC: 0.7823280493969287
---------- End MLPClassifier ----------

 ---------- SVC ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 33, 0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 12813633}
Accuracy: 0.5440510119183217
Confusion matrix:
[[6971270       0]
 [5842363       0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.54      1.00      0.70   6971270
           1       0.00      0.00      0.00   5842363

    accuracy                           0.54  12813633
   macro avg       0.27      0.50      0.35  12813633
weighted avg       0.30      0.54      0.38  12813633

ROC AUC: 0.5
PR AUC: 0.45594898808167833
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
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 69, 0: 131}
y_pred {1: 4380276, 0: 8433357}
Accuracy: 0.7362314809546988
Confusion matrix:
[[6012397  958873]
 [2420960 3421403]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.71      0.86      0.78   6971270
           1       0.78      0.59      0.67   5842363

    accuracy                           0.74  12813633
   macro avg       0.75      0.72      0.72  12813633
weighted avg       0.74      0.74      0.73  12813633

ROC AUC: 0.7240366679456796
PR AUC: 0.6463597446529423
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 5778870, 0: 7034763}
Accuracy: 0.8505158529200891
Confusion matrix:
[[6045299  925971]
 [ 989464 4852899]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.86      0.87      0.86   6971270
           1       0.84      0.83      0.84   5842363

    accuracy                           0.85  12813633
   macro avg       0.85      0.85      0.85  12813633
weighted avg       0.85      0.85      0.85  12813633

ROC AUC: 0.8489065148468484
PR AUC: 0.7747627242192723
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 69, 0: 131}
y_pred {1: 7545366, 0: 5268267}
Accuracy: 0.8152644921233502
Confusion matrix:
[[4936202 2035068]
 [ 332065 5510298]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.94      0.71      0.81   6971270
           1       0.73      0.94      0.82   5842363

    accuracy                           0.82  12813633
   macro avg       0.83      0.83      0.81  12813633
weighted avg       0.84      0.82      0.81  12813633

ROC AUC: 0.8256202103569975
PR AUC: 0.71469624014201
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 69, 0: 131}
y_pred {0: 10511323, 1: 2302310}
Accuracy: 0.6114854389851808
Confusion matrix:
[[6252155  719115]
 [4259168 1583195]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.59      0.90      0.72   6971270
           1       0.69      0.27      0.39   5842363

    accuracy                           0.61  12813633
   macro avg       0.64      0.58      0.55  12813633
weighted avg       0.64      0.61      0.57  12813633

ROC AUC: 0.5839156495561275
PR AUC: 0.5187379334441592
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 69, 0: 131}
y_pred {1: 7066982, 0: 5746651}
Accuracy: 0.8274177979032176
Confusion matrix:
[[5253258 1718012]
 [ 493393 5348970]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.91      0.75      0.83   6971270
           1       0.76      0.92      0.83   5842363

    accuracy                           0.83  12813633
   macro avg       0.84      0.83      0.83  12813633
weighted avg       0.84      0.83      0.83  12813633

ROC AUC: 0.8345536574965688
PR AUC: 0.7314806915420904
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 69, 0: 131}
y_pred {0: 12813633}
Accuracy: 0.5440510119183217
Confusion matrix:
[[6971270       0]
 [5842363       0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.54      1.00      0.70   6971270
           1       0.00      0.00      0.00   5842363

    accuracy                           0.54  12813633
   macro avg       0.27      0.50      0.35  12813633
weighted avg       0.30      0.54      0.38  12813633

ROC AUC: 0.5
PR AUC: 0.45594898808167833
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 2564609, 0: 10249024}
Accuracy: 0.6371013591539574
Confusion matrix:
[[6285122  686148]
 [3963902 1878461]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.61      0.90      0.73   6971270
           1       0.73      0.32      0.45   5842363

    accuracy                           0.64  12813633
   macro avg       0.67      0.61      0.59  12813633
weighted avg       0.67      0.64      0.60  12813633

ROC AUC: 0.6115495385144917
PR AUC: 0.544852397593871
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 5902111, 0: 6911522}
Accuracy: 0.8586237018026035
Confusion matrix:
[[6035624  935646]
 [ 875898 4966465]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.87      0.87      0.87   6971270
           1       0.84      0.85      0.85   5842363

    accuracy                           0.86  12813633
   macro avg       0.86      0.86      0.86  12813633
weighted avg       0.86      0.86      0.86  12813633

ROC AUC: 0.8579317792047412
PR AUC: 0.7836742231023713
---------- End MLPClassifier ----------

 ---------- SVC ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 69, 0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 12813633}
Accuracy: 0.5440510119183217
Confusion matrix:
[[6971270       0]
 [5842363       0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.54      1.00      0.70   6971270
           1       0.00      0.00      0.00   5842363

    accuracy                           0.54  12813633
   macro avg       0.27      0.50      0.35  12813633
weighted avg       0.30      0.54      0.38  12813633

ROC AUC: 0.5
PR AUC: 0.45594898808167833
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
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 98, 0: 202}
y_pred {1: 4769837, 0: 8043796}
Accuracy: 0.7589945021837289
Confusion matrix:
[[5963455 1007815]
 [2080341 3762022]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.74      0.86      0.79   6971270
           1       0.79      0.64      0.71   5842363

    accuracy                           0.76  12813633
   macro avg       0.77      0.75      0.75  12813633
weighted avg       0.76      0.76      0.76  12813633

ROC AUC: 0.7496771942030611
PR AUC: 0.6702214137748028
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 5911207, 0: 6902426}
Accuracy: 0.8482297721497096
Confusion matrix:
[[5964484 1006786]
 [ 937942 4904421]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.86      0.86      0.86   6971270
           1       0.83      0.84      0.83   5842363

    accuracy                           0.85  12813633
   macro avg       0.85      0.85      0.85  12813633
weighted avg       0.85      0.85      0.85  12813633

ROC AUC: 0.8475195712310355
PR AUC: 0.7696821751592213
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 98, 0: 202}
y_pred {1: 7360613, 0: 5453020}
Accuracy: 0.8217857495996647
Confusion matrix:
[[5070359 1900911]
 [ 382661 5459702]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.93      0.73      0.82   6971270
           1       0.74      0.93      0.83   5842363

    accuracy                           0.82  12813633
   macro avg       0.84      0.83      0.82  12813633
weighted avg       0.84      0.82      0.82  12813633

ROC AUC: 0.8309122479478563
PR AUC: 0.723026556981519
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 98, 0: 202}
y_pred {0: 10489869, 1: 2323764}
Accuracy: 0.5981811715693746
Confusion matrix:
[[6156190  815080]
 [4333679 1508684]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.59      0.88      0.71   6971270
           1       0.65      0.26      0.37   5842363

    accuracy                           0.60  12813633
   macro avg       0.62      0.57      0.54  12813633
weighted avg       0.62      0.60      0.55  12813633

ROC AUC: 0.5706559711006584
PR AUC: 0.5058632614358072
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 98, 0: 202}
y_pred {1: 7022809, 0: 5790824}
Accuracy: 0.8305668657749133
Confusion matrix:
[[5295520 1675750]
 [ 495304 5347059]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.91      0.76      0.83   6971270
           1       0.76      0.92      0.83   5842363

    accuracy                           0.83  12813633
   macro avg       0.84      0.84      0.83  12813633
weighted avg       0.84      0.83      0.83  12813633

ROC AUC: 0.8374212656696575
PR AUC: 0.7354904212198138
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 98, 0: 202}
y_pred {0: 12813633}
Accuracy: 0.5440510119183217
Confusion matrix:
[[6971270       0]
 [5842363       0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.54      1.00      0.70   6971270
           1       0.00      0.00      0.00   5842363

    accuracy                           0.54  12813633
   macro avg       0.27      0.50      0.35  12813633
weighted avg       0.30      0.54      0.38  12813633

ROC AUC: 0.5
PR AUC: 0.45594898808167833
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 4373457, 0: 8440176}
Accuracy: 0.6944952301973999
Confusion matrix:
[[5748410 1222860]
 [2691766 3150597]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.68      0.82      0.75   6971270
           1       0.72      0.54      0.62   5842363

    accuracy                           0.69  12813633
   macro avg       0.70      0.68      0.68  12813633
weighted avg       0.70      0.69      0.69  12813633

ROC AUC: 0.6819266757952356
PR AUC: 0.5985537396529246
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 6597622, 0: 6216011}
Accuracy: 0.8498652958142316
Confusion matrix:
[[5631755 1339515]
 [ 584256 5258107]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.91      0.81      0.85   6971270
           1       0.80      0.90      0.85   5842363

    accuracy                           0.85  12813633
   macro avg       0.85      0.85      0.85  12813633
weighted avg       0.86      0.85      0.85  12813633

ROC AUC: 0.8539243562085652
PR AUC: 0.7628667754654812
---------- End MLPClassifier ----------

 ---------- SVC ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 98, 0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 12813633}
Accuracy: 0.5440510119183217
Confusion matrix:
[[6971270       0]
 [5842363       0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.54      1.00      0.70   6971270
           1       0.00      0.00      0.00   5842363

    accuracy                           0.54  12813633
   macro avg       0.27      0.50      0.35  12813633
weighted avg       0.30      0.54      0.38  12813633

ROC AUC: 0.5
PR AUC: 0.45594898808167833
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
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 132, 0: 268}
y_pred {1: 5113889, 0: 7699744}
Accuracy: 0.7647685086657312
Confusion matrix:
[[5828422 1142848]
 [1871322 3971041]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.76      0.84      0.79   6971270
           1       0.78      0.68      0.72   5842363

    accuracy                           0.76  12813633
   macro avg       0.77      0.76      0.76  12813633
weighted avg       0.77      0.76      0.76  12813633

ROC AUC: 0.7578804543939857
PR AUC: 0.6738409045586233
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 6240971, 0: 6572662}
Accuracy: 0.8444188310996577
Confusion matrix:
[[5775186 1196084]
 [ 797476 5044887]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.88      0.83      0.85   6971270
           1       0.81      0.86      0.84   5842363

    accuracy                           0.84  12813633
   macro avg       0.84      0.85      0.84  12813633
weighted avg       0.85      0.84      0.84  12813633

ROC AUC: 0.8459638961082797
PR AUC: 0.760247392957175
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 132, 0: 268}
y_pred {1: 7383341, 0: 5430292}
Accuracy: 0.8189412791828828
Confusion matrix:
[[5040771 1930499]
 [ 389521 5452842]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.93      0.72      0.81   6971270
           1       0.74      0.93      0.82   5842363

    accuracy                           0.82  12813633
   macro avg       0.83      0.83      0.82  12813633
weighted avg       0.84      0.82      0.82  12813633

ROC AUC: 0.8282030182976249
PR AUC: 0.719692737475798
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 132, 0: 268}
y_pred {0: 10333315, 1: 2480318}
Accuracy: 0.5928562180608732
Confusion matrix:
[[6043797  927473]
 [4289518 1552845]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.58      0.87      0.70   6971270
           1       0.63      0.27      0.37   5842363

    accuracy                           0.59  12813633
   macro avg       0.61      0.57      0.54  12813633
weighted avg       0.60      0.59      0.55  12813633

ROC AUC: 0.5663741926914944
PR AUC: 0.5011647249564791
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 132, 0: 268}
y_pred {1: 8769637, 0: 4043996}
Accuracy: 0.7441180030675141
Confusion matrix:
[[3868244 3103026]
 [ 175752 5666611]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.96      0.55      0.70   6971270
           1       0.65      0.97      0.78   5842363

    accuracy                           0.74  12813633
   macro avg       0.80      0.76      0.74  12813633
weighted avg       0.82      0.74      0.74  12813633

ROC AUC: 0.7624006691871758
PR AUC: 0.6404404786171237
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 132, 0: 268}
y_pred {0: 12813633}
Accuracy: 0.5440510119183217
Confusion matrix:
[[6971270       0]
 [5842363       0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.54      1.00      0.70   6971270
           1       0.00      0.00      0.00   5842363

    accuracy                           0.54  12813633
   macro avg       0.27      0.50      0.35  12813633
weighted avg       0.30      0.54      0.38  12813633

ROC AUC: 0.5
PR AUC: 0.45594898808167833
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 2872078, 0: 9941555}
Accuracy: 0.6503975882561955
Confusion matrix:
[[6216574  754696]
 [3724981 2117382]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.63      0.89      0.74   6971270
           1       0.74      0.36      0.49   5842363

    accuracy                           0.65  12813633
   macro avg       0.68      0.63      0.61  12813633
weighted avg       0.68      0.65      0.62  12813633

ROC AUC: 0.6270803658441968
PR AUC: 0.5578904977823175
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 6828458, 0: 5985175}
Accuracy: 0.8455684660236484
Confusion matrix:
[[5488808 1482462]
 [ 496367 5345996]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.92      0.79      0.85   6971270
           1       0.78      0.92      0.84   5842363

    accuracy                           0.85  12813633
   macro avg       0.85      0.85      0.85  12813633
weighted avg       0.86      0.85      0.85  12813633

ROC AUC: 0.8511934763887496
PR AUC: 0.7551217460547108
---------- End MLPClassifier ----------

 ---------- SVC ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 132, 0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 12813633}
Accuracy: 0.5440510119183217
Confusion matrix:
[[6971270       0]
 [5842363       0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.54      1.00      0.70   6971270
           1       0.00      0.00      0.00   5842363

    accuracy                           0.54  12813633
   macro avg       0.27      0.50      0.35  12813633
weighted avg       0.30      0.54      0.38  12813633

ROC AUC: 0.5
PR AUC: 0.45594898808167833
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
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 167, 0: 333}
y_pred {1: 5281988, 0: 7531645}
Accuracy: 0.7999848286586638
Confusion matrix:
[[5969997 1001273]
 [1561648 4280715]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.79      0.86      0.82   6971270
           1       0.81      0.73      0.77   5842363

    accuracy                           0.80  12813633
   macro avg       0.80      0.79      0.80  12813633
weighted avg       0.80      0.80      0.80  12813633

ROC AUC: 0.7945370914504335
PR AUC: 0.7156828160406022
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1: 6846969, 0: 5966664}
Accuracy: 0.8371300317404128
Confusion matrix:
[[5425489 1545781]
 [ 541175 5301188]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.91      0.78      0.84   6971270
           1       0.77      0.91      0.84   5842363

    accuracy                           0.84  12813633
   macro avg       0.84      0.84      0.84  12813633
weighted avg       0.85      0.84      0.84  12813633

ROC AUC: 0.8428173016278159
PR AUC: 0.7447556420108408
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 167, 0: 333}
y_pred {1: 7719144, 0: 5094489}
Accuracy: 0.8068281649708556
Confusion matrix:
[[4795263 2176007]
 [ 299226 5543137]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.94      0.69      0.79   6971270
           1       0.72      0.95      0.82   5842363

    accuracy                           0.81  12813633
   macro avg       0.83      0.82      0.81  12813633
weighted avg       0.84      0.81      0.81  12813633

ROC AUC: 0.818322071568027
PR AUC: 0.7046759356842653
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 167, 0: 333}
y_pred {0: 9989771, 1: 2823862}
Accuracy: 0.6150547623769153
Confusion matrix:
[[6014247  957023]
 [3975524 1866839]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.60      0.86      0.71   6971270
           1       0.66      0.32      0.43   5842363

    accuracy                           0.62  12813633
   macro avg       0.63      0.59      0.57  12813633
weighted avg       0.63      0.62      0.58  12813633

ROC AUC: 0.5911269545669132
PR AUC: 0.5215000709627214
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 167, 0: 333}
y_pred {1: 7818261, 0: 4995372}
Accuracy: 0.7966314471469567
Confusion matrix:
[[4680376 2290894]
 [ 314996 5527367]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.94      0.67      0.78   6971270
           1       0.71      0.95      0.81   5842363

    accuracy                           0.80  12813633
   macro avg       0.82      0.81      0.80  12813633
weighted avg       0.83      0.80      0.79  12813633

ROC AUC: 0.8087324126864404
PR AUC: 0.6934470017517969
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 167, 0: 333}
y_pred {0: 12813633}
Accuracy: 0.5440510119183217
Confusion matrix:
[[6971270       0]
 [5842363       0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.54      1.00      0.70   6971270
           1       0.00      0.00      0.00   5842363

    accuracy                           0.54  12813633
   macro avg       0.27      0.50      0.35  12813633
weighted avg       0.30      0.54      0.38  12813633

ROC AUC: 0.5
PR AUC: 0.45594898808167833
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1: 4532590, 0: 8281043}
Accuracy: 0.7550799995598438
Confusion matrix:
[[6056999  914271]
 [2224044 3618319]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.73      0.87      0.79   6971270
           1       0.80      0.62      0.70   5842363

    accuracy                           0.76  12813633
   macro avg       0.76      0.74      0.75  12813633
weighted avg       0.76      0.76      0.75  12813633

ROC AUC: 0.7440880823345877
PR AUC: 0.6679688805306769
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1: 7230258, 0: 5583375}
Accuracy: 0.8399299402441134
Confusion matrix:
[[5251783 1719487]
 [ 331592 5510771]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.94      0.75      0.84   6971270
           1       0.76      0.94      0.84   5842363

    accuracy                           0.84  12813633
   macro avg       0.85      0.85      0.84  12813633
weighted avg       0.86      0.84      0.84  12813633

ROC AUC: 0.8482950885896227
PR AUC: 0.7448010939498765
---------- End MLPClassifier ----------

 ---------- SVC ----------
(12813633, 8) (12813633,)
y_test {1: 5842363, 0: 6971270} y_train {1: 167, 0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {0: 12813633}
Accuracy: 0.5440510119183217
Confusion matrix:
[[6971270       0]
 [5842363       0]]
Precision, recall and f1-score:
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           0       0.54      1.00      0.70   6971270
           1       0.00      0.00      0.00   5842363

    accuracy                           0.54  12813633
   macro avg       0.27      0.50      0.35  12813633
weighted avg       0.30      0.54      0.38  12813633

ROC AUC: 0.5
PR AUC: 0.45594898808167833
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
---------- End SVC ----------
roc_metrics
[[0.7681, 0.8023, 0.8337, 0.5652, 0.8231, 0.5, 0.597, 0.8434, 0.5], [0.724, 0.8489, 0.8256, 0.5839, 0.8346, 0.5, 0.6115, 0.8579, 0.5], [0.7497, 0.8475, 0.8309, 0.5707, 0.8374, 0.5, 0.6819, 0.8539, 0.5], [0.7579, 0.846, 0.8282, 0.5664, 0.7624, 0.5, 0.6271, 0.8512, 0.5], [0.7945, 0.8428, 0.8183, 0.5911, 0.8087, 0.5, 0.7441, 0.8483, 0.5]]
pr_metrics
[[0.6965, 0.7581, 0.7279, 0.5071, 0.7106, 0.4559, 0.5399, 0.7823, 0.4559], [0.6464, 0.7748, 0.7147, 0.5187, 0.7315, 0.4559, 0.5449, 0.7837, 0.4559], [0.6702, 0.7697, 0.723, 0.5059, 0.7355, 0.4559, 0.5986, 0.7629, 0.4559], [0.6738, 0.7602, 0.7197, 0.5012, 0.6404, 0.4559, 0.5579, 0.7551, 0.4559], [0.7157, 0.7448, 0.7047, 0.5215, 0.6934, 0.4559, 0.668, 0.7448, 0.4559]]
f1_metrics
[[0.8126, 0.8487, 0.8217, 0.7173, 0.8015, 0.7047, 0.7345, 0.8669, 0.7047], [0.7806, 0.8632, 0.8066, 0.7152, 0.8261, 0.7047, 0.73, 0.8695, 0.7047], [0.7943, 0.8598, 0.8162, 0.7051, 0.8299, 0.7047, 0.746, 0.8541, 0.7047], [0.7945, 0.8528, 0.8129, 0.6985, 0.7023, 0.7047, 0.7351, 0.8473, 0.7047], [0.8233, 0.8387, 0.7949, 0.7092, 0.7822, 0.7047, 0.7942, 0.8366, 0.7047]]


'''









