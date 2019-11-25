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

for i in [0.0000046]:
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
!!! 1e-06 !!!
(72,) (72899928,)
y_test {1: 63518632, 0: 9381296} y_train {1: 59, 0: 13}

 ---------- XGBClassifier ----------
y_pred {1: 65650697, 0: 7249231}
Accuracy: 0.9112243430473621
Confusion matrix:
[[ 5079394  4301902]
 [ 2169837 61348795]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.70      0.54      0.61   9381296
           1       0.93      0.97      0.95  63518632

   micro avg       0.91      0.91      0.91  72899928
   macro avg       0.82      0.75      0.78  72899928
weighted avg       0.90      0.91      0.91  72899928

ROC AUC: 0.753638886486902
Precision-recall: 0.9323152742010169
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
y_pred {1: 62405843, 0: 10494085}
Accuracy: 0.9212890169109632
Confusion matrix:
[[ 7068678  2312618]
 [ 3425407 60093225]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.67      0.75      0.71   9381296
           1       0.96      0.95      0.95  63518632

   micro avg       0.92      0.92      0.92  72899928
   macro avg       0.82      0.85      0.83  72899928
weighted avg       0.93      0.92      0.92  72899928

ROC AUC: 0.849779352331099
Precision-recall: 0.9580009202828718
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
y_pred {1: 61598678, 0: 11301250}
Accuracy: 0.9147199432076256
Confusion matrix:
[[ 7232818  2148478]
 [ 4068432 59450200]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.64      0.77      0.70   9381296
           1       0.97      0.94      0.95  63518632

   micro avg       0.91      0.91      0.91  72899928
   macro avg       0.80      0.85      0.82  72899928
weighted avg       0.92      0.91      0.92  72899928

ROC AUC: 0.8534659067826831
Precision-recall: 0.959112822144336
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
y_pred {1: 66508158, 0: 6391770}
Accuracy: 0.8866190101038234
Confusion matrix:
[[ 3753800  5627496]
 [ 2637970 60880662]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.59      0.40      0.48   9381296
           1       0.92      0.96      0.94  63518632

   micro avg       0.89      0.89      0.89  72899928
   macro avg       0.75      0.68      0.71  72899928
weighted avg       0.87      0.89      0.88  72899928

ROC AUC: 0.6793029817104899
Precision-recall: 0.9135559709630336
---------- End KNeighborsClassifier ----------
!!! 1e-05 !!!
(729,) (72899271,)
y_test {1: 63518072, 0: 9381199} y_train {1: 619, 0: 110}

 ---------- XGBClassifier ----------
y_pred {1: 65221280, 0: 7677991}
Accuracy: 0.9494998790865824
Confusion matrix:
[[ 6688884  2692315]
 [  989107 62528965]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.87      0.71      0.78   9381199
           1       0.96      0.98      0.97  63518072

   micro avg       0.95      0.95      0.95  72899271
   macro avg       0.91      0.85      0.88  72899271
weighted avg       0.95      0.95      0.95  72899271

ROC AUC: 0.8487187213496794
Precision-recall: 0.9573591844166848
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
y_pred {1: 63970581, 0: 8928690}
Accuracy: 0.9415619807775581
Confusion matrix:
[[ 7024900  2356299]
 [ 1903790 61614282]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.79      0.75      0.77   9381199
           1       0.96      0.97      0.97  63518072

   micro avg       0.94      0.94      0.94  72899271
   macro avg       0.87      0.86      0.87  72899271
weighted avg       0.94      0.94      0.94  72899271

ROC AUC: 0.8594275535802908
Precision-recall: 0.9604128390879261
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
y_pred {1: 63750700, 0: 9148571}
Accuracy: 0.9410830047943827
Confusion matrix:
[[ 7117382  2263817]
 [ 2031189 61486883]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.78      0.76      0.77   9381199
           1       0.96      0.97      0.97  63518072

   micro avg       0.94      0.94      0.94  72899271
   macro avg       0.87      0.86      0.87  72899271
weighted avg       0.94      0.94      0.94  72899271

ROC AUC: 0.863353811331003
Precision-recall: 0.9615099192958095
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
y_pred {1: 64579208, 0: 8320063}
Accuracy: 0.9172075397022832
Confusion matrix:
[[ 5832876  3548323]
 [ 2487187 61030885]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.70      0.62      0.66   9381199
           1       0.95      0.96      0.95  63518072

   micro avg       0.92      0.92      0.92  72899271
   macro avg       0.82      0.79      0.81  72899271
weighted avg       0.91      0.92      0.92  72899271

ROC AUC: 0.7913025795148146
Precision-recall: 0.9421671934089035
---------- End KNeighborsClassifier ----------
!!! 0.0001 !!!
(7290,) (72892710,)
y_test {1: 63512321, 0: 9380389} y_train {1: 6370, 0: 920}

 ---------- XGBClassifier ----------
y_pred {1: 65136249, 0: 7756461}
Accuracy: 0.9649712296332514
Confusion matrix:
[[ 7291754  2088635]
 [  464707 63047614]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.94      0.78      0.85   9380389
           1       0.97      0.99      0.98  63512321

   micro avg       0.96      0.96      0.96  72892710
   macro avg       0.95      0.89      0.92  72892710
weighted avg       0.96      0.96      0.96  72892710

ROC AUC: 0.8850117283208269
Precision-recall: 0.9672274040513297
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
y_pred {1: 64608218, 0: 8284492}
Accuracy: 0.9423438777348242
Confusion matrix:
[[ 6731085  2649304]
 [ 1553407 61958914]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.81      0.72      0.76   9380389
           1       0.96      0.98      0.97  63512321

   micro avg       0.94      0.94      0.94  72892710
   macro avg       0.89      0.85      0.86  72892710
weighted avg       0.94      0.94      0.94  72892710

ROC AUC: 0.8465557825015777
Precision-recall: 0.9568497626419583
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
y_pred {1: 61605547, 0: 11287163}
Accuracy: 0.9322772880854615
Confusion matrix:
[[ 7865530  1514859]
 [ 3421633 60090688]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.70      0.84      0.76   9380389
           1       0.98      0.95      0.96  63512321

   micro avg       0.93      0.93      0.93  72892710
   macro avg       0.84      0.89      0.86  72892710
weighted avg       0.94      0.93      0.93  72892710

ROC AUC: 0.8923171694463289
Precision-recall: 0.969802222570879
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
y_pred {1: 64491701, 0: 8401009}
Accuracy: 0.9397825104869884
Confusion matrix:
[[ 6695991  2684398]
 [ 1705018 61807303]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.80      0.71      0.75   9380389
           1       0.96      0.97      0.97  63512321

   micro avg       0.94      0.94      0.94  72892710
   macro avg       0.88      0.84      0.86  72892710
weighted avg       0.94      0.94      0.94  72892710

ROC AUC: 0.8434916219609028
Precision-recall: 0.9560388053290056
---------- End KNeighborsClassifier ----------
!!! 0.001 !!!
(72900,) (72827100,)
y_test {1: 63455375, 0: 9371725} y_train {1: 63316, 0: 9584}

 ---------- XGBClassifier ----------
y_pred {1: 64838331, 0: 7988769}
Accuracy: 0.9685091676038178
Confusion matrix:
[[ 7533554  1838171]
 [  455215 63000160]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.94      0.80      0.87   9371725
           1       0.97      0.99      0.98  63455375

   micro avg       0.97      0.97      0.97  72827100
   macro avg       0.96      0.90      0.93  72827100
weighted avg       0.97      0.97      0.97  72827100

ROC AUC: 0.8983430634099094
Precision-recall: 0.9709301555262898
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
y_pred {1: 64365726, 0: 8461374}
Accuracy: 0.9427584923744046
Confusion matrix:
[[ 6832183  2539542]
 [ 1629191 61826184]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.81      0.73      0.77   9371725
           1       0.96      0.97      0.97  63455375

   micro avg       0.94      0.94      0.94  72827100
   macro avg       0.88      0.85      0.87  72827100
weighted avg       0.94      0.94      0.94  72827100

ROC AUC: 0.8516731318996398
Precision-recall: 0.9582541835181273
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
y_pred {1: 65253306, 0: 7573794}
Accuracy: 0.9419462672549093
Confusion matrix:
[[ 6358817  3012908]
 [ 1214977 62240398]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.84      0.68      0.75   9371725
           1       0.95      0.98      0.97  63455375

   micro avg       0.94      0.94      0.94  72827100
   macro avg       0.90      0.83      0.86  72827100
weighted avg       0.94      0.94      0.94  72827100

ROC AUC: 0.8296819440451648
Precision-recall: 0.9522476492088243
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
y_pred {1: 63532265, 0: 9294835}
Accuracy: 0.9546614653061841
Confusion matrix:
[[ 7682343  1689382]
 [ 1612492 61842883]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.83      0.82      0.82   9371725
           1       0.97      0.97      0.97  63455375

   micro avg       0.95      0.95      0.95  72827100
   macro avg       0.90      0.90      0.90  72827100
weighted avg       0.95      0.95      0.95  72827100

ROC AUC: 0.8971624241849623
Precision-recall: 0.9708147241659649
---------- End KNeighborsClassifier ----------
!!! 0.01 !!!
(729000,) (72171000,)
y_test {1: 62883803, 0: 9287197} y_train {0: 94112, 1: 634888}

 ---------- XGBClassifier ----------
y_pred {1: 64322965, 0: 7848035}
Accuracy: 0.9688511174848623
Confusion matrix:
[[ 7443593  1843604]
 [  404442 62479361]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.95      0.80      0.87   9287197
           1       0.97      0.99      0.98  62883803

   micro avg       0.97      0.97      0.97  72171000
   macro avg       0.96      0.90      0.93  72171000
weighted avg       0.97      0.97      0.97  72171000

ROC AUC: 0.8975290758066602
Precision-recall: 0.9706950274671161
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
y_pred {1: 63895241, 0: 8275759}
Accuracy: 0.9428275068933505
Confusion matrix:
[[ 6718380  2568817]
 [ 1557379 61326424]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.81      0.72      0.77   9287197
           1       0.96      0.98      0.97  62883803

   micro avg       0.94      0.94      0.94  72171000
   macro avg       0.89      0.85      0.87  72171000
weighted avg       0.94      0.94      0.94  72171000

ROC AUC: 0.849318177806397
Precision-recall: 0.9576051415141368
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
y_pred {1: 64085510, 0: 8085490}
Accuracy: 0.9423312549361932
Confusion matrix:
[[ 6605338  2681859]
 [ 1480152 61403651]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.82      0.71      0.76   9287197
           1       0.96      0.98      0.97  62883803

   micro avg       0.94      0.94      0.94  72171000
   macro avg       0.89      0.84      0.86  72171000
weighted avg       0.94      0.94      0.94  72171000

ROC AUC: 0.8438463179034345
Precision-recall: 0.9561079513937878
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
y_pred {1: 62688325, 0: 9482675}
Accuracy: 0.9670704853750122
Confusion matrix:
[[ 8196658  1090539]
 [ 1286017 61597786]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.86      0.88      0.87   9287197
           1       0.98      0.98      0.98  62883803

   micro avg       0.97      0.97      0.97  72171000
   macro avg       0.92      0.93      0.93  72171000
weighted avg       0.97      0.97      0.97  72171000

ROC AUC: 0.9310627004567461
Precision-recall: 0.980327898209679
---------- End KNeighborsClassifier ----------
'''


