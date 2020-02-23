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

with open('../11/fib_all_data.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))

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
x_test, y_test = a, np.array(y)
print(x_test.shape, y_test.shape)
print('all', dict(collections.Counter(y_test)))


source_f = '../12/ml_threads/6_1.txt'
with open(source_f, 'r') as f:
    threads = f.readlines()

x_train, y_train = [], []
for t in threads:
    tr = list(map(float, t.replace('\n', '').split(',')))
    x_train.append(tr[:-1])
    y_train.append(tr[-1])

x, y = np.array(x_train), np.array(y_train)

#print(x_train[:100,:].shape)


roc_metrics, pr_metrics, f1_metrics = [], [], []
roc_metric, pr_metric, f1_metric = [], [], []

for cut in [100, 200, 300, 400, 500]:

    print('\n\n\n', '#'*10, cut, '#'*10)
    x_train, y_train = x[:cut,:], y[:cut]

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



'''
(72900000, 8) (72900000,)
all {1: 63518691, 0: 9381309}



 ########## 100 ##########

 ---------- XGBClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 33, 0.0: 67}
y_pred {1.0: 49480557, 0.0: 23419443}
Accuracy: 0.7928843347050755
Confusion matrix:
[[ 8851010   530299]
 [14568433 48950258]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.38      0.94      0.54   9381309
           1       0.99      0.77      0.87  63518691

    accuracy                           0.79  72900000
   macro avg       0.68      0.86      0.70  72900000
weighted avg       0.91      0.79      0.82  72900000

ROC AUC: 0.8570580912240032
PR AUC: 0.9622254670084294
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 33, 0.0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1.0: 61562165, 0.0: 11337835}
Accuracy: 0.922334622770919
Confusion matrix:
[[ 7528669  1852640]
 [ 3809166 59709525]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.66      0.80      0.73   9381309
           1       0.97      0.94      0.95  63518691

    accuracy                           0.92  72900000
   macro avg       0.82      0.87      0.84  72900000
weighted avg       0.93      0.92      0.93  72900000

ROC AUC: 0.8712743748807678
PR AUC: 0.9639936133614551
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 33, 0.0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
y_pred {1.0: 61044864, 0.0: 11855136}
Accuracy: 0.9180395747599451
Confusion matrix:
[[ 7630765  1750544]
 [ 4224371 59294320]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.64      0.81      0.72   9381309
           1       0.97      0.93      0.95  63518691

    accuracy                           0.92  72900000
   macro avg       0.81      0.87      0.84  72900000
weighted avg       0.93      0.92      0.92  72900000

ROC AUC: 0.8734474643211912
PR AUC: 0.9646723204298336
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 33, 0.0: 67}
y_pred {0.0: 54632183, 1.0: 18267817}
Accuracy: 0.36637303155006856
Confusion matrix:
[[ 8911043   470266]
 [45721140 17797551]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.16      0.95      0.28   9381309
           1       0.97      0.28      0.44  63518691

    accuracy                           0.37  72900000
   macro avg       0.57      0.62      0.36  72900000
weighted avg       0.87      0.37      0.42  72900000

ROC AUC: 0.6150329735335501
PR AUC: 0.9001570562393508
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 33, 0.0: 67}
y_pred {1.0: 58501773, 0.0: 14398227}
Accuracy: 0.9063527846364884
Confusion matrix:
[[ 8476327   904982]
 [ 5921900 57596791]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.59      0.90      0.71   9381309
           1       0.98      0.91      0.94  63518691

    accuracy                           0.91  72900000
   macro avg       0.79      0.91      0.83  72900000
weighted avg       0.93      0.91      0.91  72900000

ROC AUC: 0.9051513396029623
PR AUC: 0.973975278110118
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 33, 0.0: 67}
y_pred {1.0: 19755900, 0.0: 53144100}
Accuracy: 0.3927734293552812
Confusion matrix:
[[ 9129296   252013]
 [44014804 19503887]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.17      0.97      0.29   9381309
           1       0.99      0.31      0.47  63518691

    accuracy                           0.39  72900000
   macro avg       0.58      0.64      0.38  72900000
weighted avg       0.88      0.39      0.45  72900000

ROC AUC: 0.6400970683830742
PR AUC: 0.9069101181698512
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 33, 0.0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1.0: 27736366, 0.0: 45163634}
Accuracy: 0.5005758436213992
Confusion matrix:
[[ 9068461   312848]
 [36095173 27423518]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.20      0.97      0.33   9381309
           1       0.99      0.43      0.60  63518691

    accuracy                           0.50  72900000
   macro avg       0.59      0.70      0.47  72900000
weighted avg       0.89      0.50      0.57  72900000

ROC AUC: 0.6991956663975556
PR AUC: 0.9220022964920123
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 33, 0.0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1.0: 58371015, 0.0: 14528985}
Accuracy: 0.8999976406035666
Confusion matrix:
[[ 8310061  1071248]
 [ 6218924 57299767]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.57      0.89      0.70   9381309
           1       0.98      0.90      0.94  63518691

    accuracy                           0.90  72900000
   macro avg       0.78      0.89      0.82  72900000
weighted avg       0.93      0.90      0.91  72900000

ROC AUC: 0.8939516996290774
PR AUC: 0.970845038625496
---------- End MLPClassifier ----------

 ---------- SVC ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 33, 0.0: 67}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {1.0: 26962, 0.0: 72873038}
Accuracy: 0.1289778463648834
Confusion matrix:
[[ 9378416     2893]
 [63494622    24069]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.13      1.00      0.23   9381309
           1       0.89      0.00      0.00  63518691

    accuracy                           0.13  72900000
   macro avg       0.51      0.50      0.11  72900000
weighted avg       0.79      0.13      0.03  72900000

ROC AUC: 0.5000352743470864
PR AUC: 0.8713207383308499
---------- End SVC ----------



 ########## 200 ##########

 ---------- XGBClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 69, 0.0: 131}
y_pred {1.0: 54780278, 0.0: 18119722}
Accuracy: 0.852467146776406
Confusion matrix:
[[ 8372943  1008366]
 [ 9746779 53771912]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.46      0.89      0.61   9381309
           1       0.98      0.85      0.91  63518691

    accuracy                           0.85  72900000
   macro avg       0.72      0.87      0.76  72900000
weighted avg       0.91      0.85      0.87  72900000

ROC AUC: 0.8695329391397573
PR AUC: 0.9646703698908299
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 69, 0.0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1.0: 62325191, 0.0: 10574809}
Accuracy: 0.9315652126200274
Confusion matrix:
[[ 7483611  1897698]
 [ 3091198 60427493]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.71      0.80      0.75   9381309
           1       0.97      0.95      0.96  63518691

    accuracy                           0.93  72900000
   macro avg       0.84      0.87      0.86  72900000
weighted avg       0.94      0.93      0.93  72900000

ROC AUC: 0.8745245254475192
PR AUC: 0.9647707728465691
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 69, 0.0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
y_pred {1.0: 62467704, 0.0: 10432296}
Accuracy: 0.932191536351166
Confusion matrix:
[[ 7435184  1946125]
 [ 2997112 60521579]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.71      0.79      0.75   9381309
           1       0.97      0.95      0.96  63518691

    accuracy                           0.93  72900000
   macro avg       0.84      0.87      0.86  72900000
weighted avg       0.94      0.93      0.93  72900000

ROC AUC: 0.8726841056962242
PR AUC: 0.964243821085404
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 69, 0.0: 131}
y_pred {0.0: 50614595, 1.0: 22285405}
Accuracy: 0.4128573388203018
Confusion matrix:
[[ 8596602   784707]
 [42017993 21500698]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.17      0.92      0.29   9381309
           1       0.96      0.34      0.50  63518691

    accuracy                           0.41  72900000
   macro avg       0.57      0.63      0.39  72900000
weighted avg       0.86      0.41      0.47  72900000

ROC AUC: 0.6274241225211241
PR AUC: 0.9029535890961998
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 69, 0.0: 131}
y_pred {1.0: 64939872, 0.0: 7960128}
Accuracy: 0.9354426474622771
Confusion matrix:
[[ 6317603  3063706]
 [ 1642525 61876166]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.79      0.67      0.73   9381309
           1       0.95      0.97      0.96  63518691

    accuracy                           0.94  72900000
   macro avg       0.87      0.82      0.85  72900000
weighted avg       0.93      0.94      0.93  72900000

ROC AUC: 0.8237827708272639
PR AUC: 0.9507146647330629
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 69, 0.0: 131}
y_pred {1.0: 19755900, 0.0: 53144100}
Accuracy: 0.3927734293552812
Confusion matrix:
[[ 9129296   252013]
 [44014804 19503887]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.17      0.97      0.29   9381309
           1       0.99      0.31      0.47  63518691

    accuracy                           0.39  72900000
   macro avg       0.58      0.64      0.38  72900000
weighted avg       0.88      0.39      0.45  72900000

ROC AUC: 0.6400970683830742
PR AUC: 0.9069101181698512
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 69, 0.0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1.0: 42637945, 0.0: 30262055}
Accuracy: 0.6907206035665295
Confusion matrix:
[[ 8548448   832861]
 [21713607 41805084]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.28      0.91      0.43   9381309
           1       0.98      0.66      0.79  63518691

    accuracy                           0.69  72900000
   macro avg       0.63      0.78      0.61  72900000
weighted avg       0.89      0.69      0.74  72900000

ROC AUC: 0.7846876460514133
PR AUC: 0.9431528060149597
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 69, 0.0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1.0: 59610105, 0.0: 13289895}
Accuracy: 0.918138463648834
Confusion matrix:
[[ 8351749  1029560]
 [ 4938146 58580545]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.63      0.89      0.74   9381309
           1       0.98      0.92      0.95  63518691

    accuracy                           0.92  72900000
   macro avg       0.81      0.91      0.84  72900000
weighted avg       0.94      0.92      0.92  72900000

ROC AUC: 0.9062554629735704
PR AUC: 0.9740666115574971
---------- End MLPClassifier ----------

 ---------- SVC ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 69, 0.0: 131}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {1.0: 105753, 0.0: 72794247}
Accuracy: 0.12964235939643348
Confusion matrix:
[[ 9363242    18067]
 [63431005    87686]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.13      1.00      0.23   9381309
           1       0.83      0.00      0.00  63518691

    accuracy                           0.13  72900000
   macro avg       0.48      0.50      0.12  72900000
weighted avg       0.74      0.13      0.03  72900000

ROC AUC: 0.49972731245273044
PR AUC: 0.8712544410088244
---------- End SVC ----------



 ########## 300 ##########

 ---------- XGBClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 98, 0.0: 202}
y_pred {1.0: 56071395, 0.0: 16828605}
Accuracy: 0.8688821399176955
Confusion matrix:
[[ 8325711  1055598]
 [ 8502894 55015797]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.49      0.89      0.64   9381309
           1       0.98      0.87      0.92  63518691

    accuracy                           0.87  72900000
   macro avg       0.74      0.88      0.78  72900000
weighted avg       0.92      0.87      0.88  72900000

ROC AUC: 0.8768070812018169
PR AUC: 0.9664675018939006
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 98, 0.0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1.0: 62035704, 0.0: 10864296}
Accuracy: 0.9329091769547325
Confusion matrix:
[[ 7677342  1703967]
 [ 3186954 60331737]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.71      0.82      0.76   9381309
           1       0.97      0.95      0.96  63518691

    accuracy                           0.93  72900000
   macro avg       0.84      0.88      0.86  72900000
weighted avg       0.94      0.93      0.93  72900000

ROC AUC: 0.8840961345008749
PR AUC: 0.9674539267922637
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 98, 0.0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
y_pred {1.0: 61604756, 0.0: 11295244}
Accuracy: 0.9309881893004115
Confusion matrix:
[[ 7822796  1558513]
 [ 3472448 60046243]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.69      0.83      0.76   9381309
           1       0.97      0.95      0.96  63518691

    accuracy                           0.93  72900000
   macro avg       0.83      0.89      0.86  72900000
weighted avg       0.94      0.93      0.93  72900000

ROC AUC: 0.8896011413992975
PR AUC: 0.9690493461535252
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 98, 0.0: 202}
y_pred {0.0: 50389694, 1.0: 22510306}
Accuracy: 0.415507585733882
Confusion matrix:
[[ 8580753   800556]
 [41808941 21709750]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.17      0.91      0.29   9381309
           1       0.96      0.34      0.50  63518691

    accuracy                           0.42  72900000
   macro avg       0.57      0.63      0.40  72900000
weighted avg       0.86      0.42      0.48  72900000

ROC AUC: 0.628225005343491
PR AUC: 0.9031408322282983
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 98, 0.0: 202}
y_pred {1.0: 46972300, 0.0: 25927700}
Accuracy: 0.7689898353909465
Confusion matrix:
[[ 9234184   147125]
 [16693516 46825175]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.36      0.98      0.52   9381309
           1       1.00      0.74      0.85  63518691

    accuracy                           0.77  72900000
   macro avg       0.68      0.86      0.69  72900000
weighted avg       0.91      0.77      0.81  72900000

ROC AUC: 0.8607522813882418
PR AUC: 0.9638703387671547
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 98, 0.0: 202}
y_pred {1.0: 19755900, 0.0: 53144100}
Accuracy: 0.3927734293552812
Confusion matrix:
[[ 9129296   252013]
 [44014804 19503887]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.17      0.97      0.29   9381309
           1       0.99      0.31      0.47  63518691

    accuracy                           0.39  72900000
   macro avg       0.58      0.64      0.38  72900000
weighted avg       0.88      0.39      0.45  72900000

ROC AUC: 0.6400970683830742
PR AUC: 0.9069101181698512
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 98, 0.0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1.0: 37995179, 0.0: 34904821}
Accuracy: 0.6320502331961592
Confusion matrix:
[[ 8731296   650013]
 [26173525 37345166]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.25      0.93      0.39   9381309
           1       0.98      0.59      0.74  63518691

    accuracy                           0.63  72900000
   macro avg       0.62      0.76      0.57  72900000
weighted avg       0.89      0.63      0.69  72900000

ROC AUC: 0.7593258502566873
PR AUC: 0.9369147150520261
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 98, 0.0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1.0: 61642681, 0.0: 11257319}
Accuracy: 0.9432080932784637
Confusion matrix:
[[ 8249249  1132060]
 [ 3008070 60510621]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.73      0.88      0.80   9381309
           1       0.98      0.95      0.97  63518691

    accuracy                           0.94  72900000
   macro avg       0.86      0.92      0.88  72900000
weighted avg       0.95      0.94      0.95  72900000

ROC AUC: 0.9159854495008553
PR AUC: 0.9764105581484884
---------- End MLPClassifier ----------

 ---------- SVC ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 98, 0.0: 202}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {1.0: 187309, 0.0: 72712691}
Accuracy: 0.13043495198902608
Confusion matrix:
[[ 9351354    29955]
 [63361337   157354]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.13      1.00      0.23   9381309
           1       0.84      0.00      0.00  63518691

    accuracy                           0.13  72900000
   macro avg       0.48      0.50      0.12  72900000
weighted avg       0.75      0.13      0.03  72900000

ROC AUC: 0.4996421176816233
PR AUC: 0.8712352543565264
---------- End SVC ----------



 ########## 400 ##########

 ---------- XGBClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 132, 0.0: 268}
y_pred {1.0: 56490437, 0.0: 16409563}
Accuracy: 0.8778126200274349
Confusion matrix:
[[ 8441706   939603]
 [ 7967857 55550834]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.51      0.90      0.65   9381309
           1       0.98      0.87      0.93  63518691

    accuracy                           0.88  72900000
   macro avg       0.75      0.89      0.79  72900000
weighted avg       0.92      0.88      0.89  72900000

ROC AUC: 0.8872009708813576
PR AUC: 0.9693108103268361
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 132, 0.0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1.0: 61846304, 0.0: 11053696}
Accuracy: 0.9335706035665295
Confusion matrix:
[[ 7796151  1585158]
 [ 3257545 60261146]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.71      0.83      0.76   9381309
           1       0.97      0.95      0.96  63518691

    accuracy                           0.93  72900000
   macro avg       0.84      0.89      0.86  72900000
weighted avg       0.94      0.93      0.94  72900000

ROC AUC: 0.8898726820851446
PR AUC: 0.9690841498484397
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 132, 0.0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
y_pred {1.0: 60782169, 0.0: 12117831}
Accuracy: 0.9256680658436214
Confusion matrix:
[[ 8040171  1341138]
 [ 4077660 59441031]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.66      0.86      0.75   9381309
           1       0.98      0.94      0.96  63518691

    accuracy                           0.93  72900000
   macro avg       0.82      0.90      0.85  72900000
weighted avg       0.94      0.93      0.93  72900000

ROC AUC: 0.8964226315911531
PR AUC: 0.9710905634132007
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 132, 0.0: 268}
y_pred {0.0: 48931133, 1.0: 23968867}
Accuracy: 0.4342158847736626
Confusion matrix:
[[ 8533390   847919]
 [40397743 23120948]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.17      0.91      0.29   9381309
           1       0.96      0.36      0.53  63518691

    accuracy                           0.43  72900000
   macro avg       0.57      0.64      0.41  72900000
weighted avg       0.86      0.43      0.50  72900000

ROC AUC: 0.6368092029689889
PR AUC: 0.9052782385658639
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 132, 0.0: 268}
y_pred {1.0: 64077553, 0.0: 8822447}
Accuracy: 0.9303368449931413
Confusion matrix:
[[ 6562656  2818653]
 [ 2259791 61258900]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.74      0.70      0.72   9381309
           1       0.96      0.96      0.96  63518691

    accuracy                           0.93  72900000
   macro avg       0.85      0.83      0.84  72900000
weighted avg       0.93      0.93      0.93  72900000

ROC AUC: 0.8319845428339239
PR AUC: 0.9529985241401331
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 132, 0.0: 268}
y_pred {1.0: 19755900, 0.0: 53144100}
Accuracy: 0.3927734293552812
Confusion matrix:
[[ 9129296   252013]
 [44014804 19503887]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.17      0.97      0.29   9381309
           1       0.99      0.31      0.47  63518691

    accuracy                           0.39  72900000
   macro avg       0.58      0.64      0.38  72900000
weighted avg       0.88      0.39      0.45  72900000

ROC AUC: 0.6400970683830742
PR AUC: 0.9069101181698512
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 132, 0.0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1.0: 35949914, 0.0: 36950086}
Accuracy: 0.6072543072702332
Confusion matrix:
[[ 8850117   531192]
 [28099969 35418722]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.24      0.94      0.38   9381309
           1       0.99      0.56      0.71  63518691

    accuracy                           0.61  72900000
   macro avg       0.61      0.75      0.55  72900000
weighted avg       0.89      0.61      0.67  72900000

ROC AUC: 0.7504943216862312
PR AUC: 0.9348309276787905
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 132, 0.0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1.0: 61881767, 0.0: 11018233}
Accuracy: 0.953244060356653
Confusion matrix:
[[ 8495517   885792]
 [ 2522716 60995975]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.77      0.91      0.83   9381309
           1       0.99      0.96      0.97  63518691

    accuracy                           0.95  72900000
   macro avg       0.88      0.93      0.90  72900000
weighted avg       0.96      0.95      0.95  72900000

ROC AUC: 0.9329314703952749
PR AUC: 0.9811432792815026
---------- End MLPClassifier ----------

 ---------- SVC ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 132, 0.0: 268}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {1.0: 374429, 0.0: 72525571}
Accuracy: 0.13231744855967079
Confusion matrix:
[[ 9326411    54898]
 [63199160   319531]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.13      0.99      0.23   9381309
           1       0.85      0.01      0.01  63518691

    accuracy                           0.13  72900000
   macro avg       0.49      0.50      0.12  72900000
weighted avg       0.76      0.13      0.04  72900000

ROC AUC: 0.49958932746722373
PR AUC: 0.871222434046437
---------- End SVC ----------



 ########## 500 ##########

 ---------- XGBClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 167, 0.0: 333}
y_pred {1.0: 56439359, 0.0: 16460641}
Accuracy: 0.8870838957475995
Confusion matrix:
[[ 8805183   576126]
 [ 7655458 55863233]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.53      0.94      0.68   9381309
           1       0.99      0.88      0.93  63518691

    accuracy                           0.89  72900000
   macro avg       0.76      0.91      0.81  72900000
weighted avg       0.93      0.89      0.90  72900000

ROC AUC: 0.9090324848440496
PR AUC: 0.9755126297469532
---------- End XGBClassifier ----------

 ---------- LogisticRegression ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 167, 0.0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
y_pred {1.0: 62436026, 0.0: 10463974}
Accuracy: 0.9365425651577504
Confusion matrix:
[[ 7609618  1771691]
 [ 2854356 60664335]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.73      0.81      0.77   9381309
           1       0.97      0.96      0.96  63518691

    accuracy                           0.94  72900000
   macro avg       0.85      0.88      0.87  72900000
weighted avg       0.94      0.94      0.94  72900000

ROC AUC: 0.8831047281531679
PR AUC: 0.9671161851766593
---------- End LogisticRegression ----------

 ---------- LinearSVC ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 167, 0.0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
y_pred {1.0: 64690752, 0.0: 8209248}
Accuracy: 0.943163525377229
Confusion matrix:
[[ 6723589  2657720]
 [ 1485659 62033032]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.82      0.72      0.76   9381309
           1       0.96      0.98      0.97  63518691

    accuracy                           0.94  72900000
   macro avg       0.89      0.85      0.87  72900000
weighted avg       0.94      0.94      0.94  72900000

ROC AUC: 0.8466555978632997
PR AUC: 0.9568675430141627
---------- End LinearSVC ----------

 ---------- KNeighborsClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 167, 0.0: 333}
y_pred {0.0: 47269663, 1.0: 25630337}
Accuracy: 0.457579890260631
Confusion matrix:
[[ 8554273   827036]
 [38715390 24803301]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.18      0.91      0.30   9381309
           1       0.97      0.39      0.56  63518691

    accuracy                           0.46  72900000
   macro avg       0.57      0.65      0.43  72900000
weighted avg       0.87      0.46      0.52  72900000

ROC AUC: 0.6511651899498029
PR AUC: 0.908963317665886
---------- End KNeighborsClassifier ----------

 ---------- SGDClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 167, 0.0: 333}
y_pred {1.0: 61348373, 0.0: 11551627}
Accuracy: 0.9171923182441701
Confusion matrix:
[[ 7448128  1933181]
 [ 4103499 59415192]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.64      0.79      0.71   9381309
           1       0.97      0.94      0.95  63518691

    accuracy                           0.92  72900000
   macro avg       0.81      0.86      0.83  72900000
weighted avg       0.93      0.92      0.92  72900000

ROC AUC: 0.8646648429442287
PR AUC: 0.9622106193797921
---------- End SGDClassifier ----------

 ---------- BernoulliNB ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 167, 0.0: 333}
y_pred {1.0: 19755900, 0.0: 53144100}
Accuracy: 0.3927734293552812
Confusion matrix:
[[ 9129296   252013]
 [44014804 19503887]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.17      0.97      0.29   9381309
           1       0.99      0.31      0.47  63518691

    accuracy                           0.39  72900000
   macro avg       0.58      0.64      0.38  72900000
weighted avg       0.88      0.39      0.45  72900000

ROC AUC: 0.6400970683830742
PR AUC: 0.9069101181698512
---------- End BernoulliNB ----------

 ---------- RandomForestClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 167, 0.0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
y_pred {1.0: 36623752, 0.0: 36276248}
Accuracy: 0.6160888751714677
Confusion matrix:
[[ 8835218   546091]
 [27441030 36077661]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.24      0.94      0.39   9381309
           1       0.99      0.57      0.72  63518691

    accuracy                           0.62  72900000
   macro avg       0.61      0.75      0.55  72900000
weighted avg       0.89      0.62      0.68  72900000

ROC AUC: 0.7548872120332266
PR AUC: 0.9359359877062897
---------- End RandomForestClassifier ----------

 ---------- MLPClassifier ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 167, 0.0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:566: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
y_pred {1.0: 62010752, 0.0: 10889248}
Accuracy: 0.9515916460905349
Confusion matrix:
[[ 8370794  1010515]
 [ 2518454 61000237]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.77      0.89      0.83   9381309
           1       0.98      0.96      0.97  63518691

    accuracy                           0.95  72900000
   macro avg       0.88      0.93      0.90  72900000
weighted avg       0.96      0.95      0.95  72900000

ROC AUC: 0.9263175996896638
PR AUC: 0.9792479813713207
---------- End MLPClassifier ----------

 ---------- SVC ----------
(72900000, 8) (72900000,)
y_test {1: 63518691, 0: 9381309} y_train {1.0: 167, 0.0: 333}
/home/mihailselezniov/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
y_pred {1.0: 459244, 0.0: 72440756}
Accuracy: 0.13329466392318245
Confusion matrix:
[[ 9319623    61686]
 [63121133   397558]]
Precision, recall and f1-score:
              precision    recall  f1-score   support

           0       0.13      0.99      0.23   9381309
           1       0.87      0.01      0.01  63518691

    accuracy                           0.13  72900000
   macro avg       0.50      0.50      0.12  72900000
weighted avg       0.77      0.13      0.04  72900000

ROC AUC: 0.4998417493038752
PR AUC: 0.8712773748623627
---------- End SVC ----------
roc_metrics
[[0.8571, 0.8713, 0.8734, 0.615, 0.9052, 0.6401, 0.6992, 0.894, 0.5], [0.8695, 0.8745, 0.8727, 0.6274, 0.8238, 0.6401, 0.7847, 0.9063, 0.4997], [0.8768, 0.8841, 0.8896, 0.6282, 0.8608, 0.6401, 0.7593, 0.916, 0.4996], [0.8872, 0.8899, 0.8964, 0.6368, 0.832, 0.6401, 0.7505, 0.9329, 0.4996], [0.909, 0.8831, 0.8467, 0.6512, 0.8647, 0.6401, 0.7549, 0.9263, 0.4998]]
pr_metrics
[[0.9622, 0.964, 0.9647, 0.9002, 0.974, 0.9069, 0.922, 0.9708, 0.8713], [0.9647, 0.9648, 0.9642, 0.903, 0.9507, 0.9069, 0.9432, 0.9741, 0.8713], [0.9665, 0.9675, 0.969, 0.9031, 0.9639, 0.9069, 0.9369, 0.9764, 0.8712], [0.9693, 0.9691, 0.9711, 0.9053, 0.953, 0.9069, 0.9348, 0.9811, 0.8712], [0.9755, 0.9671, 0.9569, 0.909, 0.9622, 0.9069, 0.9359, 0.9792, 0.8713]]
f1_metrics
[[0.5397, 0.7267, 0.7186, 0.2784, 0.7129, 0.292, 0.3325, 0.6951, 0.228], [0.6089, 0.75, 0.7505, 0.2866, 0.7286, 0.292, 0.4313, 0.7368, 0.2279], [0.6353, 0.7584, 0.7567, 0.2871, 0.523, 0.292, 0.3943, 0.7994, 0.2278], [0.6546, 0.763, 0.748, 0.2927, 0.721, 0.292, 0.382, 0.8329, 0.2277], [0.6815, 0.7669, 0.7645, 0.302, 0.7116, 0.292, 0.387, 0.8259, 0.2278]]
'''









