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
#from sklearn.externals import joblib
from xgboost import XGBClassifier, XGBRegressor
import collections


ex_name = '1_rod_98_500.txt'
folder_name = 'ml_threads/'
f_name = folder_name + ex_name

'''
Y = []
for file_index in range(1, 101):
    with open('Y/{}.txt'.format(file_index), 'r') as f:
        y_str = f.readlines()
    y = [int(i) for i in y_str[0]]
    Y.extend(y)
'''
with open('../18/Y/Y.txt', 'r') as f:
    y_str = f.readlines()
y = [int(i) for i in y_str[0]]
#print(len(y))
y_test = np.array(y)

def cartesian_coord(*arrays):
    grid = np.meshgrid(*arrays)        
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points

a = np.arange(100)
x_test = cartesian_coord(*4*[a])
xt = np.array_split(x_test, 10)

#print(x_test[0])
#print(x_test[1])
#print(x_test[-1])
#print(len(x_test))



with open(f_name, 'r') as f:
    threads = f.readlines()

roc_metrics, pr_metrics, f1_metrics = [], [], []
roc_metric, pr_metric, f1_metric = [], [], []

for cut in [100, 200, 300, 400, 500]:
    #cut = 500
    print('\n\n\n', '#'*10, cut, '#'*10)

    x_train, y_train = [], []
    for t in threads[:cut]:
        tr = list(map(float, t.replace('\n', '').split(',')))
        x_train.append(tr[:-1])
        y_train.append(tr[-1])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = x_train + 0.5
    #print(x_train)
    #break


    def fit_model(model):
        global roc_metric
        global pr_metric
        global f1_metric
        print('\n', '-'*10, model.__class__.__name__, '-'*10)
        print(x_test.shape, y_test.shape)
        print('y_test', dict(collections.Counter(y_test)), 'y_train', dict(collections.Counter(y_train)))
        # fit model on training data
        model.fit(x_train, y_train)

        y_pred = model.predict(xt[0])
        for i in range(1, len(xt)):
            y_pred = np.concatenate((y_pred, model.predict(xt[i])))


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
    fit_model(MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(50, 50, 50), max_iter=100000, random_state=42))
    fit_model(MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(70, 70, 70), max_iter=100000, random_state=42))
    fit_model(MLPClassifier(activation='tanh', solver='adam', hidden_layer_sizes=(90, 90, 90), max_iter=100000, random_state=42))
    fit_model(MLPClassifier(activation='tanh', solver='adam', hidden_layer_sizes=(70, 70, 70), max_iter=100000, random_state=42))

    roc_metrics.append(roc_metric[:])
    pr_metrics.append(pr_metric[:])
    f1_metrics.append(f1_metric[:])

    #break


print('roc_metrics')
print(roc_metrics)
print('pr_metrics')
print(pr_metrics)
print('f1_metrics')
print(f1_metrics)
print()

