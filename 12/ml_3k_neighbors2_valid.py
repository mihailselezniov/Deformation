import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import collections
import math
import sys
import time
import pickle
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

f_name = 'ml_threads/6_1.txt'
n_cors = 6


X = joblib.load('border_vars/X.j')
print('X')
Y = joblib.load('border_vars/Y.j')
print('Y')

#print(sys.getsizeof(way_dict))
print(X.shape, Y.shape)
print('all', dict(collections.Counter(Y)))


def fit_model(model):
    #print('-'*10, model.__class__.__name__, '-'*10)
    model.fit(x_train, y_train)
    return model.predict(X)
    '''
    try:
        return model.predict(X)
    except:
        #print('Error')
        pass
    return [0]*len(X)
    '''


#x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.0001, random_state=42)
while 1:
    with open(f_name, 'r') as f:
        threads = f.readlines()

    x_train, y_train = [], []
    for t in threads:
        tr = list(map(int, t.replace('\n', '').split(',')))
        x_train.append(tr[:-1])
        y_train.append(tr[-1])
    x_train, y_train = np.array(x_train), np.array(y_train)

    print('#', len(threads), 'y_train', dict(collections.Counter(y_train)))

    y_preds = []
    y_preds.append(fit_model(XGBClassifier(random_state=42, n_jobs=n_cors)))
    y_preds.append(fit_model(LogisticRegression()))
    y_preds.append(fit_model(LinearSVC(random_state=42, tol=1e-5)))
    #y_preds.append(fit_model(KNeighborsClassifier(n_neighbors=5)))
    y_preds.append(fit_model(SGDClassifier(random_state=42, n_jobs=n_cors)))
    y_preds.append(fit_model(BernoulliNB()))
    y_preds.append(fit_model(RandomForestClassifier(random_state=42, n_jobs=n_cors)))
    #y_preds.append(fit_model(MLPClassifier()))
    y_preds.append(fit_model(SVC(random_state=42)))# Radial basis function kernel


    y_pred = [sum(i) for i in zip(*y_preds)]
    d_pred = dict(collections.Counter(y_pred))
    min_sum = min(list(d_pred))
    print('y_pred', d_pred, min_sum)

    threads_arr = {
        1: [list(x_train[i]) for i in range(len(x_train)) if y_train[i] == 1],
        0: [list(x_train[i]) for i in range(len(x_train)) if y_train[i] == 0]
    }

    result_dis, result_dis_1, result_id = 0, 0, 0
    for tid, i in enumerate(y_pred):
        if i != min_sum:
            continue
        l0 = list(X[tid])
        dis = min([sum(map(lambda x: (x[0] - x[1])**2, zip(thread, l0))) for thread in threads_arr[0]])
        if dis == result_dis:
            dis1 = min([sum(map(lambda x: (x[0] - x[1])**2, zip(thread, l0))) for thread in threads_arr[1]])
            if dis1 > result_dis_1:
                result_dis_1 = dis1
                result_dis = dis
                result_id = tid
        if dis > result_dis:
            result_dis_1 = min([sum(map(lambda x: (x[0] - x[1])**2, zip(thread, l0))) for thread in threads_arr[1]])
            result_dis = dis
            result_id = tid

    print(result_dis, result_dis_1, result_id, Y[result_id])
    row_data = list(X[result_id]) + [int(Y[result_id])]
    row_thread = ','.join(list(map(lambda x: str(int(x)), row_data)))
    print(row_thread)
    break
    with open(f_name, 'a') as f:
        f.write(row_thread + '\n')


'''
(72900000, 8) (72900000,)
all {1: 63518691, 0: 9381309}
# 2 y_train {1: 1, 0: 1}
y_pred {6: 1008, 5: 489108, 4: 8584869, 3: 14710773, 2: 26494604, 1: 20450374, 0: 2169264} 0
225 247 7217289 0
0,9,9,0,1,3,1,9,0
'''
