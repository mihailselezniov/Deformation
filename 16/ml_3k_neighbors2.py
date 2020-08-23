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
#import joblib
from sklearn.externals import joblib
import collections
import math
import sys
import time
import pickle
import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


'''
stream = os.popen('go run dis_calc.go')
output = stream.read()
print(output)
'''
ex_name = '7_1.txt'
folder_name = 'ml_threads/'
folder_name_y = 'ml_y_pred/'
f_name = folder_name + ex_name
f_y_pred = folder_name_y + 'y{}_' + ex_name
n_cors = 4
Y_step = 729000#0
Y_step = 7290000

X = joblib.load('../12/border_vars/X.j')
print('X')
Y = joblib.load('../12/border_vars/Y.j')
print('Y')

#print(sys.getsizeof(way_dict))
print(X.shape, Y.shape)
print('all', dict(collections.Counter(Y)))


def fit_model(model):
    #print('-'*10, model.__class__.__name__, '-'*10)
    model.fit(x_train, y_train)
    return model.predict(X)


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
    #y_preds.append(fit_model(XGBClassifier(random_state=42, n_jobs=n_cors)))
    y_preds.append(fit_model(LogisticRegression(random_state=42)))
    y_preds.append(fit_model(LinearSVC(random_state=42, tol=1e-5)))
    #y_preds.append(fit_model(KNeighborsClassifier(n_neighbors=5)))
    y_preds.append(fit_model(SGDClassifier(random_state=42, n_jobs=n_cors)))
    #y_preds.append(fit_model(BernoulliNB()))
    #y_preds.append(fit_model(RandomForestClassifier(random_state=42, n_jobs=n_cors)))
    
    """
    print('Start MLP')
    try:
        print(fit_model(MLPClassifier(max_iter=100000)))
    except:
        print('MLP faile')
    print('End MLP')
    break
    """

    y_preds.append(fit_model(MLPClassifier(random_state=42, max_iter=100000)))
    #y_preds.append(fit_model(SVC(random_state=42)))# Radial basis function kernel


    y_pred = [sum(i) for i in zip(*y_preds)]
    d_pred = dict(collections.Counter(y_pred))
    min_sum = min(list(d_pred))
    print('y_pred', d_pred, min_sum)


    rows_y = []
    for a0 in range(10):
        all_num = 0
        current_state = 0
        min_iy = a0 * Y_step
        max_iy = (a0 + 1) * Y_step
        rows_y.append([])
        for iy in range(min_iy, max_iy):
            new_state = 1
            if y_pred[iy] == min_sum:
                new_state = 0
            if new_state != current_state:
                rows_y[a0].append('{}\n'.format(all_num))
                current_state = new_state
                all_num = 1
            else:
                all_num += 1
        rows_y[a0].append('{}\n'.format(all_num))

    for a0 in range(10):
        with open(f_y_pred.format(a0), 'w') as f:
            f.write(''.join(rows_y[a0]))

    #print('Start Go')
    stream = os.popen('go run dis_calc.go')
    #print('End Go')
    output = stream.read()
    print(output[:-1])
    result_id = int(output.split()[2])
    row_data = list(X[result_id]) + [int(Y[result_id])]
    row_thread = ','.join(list(map(lambda x: str(int(x)), row_data)))
    print(row_thread)
    with open(f_name, 'a') as f:
        f.write(row_thread + '\n')
    #break

'''







'''
