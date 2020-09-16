import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.neural_network import MLPClassifier
import joblib
import collections
import math
import sys
import time
import pickle
import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


ex_name = '11_1_mlp_2.txt'
folder_name = 'ml_threads/'
folder_name_y = 'ml_y_pred/'
f_name = folder_name + ex_name
f_y_pred = folder_name_y + 'y{}_' + ex_name
n_cors = 4
Y_step = 729000#0
Y_step = 7290000

X = joblib.load('border_vars/X.j')
print('X')
Y = joblib.load('border_vars/Y.j')
print('Y')

print(X.shape, Y.shape)
print('all', dict(collections.Counter(Y)))


def fit_model(model):
    model.fit(x_train, y_train)

    pred_p = model.predict_proba(X[:20000000])
    pred_p = np.concatenate((pred_p, model.predict_proba(X[20000000:40000000])))
    pred_p = np.concatenate((pred_p, model.predict_proba(X[40000000:60000000])))
    pred_p = np.concatenate((pred_p, model.predict_proba(X[60000000:])))
    return pred_p



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


    y_pred_p = fit_model(MLPClassifier(activation='tanh', solver='adam', hidden_layer_sizes=(90, 90, 90), max_iter=100000, random_state=42))


    min_count = 0
    rows_y = []
    for a0 in range(10):
        all_num = 0
        current_state = 0
        min_iy = a0 * Y_step
        max_iy = (a0 + 1) * Y_step
        rows_y.append([])
        for iy in range(min_iy, max_iy):
            new_state = 1
            if 0.8 < y_pred_p[iy][0] < 0.9:
                new_state = 0
                min_count += 1
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

    print('{}: {}'.format('0.8 < y_pred_p[iy][0] < 0.9', min_count))
    stream = os.popen('go run dis_calc.go')
    output = stream.read()
    print(output[:-1])
    result_id = int(output.split()[2])
    row_data = list(X[result_id]) + [int(Y[result_id])]
    row_thread = ','.join(list(map(lambda x: str(int(x)), row_data)))
    print(row_thread)
    with open(f_name, 'a') as f:
        f.write(row_thread + '\n')
