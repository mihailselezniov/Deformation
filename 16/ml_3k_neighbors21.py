'''
Xr = [[0, 4, 1, 0, 8, 1, 1, 8], [0, 4, 3, 4, 7, 1, 1, 6], [0, 4, 3, 7, 4, 1, 1, 4], [0, 5, 9, 6, 8, 3, 1, 6], [0, 6, 7, 1, 2, 1, 9, 8], [0, 6, 8, 7, 2, 1, 2, 2], [0, 9, 7, 9, 3, 1, 5, 4], [0, 9, 8, 0, 9, 1, 5, 7], [1, 3, 4, 5, 1, 3, 1, 3], [1, 3, 7, 0, 2, 1, 3, 5], [1, 3, 8, 5, 1, 7, 1, 5], [1, 4, 9, 0, 1, 1, 5, 4], [1, 5, 9, 9, 1, 7, 2, 4], [1, 6, 5, 3, 7, 1, 2, 4], [1, 6, 5, 8, 3, 9, 1, 5], [1, 6, 7, 6, 8, 1, 2, 3], [1, 7, 8, 0, 4, 2, 5, 9], [1, 8, 1, 4, 1, 4, 1, 2], [1, 9, 5, 2, 5, 1, 2, 2], [1, 9, 6, 4, 7, 1, 7, 9], [2, 3, 5, 9, 2, 8, 1, 9], [2, 3, 8, 7, 3, 3, 1, 4], [2, 3, 9, 2, 1, 5, 1, 3], [2, 3, 9, 5, 5, 1, 4, 9], [2, 4, 3, 7, 5, 3, 1, 9], [2, 4, 4, 6, 1, 2, 3, 7], [2, 4, 5, 6, 4, 4, 1, 7], [2, 5, 0, 5, 2, 1, 1, 3], [2, 5, 5, 6, 6, 5, 1, 9], [2, 5, 8, 7, 4, 3, 1, 2], [2, 6, 1, 7, 8, 2, 1, 4], [2, 7, 6, 6, 1, 3, 3, 4], [2, 8, 3, 0, 5, 1, 3, 4], [2, 8, 4, 5, 4, 1, 6, 6], [2, 8, 9, 8, 8, 7, 1, 6], [2, 9, 1, 6, 8, 1, 3, 6], [3, 4, 2, 7, 7, 2, 1, 4], [3, 4, 8, 7, 3, 2, 4, 8], [3, 4, 9, 2, 4, 1, 6, 5], [3, 4, 9, 8, 8, 1, 5, 7], [3, 5, 4, 9, 7, 1, 4, 9], [3, 5, 9, 0, 2, 4, 3, 6], [3, 6, 4, 1, 1, 3, 2, 3], [3, 6, 5, 5, 4, 5, 2, 9], [3, 6, 8, 7, 2, 3, 4, 7], [3, 7, 0, 3, 5, 3, 1, 7], [3, 7, 1, 2, 1, 8, 1, 8], [3, 7, 3, 9, 8, 1, 4, 6], [3, 7, 8, 8, 3, 2, 7, 6], [3, 8, 0, 9, 9, 3, 1, 6], [3, 8, 3, 8, 4, 1, 7, 7], [3, 8, 4, 0, 1, 1, 4, 3], [3, 9, 0, 2, 6, 3, 1, 6], [3, 9, 0, 5, 2, 5, 1, 5], [3, 9, 4, 4, 1, 3, 2, 2], [4, 2, 5, 3, 2, 2, 1, 7], [4, 3, 2, 6, 5, 2, 1, 5], [4, 4, 3, 9, 6, 1, 3, 7], [4, 5, 7, 9, 5, 3, 3, 6], [4, 6, 5, 4, 3, 6, 2, 8], [4, 7, 3, 3, 9, 1, 5, 7], [4, 7, 4, 6, 1, 2, 4, 5], [4, 8, 3, 4, 1, 2, 5, 9], [4, 8, 3, 7, 1, 1, 1, 1], [4, 8, 6, 3, 1, 1, 7, 2], [5, 2, 7, 1, 5, 1, 2, 5], [5, 3, 9, 4, 5, 2, 2, 5], [5, 4, 6, 1, 8, 1, 1, 1], [5, 5, 0, 0, 5, 1, 1, 3], [5, 5, 5, 1, 4, 1, 6, 8], [5, 5, 5, 4, 2, 3, 2, 3], [5, 6, 5, 4, 6, 4, 1, 2], [5, 8, 2, 7, 4, 2, 3, 5], [5, 8, 5, 0, 8, 3, 2, 3], [5, 8, 5, 6, 3, 2, 7, 9], [5, 8, 6, 4, 2, 4, 3, 3], [5, 8, 6, 7, 6, 1, 9, 5], [5, 9, 0, 7, 2, 6, 1, 5], [5, 9, 7, 5, 7, 6, 1, 2], [6, 3, 1, 1, 7, 2, 1, 9], [6, 4, 7, 6, 5, 3, 2, 5], [6, 4, 8, 8, 9, 1, 5, 5], [6, 7, 4, 5, 3, 2, 4, 7], [6, 7, 7, 8, 3, 2, 5, 3], [6, 7, 8, 9, 9, 2, 7, 8], [6, 8, 5, 2, 8, 2, 5, 6], [6, 8, 8, 2, 6, 2, 5, 3], [6, 9, 0, 8, 4, 1, 2, 3], [6, 9, 3, 1, 8, 2, 3, 3], [7, 2, 8, 3, 5, 1, 2, 7], [7, 4, 7, 8, 5, 2, 3, 5], [7, 5, 2, 8, 4, 4, 1, 5], [7, 6, 4, 1, 6, 5, 2, 9], [7, 6, 4, 7, 8, 1, 6, 7], [7, 7, 1, 5, 4, 4, 1, 4], [7, 7, 4, 8, 1, 1, 5, 7], [7, 7, 5, 8, 4, 1, 1, 1], [7, 9, 4, 5, 6, 1, 6, 3], [8, 4, 3, 9, 3, 4, 1, 7], [8, 4, 6, 8, 2, 9, 1, 8], [8, 5, 2, 0, 7, 1, 1, 1], [8, 5, 3, 8, 8, 6, 1, 8], [8, 5, 4, 7, 5, 1, 4, 9], [8, 5, 7, 8, 9, 1, 7, 7], [8, 6, 4, 9, 7, 3, 3, 6], [8, 7, 8, 8, 7, 1, 9, 5], [8, 7, 9, 0, 6, 6, 1, 2], [8, 8, 4, 1, 1, 6, 2, 5], [8, 8, 9, 4, 5, 4, 5, 9], [8, 9, 2, 0, 8, 1, 4, 3], [8, 9, 5, 3, 3, 5, 3, 5], [8, 9, 6, 4, 2, 1, 9, 6], [9, 3, 9, 7, 9, 4, 1, 6], [9, 6, 2, 5, 8, 3, 2, 6], [9, 7, 0, 7, 7, 2, 1, 3], [9, 7, 8, 1, 4, 1, 8, 4], [9, 7, 9, 2, 2, 3, 5, 5], [9, 7, 9, 2, 8, 2, 1, 1], [9, 8, 6, 8, 5, 7, 1, 2], [9, 9, 5, 4, 9, 4, 4, 8], [9, 9, 8, 2, 4, 4, 5, 7]]
Xr_dis = [[sum(sum(map((lambda x, y: (x-y)**2), i, j))**0.5 for j in Xr), Xr[ix]] for ix, i in enumerate(Xr)]
for i in sorted(Xr_dis)[:50]:
    print(i)
#print(sum(map((lambda x, y: (x-y)**2), i, j))**0.5)
'''

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


ex_name = '12_1_mlp_2.txt'
folder_name = 'ml_threads/'
folder_name_y = 'ml_y_pred/'
f_name = folder_name + ex_name
f_y_pred = folder_name_y + 'y{}_' + ex_name
n_cors = 4
Y_step = 729000#0
Y_step = 7290000

X = joblib.load('border_vars/X.j')
xt0 = X[:20000000]
xt1 = X[20000000:40000000]
xt2 = X[40000000:60000000]
xt3 = X[60000000:]
print('X', list(X[0, :]))

Y = joblib.load('border_vars/Y.j')
print('Y')

print(X.shape, Y.shape)
print('all', dict(collections.Counter(Y)))


def fit_model(model):
    model.fit(x_train, y_train)
    pred_p = model.predict_proba(xt0)
    pred_p = np.concatenate((pred_p, model.predict_proba(xt1)))
    pred_p = np.concatenate((pred_p, model.predict_proba(xt2)))
    pred_p = np.concatenate((pred_p, model.predict_proba(xt3)))
    return pred_p


step_pred = 10000


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


    y_pred_p = fit_model(MLPClassifier(max_iter=100000, random_state=42))


    y_pred_p0 = y_pred_p[:, 0]

    #step_pred = 10000
    change_step = 100
    #0.00100001
    level = 0.75

    r = np.logical_and(level < y_pred_p0, y_pred_p0 < (level + 0.00000001*step_pred))
    nz = np.count_nonzero(r)
    print(nz)
    flag_nz_up = 1
    flag_nz_count = 0
    while not (1000 <= nz <= 3000):
        if 1000 > nz:
            if not flag_nz_up:
                change_step = int(change_step / 2)
            else:
                change_step = change_step * 2

            if not flag_nz_up:
                change_step = 100
            flag_nz_up = 1

            step_pred += change_step

        elif 3000 < nz:
            if flag_nz_up:
                change_step = int(change_step / 2)
            else:
                change_step = change_step * 2

            if flag_nz_up:
                change_step = 100
            flag_nz_up = 0

            step_pred -= change_step

        r = np.logical_and(level < y_pred_p0, y_pred_p0 < (level + 0.00000001*step_pred))
        nz = np.count_nonzero(r)
        print(nz, step_pred, change_step)

    #break


    #r = np.logical_and(0.85 < y_pred_p0, y_pred_p0 < 0.8501)

    Xri = [[X[ix], ix] for ix, val in enumerate(r) if val]

    #Xr = X[r]
    #print(len(Xr))
    #Xri = Xri.tolist()
    Xr_dis = [[sum(sum(map((lambda x, y: (x-y)**2), i[0], j[0]))**0.5 for j in Xri), Xri[ix]] for ix, i in enumerate(Xri)]
    #for i in sorted(Xr_dis)[:50]:
    #    print(i)

    result_id = sorted(Xr_dis)[0][-1][-1]
    print(result_id)
    row_data = list(X[result_id]) + [int(Y[result_id])]
    #print(result_id, row_data)

    '''
    break

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
            if 0.85 < y_pred_p[iy][0] < 0.850005:
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

    print('{}: {}'.format('0.85 < y_pred_p[iy][0] < 0.8500001', min_count))
    #break
    stream = os.popen('go run dis_calc.go')
    output = stream.read()
    print(output[:-1])
    result_id = int(output.split()[2])
    row_data = list(X[result_id]) + [int(Y[result_id])]
    '''
    row_thread = ','.join(list(map(lambda x: str(int(x)), row_data)))
    print(row_thread)
    with open(f_name, 'a') as f:
        f.write(row_thread + '\n')

'''
[[0, 4, 1, 0, 8, 1, 1, 8], [0, 4, 3, 4, 7, 1, 1, 6], [0, 4, 3, 7, 4, 1, 1, 4], [0, 5, 9, 6, 8, 3, 1, 6], [0, 6, 7, 1, 2, 1, 9, 8], [0, 6, 8, 7, 2, 1, 2, 2], [0, 9, 7, 9, 3, 1, 5, 4], [0, 9, 8, 0, 9, 1, 5, 7], [1, 3, 4, 5, 1, 3, 1, 3], [1, 3, 7, 0, 2, 1, 3, 5], [1, 3, 8, 5, 1, 7, 1, 5], [1, 4, 9, 0, 1, 1, 5, 4], [1, 5, 9, 9, 1, 7, 2, 4], [1, 6, 5, 3, 7, 1, 2, 4], [1, 6, 5, 8, 3, 9, 1, 5], [1, 6, 7, 6, 8, 1, 2, 3], [1, 7, 8, 0, 4, 2, 5, 9], [1, 8, 1, 4, 1, 4, 1, 2], [1, 9, 5, 2, 5, 1, 2, 2], [1, 9, 6, 4, 7, 1, 7, 9], [2, 3, 5, 9, 2, 8, 1, 9], [2, 3, 8, 7, 3, 3, 1, 4], [2, 3, 9, 2, 1, 5, 1, 3], [2, 3, 9, 5, 5, 1, 4, 9], [2, 4, 3, 7, 5, 3, 1, 9], [2, 4, 4, 6, 1, 2, 3, 7], [2, 4, 5, 6, 4, 4, 1, 7], [2, 5, 0, 5, 2, 1, 1, 3], [2, 5, 5, 6, 6, 5, 1, 9], [2, 5, 8, 7, 4, 3, 1, 2], [2, 6, 1, 7, 8, 2, 1, 4], [2, 7, 6, 6, 1, 3, 3, 4], [2, 8, 3, 0, 5, 1, 3, 4], [2, 8, 4, 5, 4, 1, 6, 6], [2, 8, 9, 8, 8, 7, 1, 6], [2, 9, 1, 6, 8, 1, 3, 6], [3, 4, 2, 7, 7, 2, 1, 4], [3, 4, 8, 7, 3, 2, 4, 8], [3, 4, 9, 2, 4, 1, 6, 5], [3, 4, 9, 8, 8, 1, 5, 7], [3, 5, 4, 9, 7, 1, 4, 9], [3, 5, 9, 0, 2, 4, 3, 6], [3, 6, 4, 1, 1, 3, 2, 3], [3, 6, 5, 5, 4, 5, 2, 9], [3, 6, 8, 7, 2, 3, 4, 7], [3, 7, 0, 3, 5, 3, 1, 7], [3, 7, 1, 2, 1, 8, 1, 8], [3, 7, 3, 9, 8, 1, 4, 6], [3, 7, 8, 8, 3, 2, 7, 6], [3, 8, 0, 9, 9, 3, 1, 6], [3, 8, 3, 8, 4, 1, 7, 7], [3, 8, 4, 0, 1, 1, 4, 3], [3, 9, 0, 2, 6, 3, 1, 6], [3, 9, 0, 5, 2, 5, 1, 5], [3, 9, 4, 4, 1, 3, 2, 2], [4, 2, 5, 3, 2, 2, 1, 7], [4, 3, 2, 6, 5, 2, 1, 5], [4, 4, 3, 9, 6, 1, 3, 7], [4, 5, 7, 9, 5, 3, 3, 6], [4, 6, 5, 4, 3, 6, 2, 8], [4, 7, 3, 3, 9, 1, 5, 7], [4, 7, 4, 6, 1, 2, 4, 5], [4, 8, 3, 4, 1, 2, 5, 9], [4, 8, 3, 7, 1, 1, 1, 1], [4, 8, 6, 3, 1, 1, 7, 2], [5, 2, 7, 1, 5, 1, 2, 5], [5, 3, 9, 4, 5, 2, 2, 5], [5, 4, 6, 1, 8, 1, 1, 1], [5, 5, 0, 0, 5, 1, 1, 3], [5, 5, 5, 1, 4, 1, 6, 8], [5, 5, 5, 4, 2, 3, 2, 3], [5, 6, 5, 4, 6, 4, 1, 2], [5, 8, 2, 7, 4, 2, 3, 5], [5, 8, 5, 0, 8, 3, 2, 3], [5, 8, 5, 6, 3, 2, 7, 9], [5, 8, 6, 4, 2, 4, 3, 3], [5, 8, 6, 7, 6, 1, 9, 5], [5, 9, 0, 7, 2, 6, 1, 5], [5, 9, 7, 5, 7, 6, 1, 2], [6, 3, 1, 1, 7, 2, 1, 9], [6, 4, 7, 6, 5, 3, 2, 5], [6, 4, 8, 8, 9, 1, 5, 5], [6, 7, 4, 5, 3, 2, 4, 7], [6, 7, 7, 8, 3, 2, 5, 3], [6, 7, 8, 9, 9, 2, 7, 8], [6, 8, 5, 2, 8, 2, 5, 6], [6, 8, 8, 2, 6, 2, 5, 3], [6, 9, 0, 8, 4, 1, 2, 3], [6, 9, 3, 1, 8, 2, 3, 3], [7, 2, 8, 3, 5, 1, 2, 7], [7, 4, 7, 8, 5, 2, 3, 5], [7, 5, 2, 8, 4, 4, 1, 5], [7, 6, 4, 1, 6, 5, 2, 9], [7, 6, 4, 7, 8, 1, 6, 7], [7, 7, 1, 5, 4, 4, 1, 4], [7, 7, 4, 8, 1, 1, 5, 7], [7, 7, 5, 8, 4, 1, 1, 1], [7, 9, 4, 5, 6, 1, 6, 3], [8, 4, 3, 9, 3, 4, 1, 7], [8, 4, 6, 8, 2, 9, 1, 8], [8, 5, 2, 0, 7, 1, 1, 1], [8, 5, 3, 8, 8, 6, 1, 8], [8, 5, 4, 7, 5, 1, 4, 9], [8, 5, 7, 8, 9, 1, 7, 7], [8, 6, 4, 9, 7, 3, 3, 6], [8, 7, 8, 8, 7, 1, 9, 5], [8, 7, 9, 0, 6, 6, 1, 2], [8, 8, 4, 1, 1, 6, 2, 5], [8, 8, 9, 4, 5, 4, 5, 9], [8, 9, 2, 0, 8, 1, 4, 3], [8, 9, 5, 3, 3, 5, 3, 5], [8, 9, 6, 4, 2, 1, 9, 6], [9, 3, 9, 7, 9, 4, 1, 6], [9, 6, 2, 5, 8, 3, 2, 6], [9, 7, 0, 7, 7, 2, 1, 3], [9, 7, 8, 1, 4, 1, 8, 4], [9, 7, 9, 2, 2, 3, 5, 5], [9, 7, 9, 2, 8, 2, 1, 1], [9, 8, 6, 8, 5, 7, 1, 2], [9, 9, 5, 4, 9, 4, 4, 8], [9, 9, 8, 2, 4, 4, 5, 7]]
'''
