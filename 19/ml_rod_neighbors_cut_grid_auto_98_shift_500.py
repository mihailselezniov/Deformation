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

ex_name = '1_rod_98_500.txt'
ppl = [0.8, 0.9]# predict_proba_limits
folder_name = 'ml_threads/'
f_name = folder_name + ex_name

'''
X = []
n = tuple([i for i in range(99)])
for i0 in n:
    for i1 in n:
        for i2 in n:
            for i3 in n:
                X.append([i0, i1, i2, i3])

X = np.array(X)
'''

def cartesian_coord(*arrays):
    grid = np.meshgrid(*arrays)        
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points



def pred_model(model):
    pred_p = model.predict_proba(xt[0])
    for i in range(1, len(xt)):
        pred_p = np.concatenate((pred_p, model.predict_proba(xt[i])))
    return pred_p

def get_closest(grid, zero, one):
    min_dis = np.linalg.norm(grid - zero[0], axis=1)
    for i in range(1, len(zero)):
        min_dis = np.minimum(min_dis, np.linalg.norm(grid - zero[i], axis=1))
    max_dis = min_dis.max()
    ind_max = min_dis == max_dis
    low_grid = grid[ind_max]

    min_dis = np.linalg.norm(low_grid - one[0], axis=1)
    for i in range(1, len(one)):
        min_dis = np.minimum(min_dis, np.linalg.norm(low_grid - one[i], axis=1))
    max_dis = min_dis.max()
    ind_max = min_dis == max_dis
    return low_grid[ind_max][0]

while 1:

    with open(f_name, 'r') as f:
        threads = f.readlines()

    if len(threads) >= 500:
        break

    a = np.arange(99)
    X = cartesian_coord(*4*[a])
    X = X+(0.002*(len(threads)-1))
    #print(X[0])
    #print(X[1])
    #print(X[-1])

    xt = np.array_split(X, 10)


    x_train, y_train = [], []
    for t in threads:
        tr = list(map(float, t.replace('\n', '').split(',')))
        x_train.append(tr[:-1])
        y_train.append(tr[-1])
    x_train, y_train = np.array(x_train), np.array(y_train)
    zero, one = x_train[y_train == 0], x_train[y_train == 1]
    #print(zero, one)
    #print(x_train)

    print('#', len(threads), 'y_train', dict(collections.Counter(y_train)))

    model = MLPClassifier(max_iter=100000, random_state=42)
    model.fit(x_train, y_train)

    pred_p = pred_model(model)[:, 0]

    num_zeros = np.count_nonzero(0.5 < pred_p)
    num_ones = np.count_nonzero(0.5 > pred_p)
    print({1: num_ones, 0: num_zeros})
    sum_all_nums = num_zeros + num_ones
    pre_zeros = round(num_zeros*100/sum_all_nums, 2)
    pre_ones = round(num_ones*100/sum_all_nums, 2)
    print({1: pre_ones, 0: pre_zeros})

    if len(threads) <= 10:
        ppl = [0.45, 0.55]
    else:
        mid = round(num_ones/sum_all_nums, 2)
        ppl = [mid-0.05, mid+0.05]
        if ppl[1] > 0.95:
            ppl = [0.9, 0.95]
        print('mid =', mid)
    print(ppl)

    Xr = X[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])]
    print(len(Xr))

    point = get_closest(Xr, zero, one)
    str_point = ' '.join(list(map(str, point)))

    stream = os.popen('./rod3 {}'.format(str_point))
    output = stream.read()
    row_data = list(point) + [int(output.split()[-1])]
    row_thread = ','.join(list(map(lambda x: str(x), row_data)))
    print(row_thread)
    with open(f_name, 'a') as f:
        f.write(row_thread + '\n')
