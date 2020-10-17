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


ex_name = '13_1_mlp_3.txt'
ppl = [0.85, 0.9]# predict_proba_limits
folder_name = 'ml_threads/'
f_name = folder_name + ex_name
X = joblib.load('../16/border_vars/X.j')*10
xt0 = X[:20000000]
xt1 = X[20000000:40000000]
xt2 = X[40000000:60000000]
xt3 = X[60000000:]

par = {}
parts = 100
par['length'] = {'Min': 10.0, 'Max': 100.0}
par['diameter'] = {'Min': 0.01, 'Max': 0.5}
par['young'] = {'Min': 60.0, 'Max': 300.0}
par['density'] = {'Min': 1000.0, 'Max': 2000.0}
par['pressure_time'] = {'Min': 0.0, 'Max': 100.0}
par['pressure_radius'] = {'Min': 0.0, 'Max': 5.0}
par['pressure_amplitude'] = {'Min': 0.0, 'Max': 200.0}
par['strength'] = {'Min': 0.2, 'Max': 10.0}
order_par = ['length', 'diameter', 'young', 'density', 'pressure_time', 'pressure_radius', 'pressure_amplitude', 'strength']

def get_list(Min, Max):
    return list(map(lambda x: round(x, 4), np.arange(Min, Max+0.01, (Max-Min)/(parts-1))))

def get_raw(par_inxs):
    return [get_list(**par[par_name])[par_inxs[pi]] for pi, par_name in enumerate(order_par)]

def get_range(num, start, end, epoch):
    step = 1 if epoch else 3
    parts = 2 if epoch else 3
    left = (num - (parts * step)) if num != start else num
    right = (num + (parts * step) + 1) if num != end else num + 1
    r = range(left, right, step)
    return r if step > 0 else r[::-1]

def get_segments(point, min_max, epoch):
    e = [tuple(enumerate(get_range(point[i], min_max[i][0], min_max[i][1], epoch))) for i in range(len(point))]
    x_test = []
    for i0, l in e[0]:
        for i1, di in e[1]:
            for i2, y in e[2]:
                for i3, de in e[3]:
                    for i4, pt in e[4]:
                        for i5, pr in e[5]:
                            for i6, pa in e[6]:
                                for i7, s in e[7]:
                                    x_test.append([l, di, y, de, pt, pr, pa, s])
    x_test = np.array(x_test)
    return x_test

def get_min_max(grid):
    return np.stack((np.amin(grid, axis=0), np.amax(grid, axis=0)), axis=-1)

min_max_X = get_min_max(X)

def pred_model(model):
    pred_p = model.predict_proba(xt0)
    pred_p = np.concatenate((pred_p, model.predict_proba(xt1)))
    pred_p = np.concatenate((pred_p, model.predict_proba(xt2)))
    pred_p = np.concatenate((pred_p, model.predict_proba(xt3)))
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
    print(len(low_grid[ind_max]), end=' ')
    return low_grid[ind_max][0]

while 1:
    with open(f_name, 'r') as f:
        threads = f.readlines()

    x_train, y_train = [], []
    for t in threads:
        tr = list(map(int, t.replace('\n', '').split(',')))
        x_train.append(tr[:-1])
        y_train.append(tr[-1])
    x_train, y_train = np.array(x_train), np.array(y_train)
    zero, one = x_train[y_train == 0], x_train[y_train == 1]

    print('#', len(threads), 'y_train', dict(collections.Counter(y_train)))

    model = MLPClassifier(max_iter=100000, random_state=42)
    model.fit(x_train, y_train)

    pred_p = pred_model(model)[:, 0]

    Xr = X[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])]
    print(len(Xr), end=' ')

    point = get_closest(Xr, zero, one)
    
    grid0 = get_segments(point, min_max_X, 0)
    pred_p = model.predict_proba(grid0)[:, 0]
    Xr = grid0[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])]
    point = get_closest(Xr, zero, one)

    grid1 = get_segments(point, get_min_max(grid0), 1)
    pred_p = model.predict_proba(grid1)[:, 0]
    Xr = grid1[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])]
    point = get_closest(Xr, zero, one)
    print()

    raw_point = get_raw(point)
    str_point = ','.join(list(map(str, raw_point)))

    stream = os.popen('go run solve_fiber.go {}'.format(str_point))
    output = stream.read()
    row_data = list(point) + [int(output.split()[-1])]
    row_thread = ','.join(list(map(lambda x: str(int(x)), row_data)))
    print(row_thread)
    with open(f_name, 'a') as f:
        f.write(row_thread + '\n')
