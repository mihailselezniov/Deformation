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


with open('../11/fib_all_data.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))


X = [[], [], [], [], [], [], [], [], []]

n = tuple(range(10))
i = 0
#lvl = 0
pairs0 = ((0, 7),)
pairs1 = ((0, 6), (1, 7))
pairs2 = ((0, 5), (1, 6), (2, 7))
pairs3 = ((0, 4), (1, 5), (2, 6), (3, 7))
pairs4 = ((0, 3), (1, 4), (2, 5), (3, 6), (4, 7))
pairs5 = ((0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7))
pairs6 = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7))
#pair_i = 0
a = np.empty((0,8), dtype=int)

def gen_pairs(pair):
    print(pair)
    X = [[], [], [], [], [], [], [], [], []]
    n = tuple(range(10))
    for i0 in n:
        for i1 in n:
            for i2 in n:
                for i3 in n:
                    for i4 in n:
                        for i5 in n:
                            for i6 in n:
                                for i7 in n:
                                    #if 0 not in [i4, i5, i6]:
                                    for row_inx in range(10):
                                        l = [i0, i1, i2, i3, i4, i5, i6, i7]
                                        del l[pair[0]]
                                        del l[pair[1]-1]
                                        fix = [i for i in l if i == row_inx]#(0, 4)
                                        i4 = 1 if i4 == 0 else i4
                                        i5 = 1 if i5 == 0 else i5
                                        i6 = 1 if i6 == 0 else i6
                                        if len(fix) == 6:
                                            X[row_inx-1].append(np.array([i0, i1, i2, i3, i4, i5, i6, i7])*10)
        #print(i0)
        #break
    #a = np.append(a, np.array(X), axis=0)
    for ix in range(len(X)):
        X[ix] = np.unique(np.array(X[ix]), axis=0)
    return X



a = gen_pairs(pairs0[0])
for p in pairs1: a.extend(gen_pairs(p))
for p in pairs2: a.extend(gen_pairs(p))
for p in pairs3: a.extend(gen_pairs(p))
for p in pairs4: a.extend(gen_pairs(p))
for p in pairs5: a.extend(gen_pairs(p))
for p in pairs6: a.extend(gen_pairs(p))

joblib.dump(a, 'vars/X.j')




"""
#--------------------------------------------------


ex_name = '13_1_mlp_5.txt'
ppl = [0.8, 0.9]# predict_proba_limits
folder_name = 'ml_threads/'
f_name = folder_name + ex_name


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
pred_p = model.predict_proba(X)[:, 0]
Xr = X[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])]
print(len(Xr), end=' ')

if len(Xr):
    point = get_closest(Xr, zero, one)
    print(point)

    raw_point = get_raw(point)
    str_point = ','.join(list(map(str, raw_point)))

    stream = os.popen('go run solve_fiber.go {}'.format(str_point))
    output = stream.read()
    row_data = list(point) + [int(output.split()[-1])]
    row_thread = ','.join(list(map(lambda x: str(int(x)), row_data)))
    print(row_thread)
    with open(f_name, 'a') as f:
        f.write(row_thread + '\n')
"""







"""
ex_name = '13_1_mlp_4.txt'
ppl = [0.8, 0.9]# predict_proba_limits
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

    '''
    grid0 = get_segments(point, min_max_X, 0)
    pred_p = model.predict_proba(grid0)[:, 0]
    Xr = grid0[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])]
    point = get_closest(Xr, zero, one)

    grid1 = get_segments(point, get_min_max(grid0), 1)
    pred_p = model.predict_proba(grid1)[:, 0]
    Xr = grid1[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])]
    point = get_closest(Xr, zero, one)
    '''
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
"""