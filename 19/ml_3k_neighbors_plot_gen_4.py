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

from get_shift import gen_shift
rang = [0, 1]
g_shift = gen_shift(rang)

ex_name = '1_mlp_plt_gen_4.txt'
ppl = [0.8, 0.9]# predict_proba_limits
folder_name = 'ml_threads/'
f_name = folder_name + ex_name


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
    #print('len low_grid =', len(low_grid[ind_max]))
    return low_grid[ind_max][0]

X = []
n = tuple([i for i in range(1000)])
for i0 in n:
    for i1 in n:
        X.append([i0, i1])
X = np.array(X)
X = X/100
#print(X[0])
#print(X[1])
#print(X[-1])

while 1:
    with open(f_name, 'r') as f:
        threads = f.readlines()

    if len(threads) >= 100:
        break

    X = X# + g_shift.__next__()
    xt = np.array_split(X, 5)

    x_train, y_train = [], []
    for t in threads:
        tr = list(map(float, t.replace('\n', '').split(',')))
        x_train.append(tr[:-1])
        y_train.append(tr[-1])
    x_train, y_train = np.array(x_train), np.array(y_train)
    zero, one = x_train[y_train == 0], x_train[y_train == 1]
    #print(zero, one, y_train)

    print('#', len(threads), 'y_train', dict(collections.Counter(y_train)))

    model = MLPClassifier(max_iter=100000, random_state=42)#max_iter=100000
    model.fit(x_train, y_train)

    #print(model.predict_proba(x_train)[:, 0])

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

    ppl = [0.45, 0.55]
    #print(pred_p[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])])
    Xr = X[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])]
    print(len(Xr))
    #print(Xr[:3])
    #print(Xr[-3:])

    point = get_closest(Xr, zero, one)
    #point = get_closest(Xr, np.concatenate((zero, one), axis=0), one)
    '''
    if len(zero) < len(one):
        print('> 1')
        point = get_closest(Xr, zero, one)
    else:
        print('> 0')
        point = get_closest(Xr, one, zero)
    '''
    point = [float(p) for p in point]
    #print('point:', point)

    rez = 0
    if ((point[0]-5)**2 + (point[1]-5)**2) > 2**2:
        rez = 1
    if ((point[0]-4)**2 + (point[1]-4)**2) > 2**2:
        rez = 1
    if ((point[0]-6)**2 + (point[1]-6)**2) > 2**2:
        rez = 1

    '''
    raw_point = get_raw2(point)
    #print(raw_point)
    str_point = ','.join(list(map(str, raw_point)))

    stream = os.popen('go run ../17/solve_fiber.go {}'.format(str_point))
    output = stream.read()
    '''

    row_data = list(point) + [float(rez)]
    row_thread = ','.join(list(map(lambda x: str(float(x)), row_data)))
    print(row_thread)
    #0/0
    with open(f_name, 'a') as f:
        f.write(row_thread + '\n')

"""

Xx = joblib.load('../16/border_vars/X.j')*10


#print(X[0])
#print(X[1])
#print(X[-1])

'''
X = []
n = tuple([i*10 for i in range(10)])
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
    print(i0)
X = np.array(X)
'''
#print(X[0])
#print(X[1])
#print(X[-1])


#print(len(xt), type(xt))
#print(xt[0][0], xt[-1][0])

'''
xt0 = X[:20000000]
xt1 = X[20000000:40000000]
xt2 = X[40000000:60000000]
xt3 = X[60000000:]
'''
par = {}
parts = 100*100
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
    return list(map(lambda x: round(x, 4), np.arange(Min, Max+0.01, (Max-Min)/(parts))))

def get_raw(par_inxs):
    return [get_list(**par[par_name])[int(par_inxs[pi]*100)] for pi, par_name in enumerate(order_par)]

def get_raw2(par_inxs):
    return [par[par_name]['Min']+((par[par_name]['Max']-par[par_name]['Min'])/100*par_inxs[pi]) for pi, par_name in enumerate(order_par)]

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

#min_max_X = get_min_max(X)

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
    #print('len low_grid =', len(low_grid[ind_max]))
    return low_grid[ind_max][0]

import random
shift = [0.02*i for i in range(1, 501)]
random.shuffle(shift, lambda *x: 0.42)

# -----
for i in range(2401-2):
    g_shift.__next__()
# -----

while 1:
    with open(f_name, 'r') as f:
        threads = f.readlines()

    #if len(threads) >= 500:
    #    break

    X = Xx + g_shift.__next__()#shift[len(threads)-2]# (0.02*(len(threads)-1))
    xt = np.array_split(X, 30)

    x_train, y_train = [], []
    for t in threads:
        tr = list(map(float, t.replace('\n', '').split(',')))
        x_train.append(tr[:-1])
        y_train.append(tr[-1])
    x_train, y_train = np.array(x_train), np.array(y_train)
    zero, one = x_train[y_train == 0], x_train[y_train == 1]
    #print(zero, one)

    print('#', len(threads), 'y_train', dict(collections.Counter(y_train)))

    model = MLPClassifier(max_iter=100000, random_state=42)
    model.fit(x_train, y_train)

    #print(model.predict_proba(x_train)[:, 0])

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

    #print(pred_p[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])])
    Xr = X[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])]
    print(len(Xr))
    #print(Xr[:3])
    #print(Xr[-3:])

    point = get_closest(Xr, zero, one)
    point = [float(p) for p in point]
    #print('point:', point)
    raw_point = get_raw2(point)
    #print(raw_point)
    str_point = ','.join(list(map(str, raw_point)))

    stream = os.popen('go run ../17/solve_fiber.go {}'.format(str_point))
    output = stream.read()
    row_data = list(point) + [float(output.split()[-1])]
    row_thread = ','.join(list(map(lambda x: str(float(x)), row_data)))
    print(row_thread)

    with open(f_name, 'a') as f:
        f.write(row_thread + '\n')
"""