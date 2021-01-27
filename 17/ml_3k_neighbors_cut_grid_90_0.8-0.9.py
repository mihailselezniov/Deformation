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


ex_name = '14_1_mlp_1.txt'
ppl = [0.8, 0.9]# predict_proba_limits
folder_name = 'ml_threads/'
f_name = folder_name + ex_name
X = joblib.load('../16/border_vars/X.j')*10


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

xt = np.array_split(X, 10)
#print(len(xt), type(xt))
#print(xt[0][0], xt[-1][0])

'''
xt0 = X[:20000000]
xt1 = X[20000000:40000000]
xt2 = X[40000000:60000000]
xt3 = X[60000000:]
'''
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

while 1:
    with open(f_name, 'r') as f:
        threads = f.readlines()

    if len(threads) >= 500:
        break

    x_train, y_train = [], []
    for t in threads:
        tr = list(map(int, t.replace('\n', '').split(',')))
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

    #print(pred_p[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])])
    Xr = X[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])]
    print(len(Xr))
    #print(Xr[:3])
    #print(Xr[-3:])

    point = get_closest(Xr, zero, one)

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
    point = get_closest(Xr, zero, one)
    
    grid0 = get_segments(point, min_max_X, 0)
    pred_p = model.predict_proba(grid0)[:, 0]
    Xr = grid0[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])]
    point = get_closest(Xr, zero, one)

    grid1 = get_segments(point, get_min_max(grid0), 1)
    pred_p = model.predict_proba(grid1)[:, 0]
    Xr = grid1[np.logical_and(ppl[0] < pred_p, pred_p < ppl[1])]
    point = get_closest(Xr, zero, one)

    raw_point = get_raw(point)
    str_point = ','.join(list(map(str, raw_point)))

    stream = os.popen('go run solve_fiber.go {}'.format(str_point))
    output = stream.read()
    row_data = list(point) + [int(output.split()[-1])]
    row_thread = ','.join(list(map(lambda x: str(int(x)), row_data)))
    print(row_thread)
    with open(f_name, 'a') as f:
        f.write(row_thread + '\n')
    
    #break


"""

#print(get_segments([0,0,0,0,10,10,10,0], 0))
'''
print(list(get_range(0, 0, 99, 0)))
print(list(get_range(0, 0, 99, 1)))
print()
print(list(get_range(99, 0, 99, 0)))
print(list(get_range(99, 0, 99, 1)))
print()
print(list(get_range(10, 0, 90, 0)))
print(list(get_range(84, 0, 99, 1)))
'''
'''
min_max = [[0, 99], [0, 99], [0, 99], [0, 99], [10, 99], [10, 99], [10, 99], [0, 99]]

grid0 = get_segments([10,0,0,0,20,10,10,0], min_max, 0)
grid1 = get_segments([10,0,0,0,20,10,10,0], min_max, 1)
print(grid0)
print(get_min_max(grid0))
print(grid1)
print(get_min_max(grid1))

grid0 = get_segments([10,0,0,0,20,10,10,0], min_max, 0)
grid1 = get_segments(grid0[-1], get_min_max(grid0), 1)
print(grid1[-1])
'''













"""
# Get raw
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

#print(get_raw([0,0,0,0,10,10,10,0])) # [10.0, 0.01, 60.0, 1000.0, 10.101, 0.5051, 20.202, 0.2]
#print(get_raw([99,99,99,99,99,99,99,99])) # [100.0, 0.5, 300.0, 2000.0, 100.0, 5.0, 200.0, 10.0]
"""




"""
# Calc dist
grid = [[0,0,0,0,1,1,1,0],
      [0,0,0,0,2,2,2,0],
      [0,0,0,0,3,3,3,0],
      [0,0,0,0,9,9,9,0],
      [8,9,9,9,1,1,1,9],
      [9,8,9,9,1,1,1,9],
      [9,9,8,9,1,1,1,9]]
zero =[[0,0,0,0,1,1,1,0],
       [0,0,0,0,2,2,2,0],
       [0,0,0,0,3,3,3,0]]
one = [[5,5,5,5,5,5,5,5],
       [6,6,6,6,6,6,6,6],
       [8,9,9,9,3,3,3,9]]

grid, zero, one = np.array(grid), np.array(zero), np.array(one)

min_dis = np.linalg.norm(grid - zero[0], axis=1)
print(min_dis)
for i in range(1, len(zero)):
    min_dis = np.minimum(min_dis, np.linalg.norm(grid - zero[i], axis=1))

print(np.linalg.norm(grid - zero[1], axis=1))
print(np.linalg.norm(grid - zero[2], axis=1))
print()
print(min_dis)

max_dis = min_dis.max()
ind_max = min_dis == max_dis
print(ind_max)
print(grid[ind_max])
low_grid = grid[ind_max]

min_dis = np.linalg.norm(low_grid - one[0], axis=1)
print(min_dis)
for i in range(1, len(one)):
    min_dis = np.minimum(min_dis, np.linalg.norm(low_grid - one[i], axis=1))
max_dis = min_dis.max()
ind_max = min_dis == max_dis
print(ind_max)
print(low_grid[ind_max])
print(low_grid[ind_max][0])
"""




"""
def get_range(num, start, end, epoch):
    step = 1 if epoch else 3
    #if num == end:
    #    step *= -1

    parts = 2 if epoch else 3
    '''
    if num in [start, end]:
        parts = 2 if epoch else 3
    else:
        parts = 3
    '''
    left = (num - (parts * step)) if num != start else num
    right = (num + (parts * step) + 1) if num != end else num + 1
    '''
    if step < 0:
        right, left = left, right
        right += 1
        print(left, right, step)
        #step *= -1
    '''
    r = range(left, right, step)
    return r if step > 0 else r[::-1]
"""














"""
ex_name = '10_1_mlp_3.txt'
folder_name = 'ml_threads/'
folder_name_y = 'ml_y_pred/'
f_name = folder_name + ex_name
f_y_pred = folder_name_y + 'y{}_' + ex_name
n_cors = 4
Y_step = 729000#0
Y_step = 7290000

X = joblib.load('../16/border_vars/X.j')
print('X')
Y = joblib.load('../16/border_vars/Y.j')
print('Y')

print(X.shape, Y.shape)
print('all', dict(collections.Counter(Y)))
print(X[0])
"""



    

"""
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


    y_pred_p = fit_model(MLPClassifier(max_iter=100000, random_state=42))


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
"""