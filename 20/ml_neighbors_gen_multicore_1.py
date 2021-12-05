import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.neural_network import MLPClassifier
import collections
import sys
import os
import time
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

ex_name = 'mlp_gen_1.txt'
ppl = [0.8, 0.9]# predict_proba_limits
#folder_name = 'ml_threads_1/'
#f_name = folder_name + ex_name
threads_folder = os.path.join(os.getcwd(), '../ml_threads_1')#../ml_threads_1
f_name = os.path.join(threads_folder, ex_name)
os.system('mkdir ' + threads_folder)
os.system('touch ' + f_name)


rezult_destruction = os.path.join(os.getcwd(), '../rezult_destruction_4_1')#../rezult_destruction_4_1 !!!!!!!
os.system('mkdir ' + rezult_destruction)
files = os.listdir(rezult_destruction)


N1 = 25
N2 = 201
strike_energy = np.linspace(0, 50, N2, endpoint=True)
strike_energy = list(map(lambda x: round(x, 5), strike_energy))

protection_thickness = np.linspace(2.0, 8.0, N1, endpoint=True)
protection_thickness = list(map(lambda x: round(x, 5), protection_thickness))


base_ids = {0: 0, 6: 0, 12: 0, 18: 0, 24: 1, 1250: 1, 1256: 1, 1262: 1, 1268: 1, 1274: 1, 2500: 1, 2506: 1, 2512: 1, 2518: 1, 2524: 1, 3750: 1, 3756: 1, 3762: 1, 3768: 1, 3774: 1, 5000: 1, 5006: 1, 5012: 1, 5018: 1, 5024: 1}

X = []
i = 0
for s in strike_energy:
    for p in protection_thickness:
        X.append([s, p])
        i += 1
X = np.array(X)
xt = np.array_split(X, 5)

Xids = {}
i = 0
for s in strike_energy:
    for p in protection_thickness:
        Xids[str([float(p) for p in X[i]])] = i
        i += 1



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


#X = []
#n = tuple([i for i in range(1000)])
#for i0 in n:
#    for i1 in n:
#        X.append([i0, i1])
#X = np.array(X)
#X = X/100

#print(X[0])
#print(X[1])
#print(X[-1])

state_y_train = ''
num = 0
while 1:
    work_threads = []
    files = os.listdir(rezult_destruction)
    set_f_ids = set([int(f_name.split('.')[0]) for f_name in files])

    x_train, y_train = [], []
    i = 0
    for s in strike_energy:
        for p in protection_thickness:
            if i in base_ids:
                x_train.append([s, p])
                y_train.append(base_ids[i])
            if i in set_f_ids:
                rez_f = open(os.path.join(rezult_destruction, '{}.txt'.format(i))).read()
                rez_f = ''.join(rez_f.split())
                if rez_f:
                    if rez_f in '01':
                        x_train.append([s, p])
                        y_train.append(int(rez_f))
                else:
                    work_threads.append([s, p])
            i += 1
    work_threads = np.array(work_threads)

    if state_y_train == str(y_train):
        num += 1
        print(num, 'ids:', len(set_f_ids))
        time.sleep(30)
        continue
    state_y_train = str(y_train)


    x_train, y_train = np.array(x_train), np.array(y_train)
    zero, one = x_train[y_train == 0], x_train[y_train == 1]

    print('#', 'y_train', dict(collections.Counter(y_train)))

    #X = X# + g_shift.__next__()
    

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
    print(zero)
    zero = np.concatenate((zero, work_threads), axis=0)
    print(zero)

    with open(f_name, 'r') as f:
        threads = f.readlines()
    with open(f_name, 'w') as f:
        for t in threads:
            t = ''.join([i for i in t if i.isdigit()])
            #print('t', t, int(t), t.isdigit())
            if not t.isdigit():
                continue
            if int(t) in set_f_ids:
                f.write(t + '\n')

    with open(f_name, 'a') as f:
        for i in range(200):
            point = get_closest(Xr, zero, one)
            all_zero = set([str([float(p) for p in z]) for z in zero])
            str_point = str([float(p) for p in point])
            thread_id = str(Xids[str_point])
            print(thread_id)
            f.write(thread_id + '\n')
            if str_point in all_zero:
                break
            zero = np.concatenate((zero, np.array([point])), axis=0)
    print(zero)
    #point = get_closest(Xr, np.concatenate((zero, one), axis=0), one)
    '''
    if len(zero) < len(one):
        print('> 1')
        point = get_closest(Xr, zero, one)
    else:
        print('> 0')
        point = get_closest(Xr, one, zero)
    '''

    #with open(f_name, 'r') as f:
    #threads = f.readlines()



    point = [float(p) for p in point]
    print('point:', point)



'''
    rez = 0
    if ((point[0]-5)**2 + (point[1]-5)**2) > 2**2:
        rez = 1
    if ((point[0]-4)**2 + (point[1]-4)**2) > 2**2:
        rez = 1
    if ((point[0]-6)**2 + (point[1]-6)**2) > 2**2:
        rez = 1

    
    #raw_point = get_raw2(point)
    #print(raw_point)
    #str_point = ','.join(list(map(str, raw_point)))

    #stream = os.popen('go run ../17/solve_fiber.go {}'.format(str_point))
    #output = stream.read()
    

    row_data = list(point) + [float(rez)]
    row_thread = ','.join(list(map(lambda x: str(float(x)), row_data)))
    print(row_thread)
    #0/0
    with open(f_name, 'a') as f:
        f.write(row_thread + '\n')
'''
