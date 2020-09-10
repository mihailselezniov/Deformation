import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.neural_network import MLPRegressor
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


'''
stream = os.popen('go run dis_calc.go')
output = stream.read()
print(output)
'''
ex_name = '9_1_mlp_4.txt'
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

#print(sys.getsizeof(way_dict))
print(X.shape, Y.shape)
print('all', dict(collections.Counter(Y)))


def fit_model(model):
    #print('-'*10, model.__class__.__name__, '-'*10)
    model.fit(x_train, y_train)

    #print(X[:10])
    #print(model.predict_proba(X[:10]))
    #print(model.predict(X[:10]))
    #return []

    pred_p = model.predict_proba(X[:20000000])
    pred_p = np.concatenate((pred_p, model.predict_proba(X[20000000:40000000])))
    pred_p = np.concatenate((pred_p, model.predict_proba(X[40000000:60000000])))
    pred_p = np.concatenate((pred_p, model.predict_proba(X[60000000:])))

    #pred = model.predict(X[:20000000])
    #pred = np.concatenate((pred, model.predict(X[20000000:40000000])))
    #pred = np.concatenate((pred, model.predict(X[40000000:60000000])))
    #pred = np.concatenate((pred, model.predict(X[60000000:])))

    #return pred, pred_p
    return pred_p


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


    #y_pred = fit_model(MLPRegressor(activation='tanh', solver='adam', hidden_layer_sizes=(90, 90, 90), max_iter=100000, random_state=42))# 9_1_mlp_4


    #y_pred = [sum(i) for i in zip(*y_preds)]
    #d_pred = dict(collections.Counter(y_pred))
    #min_sum = 0.05#min(list(d_pred))
    #print('y_pred', d_pred, min_sum)

    #l = list(y_pred)
    #print(min(l), max(l), sum(l)/len(l), len([i for i in l if i < 0.5562]))


    #y_pred, y_pred_p = fit_model(MLPClassifier(activation='tanh', solver='adam', hidden_layer_sizes=(90, 90, 90), max_iter=100000, random_state=42))

    #break

    #l0 = list([i[0] for i in y_pred])
    #l1 = list([i[1] for i in y_pred])
    #print(min(l0), max(l0), sum(l0)/len(l0), len([i for i in l0 if i < 0.5]))
    #print(min(l1), max(l1), sum(l1)/len(l1), len([i for i in l1 if i < 0.5]))

    #break

    '''
    print('calc')
    l0, l1, l2 = [], [], []
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            l0.append(y_pred_p[i][0])
            l2.append(y_pred_p[i][1])
        else:
            l1.append(y_pred_p[i][1])

    print(min(l0), max(l0), sum(l0)/len(l0), len([i for i in l0 if i < 0.5]), len(l0))
    print(min(l1), max(l1), sum(l1)/len(l1), len([i for i in l1 if i < 0.5]), len(l1))
    print()
    print(min(l2), max(l2), sum(l2)/len(l2), len([i for i in l2 if i < 0.5]), len(l2))

    all {1: 63518691, 0: 9381309}
    # 502 y_train {1: 33, 0: 469}
    0.5000001649695833 0.9999996421500528 0.9370813505689694 0 6550464
    0.500000283121547 0.9999997343612763 0.9925767111682586 0 66349536

    3.5784994722026855e-07 0.49999983503041673 0.06291864943107138 6550464 6550464

    break
    '''

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
            if 0.5 < y_pred_p[iy][0] < 0.6:
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

    print('{}: {}'.format('0.5 < y_pred_p[iy][0] < 0.6:', min_count))
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
