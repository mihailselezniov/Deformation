# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score, f1_score
from sklearn.neural_network import MLPClassifier
import collections
import sys


#num_activation = int(sys.argv[1])

opt_activations = ['tanh', 'relu']
opt_solvers = ['adam']
max_hidden_layer_sizes = 100
#opt_hidden_layer_sizes = [(i,i,i) for i in range(10, 200, 20)]
opt_hidden_layer_sizes = [(30, 30), (50, 50), (70, 70), (70, 30), (100, 10), (30, 70), (10, 100), (30, 50, 70), (70, 50, 30), (10, 50, 100), (100, 50, 10), (30, 70, 30), (70, 30, 70), (10, 100, 10), (100, 10, 100), (30, 30, 30, 30), (50, 50, 50, 50), (70, 70, 70, 70), (30, 30, 30, 30, 30), (50, 50, 50, 50, 50), (70, 70, 70, 70, 70)][::-1]

#print(opt_activations[num_activation])
'''
for inx_activation, opt_activation in enumerate(opt_activations):
    for inx_solver, opt_solver in enumerate(opt_solvers):
        for layer in opt_hidden_layer_sizes:
            print(inx_activation, opt_activation, inx_solver, opt_solver, layer)
'''


par = {}
parts = 10
par['pressure_time'] = {'Min': 0.0, 'Max': 100.0}
par['pressure_radius'] = {'Min': 0.0, 'Max': 5.0}
par['pressure_amplitude'] = {'Min': 0.0, 'Max': 200.0}
par['length'] = {'Min': 10.0, 'Max': 100.0}
par['diameter'] = {'Min': 0.01, 'Max': 0.5}
par['young'] = {'Min': 60.0, 'Max': 300.0}
par['density'] = {'Min': 1000.0, 'Max': 2000.0}
par['strength'] = {'Min': 0.2, 'Max': 10.0}

def get_list(Min, Max):
    return list(map(lambda x: round(x, 2), np.arange(Min, Max+0.01, (Max-Min)/(parts-1))))

e0 = tuple(enumerate(get_list(**par['length'])))
e1 = tuple(enumerate(get_list(**par['diameter'])))
e2 = tuple(enumerate(get_list(**par['young'])))
e3 = tuple(enumerate(get_list(**par['density'])))
e4 = tuple(enumerate(get_list(**par['pressure_time'])))
e5 = tuple(enumerate(get_list(**par['pressure_radius'])))
e6 = tuple(enumerate(get_list(**par['pressure_amplitude'])))
e7 = tuple(enumerate(get_list(**par['strength'])))


extreme_values = [[
        par['length']['Min'],
        par['diameter']['Min'],
        par['young']['Min'],
        par['density']['Min'],
        par['pressure_time']['Min'],#get_list(**par['pressure_time'])[1],
        par['pressure_radius']['Min'],#get_list(**par['pressure_radius'])[1],
        par['pressure_amplitude']['Min'],#get_list(**par['pressure_amplitude'])[1],
        par['strength']['Min'],
        ],
        [
        par['length']['Max'],
        par['diameter']['Max'],
        par['young']['Max'],
        par['density']['Max'],
        par['pressure_time']['Max'],
        par['pressure_radius']['Max'],
        par['pressure_amplitude']['Max'],
        par['strength']['Max'],
        ]
    ]
extreme_values = np.array(extreme_values)
#x_train = (x_train - extreme_values.min(axis=0)) / (extreme_values.max(axis=0) - extreme_values.min(axis=0))


with open('../15/data3k_2.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))

y_test = []
for i, val in enumerate(data_is_broken):
    y_test.extend([i%2]*val)

new_parts = 19
def get_new_list(Min, Max):
    return list(map(lambda x: round(x, 2), np.arange(Min, Max+0.01, (Max-Min)/(new_parts-1))))[1::2]

e2_0 = tuple(enumerate(get_new_list(**par['length'])))
e2_1 = tuple(enumerate(get_new_list(**par['diameter'])))
e2_2 = tuple(enumerate(get_new_list(**par['young'])))
e2_3 = tuple(enumerate(get_new_list(**par['density'])))
e2_4 = tuple(enumerate(get_new_list(**par['pressure_time'])))
e2_5 = tuple(enumerate(get_new_list(**par['pressure_radius'])))
e2_6 = tuple(enumerate(get_new_list(**par['pressure_amplitude'])))
e2_7 = tuple(enumerate(get_new_list(**par['strength'])))


x_test = []
for i0, l in e2_0:
    for i1, di in e2_1:
        for i2, y in e2_2:
            for i3, de in e2_3:
                for i4, pt in e2_4:
                    for i5, pr in e2_5:
                        for i6, pa in e2_6:
                            for i7, s in e2_7:
                                #if 0 not in [i4, i5, i6]:
                                x_test.append([l, di, y, de, pt, pr, pa, s])
    print(i0)
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = (x_test - extreme_values.min(axis=0)) / (extreme_values.max(axis=0) - extreme_values.min(axis=0))



def make_str(data):
    return ''.join(map(str, data))
def make_set(data):
    return {make_str(i) for i in data}

source_f = 'ml_threads/7_1_i3.txt'
with open(source_f, 'r') as f:
    threads = f.readlines()

roc_metrics, pr_metrics, f1_metrics = [], [], []
roc_metric, pr_metric, f1_metric = [], [], []

#for cut in [100, 200, 300, 400, 500]:





cut = 100#100
print('\n\n\n', '#'*10, cut, '#'*10)
x_train_dict = {}
for t in threads[:cut]:
    tr = list(map(int, t.replace('\n', '').split(',')))
    x_train_dict[make_str(tr[:-1])] = tr[-1]

i = 0
x_train, y_train = [], []
for i0, l in e0:
    for i1, di in e1:
        for i2, y in e2:
            for i3, de in e3:
                for i4, pt in e4:
                    for i5, pr in e5:
                        for i6, pa in e6:
                            for i7, s in e7:
                                if 0 not in [i4, i5, i6]:
                                    key = make_str([i0, i1, i2, i3, i4, i5, i6, i7])
                                    if key in x_train_dict:
                                        x_train.append([l, di, y, de, pt, pr, pa, s])
                                        y_train.append(x_train_dict[key])
                                i += 1
    print(i0)
x_train, y_train = np.array(x_train), np.array(y_train)


x_train = (x_train - extreme_values.min(axis=0)) / (extreme_values.max(axis=0) - extreme_values.min(axis=0))
#print(x_train)


def fit_model(model):
    global roc_metric
    global pr_metric
    global f1_metric
    #print('\n', '-'*10, model.__class__.__name__, '-'*10)
    print(x_test.shape, y_test.shape)
    print('y_test', dict(collections.Counter(y_test)), 'y_train', dict(collections.Counter(y_train)))
    # fit model on training data
    model.fit(x_train, y_train)


    #fix
    #y_pred = model.predict(x_test)
    print('predict')

    #fix
    y_pred = model.predict(x_test[:10000000])
    y_pred = np.concatenate((y_pred, model.predict(x_test[10000000:20000000])))
    y_pred = np.concatenate((y_pred, model.predict(x_test[20000000:30000000])))
    y_pred = np.concatenate((y_pred, model.predict(x_test[30000000:40000000])))
    y_pred = np.concatenate((y_pred, model.predict(x_test[40000000:])))

    """
    y_pred = model.predict(x_test[:1000000])
    for i in range(1, 45):
        inx_from, inx_to = i * 1000000, (i + 1) * 1000000
        y_pred = np.concatenate((y_pred, model.predict(x_test[inx_from:inx_to])))
    """





    print('y_pred', dict(collections.Counter(y_pred)))
    # make predictions for test data
    y_pred = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {}'.format(accuracy))

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:\n{}'.format(cm))

    print('Precision, recall and f1-score:')
    print(classification_report(y_test, y_pred))

    try:
        roc = roc_auc_score(y_test, y_pred)
        print('ROC AUC: {}'.format(roc))

        pr = average_precision_score(y_test, y_pred)
        print('PR AUC: {}'.format(pr))

        roc_metric.append(round(float(roc), 4))
        pr_metric.append(round(float(pr), 4))

        f1 = f1_score(y_test, y_pred, average=None)
        f1_metric.append(round(float(f1[0]), 4))

    except Exception as e:
        print(e)

    #print('-'*10, 'End',  model.__class__.__name__, '-'*10)

"""
roc_metric, pr_metric, f1_metric = [], [], []
fit_model(MLPClassifier(hidden_layer_sizes=(200,200,200), max_iter=100000, random_state=42))
roc_metrics.append(roc_metric[:])
pr_metrics.append(pr_metric[:])
f1_metrics.append(f1_metric[:])
"""

for inx_activation, opt_activation in enumerate(opt_activations):
    for inx_solver, opt_solver in enumerate(opt_solvers):

        roc_metric, pr_metric, f1_metric = [], [], []

        for layer in opt_hidden_layer_sizes:
            print(inx_activation, opt_activation, inx_solver, opt_solver, layer)

            fit_model(MLPClassifier(activation=opt_activation, solver=opt_solver, hidden_layer_sizes=layer, max_iter=100000, random_state=42))

        print(roc_metric, pr_metric, f1_metric)

        roc_metrics.append(roc_metric[:])
        pr_metrics.append(pr_metric[:])
        f1_metrics.append(f1_metric[:])

"""
inx_activation, opt_activation = num_activation, opt_activations[num_activation]

for inx_solver, opt_solver in enumerate(opt_solvers):

    roc_metric, pr_metric, f1_metric = [], [], []

    for layer in opt_hidden_layer_sizes:
        print(inx_activation, opt_activation, inx_solver, opt_solver, layer)

        fit_model(MLPClassifier(activation=opt_activation, solver=opt_solver, hidden_layer_sizes=layer, max_iter=100000, random_state=42))

    print(roc_metric, pr_metric, f1_metric)

    roc_metrics.append(roc_metric[:])
    pr_metrics.append(pr_metric[:])
    f1_metrics.append(f1_metric[:])
"""
print('roc_metrics')
print(roc_metrics)
print('pr_metrics')
print(pr_metrics)
print('f1_metrics')
print(f1_metrics)
print()



'''


'''
