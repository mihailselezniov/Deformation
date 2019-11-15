import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
import collections
import copy
import math
import pickle

with open('../11/fib_all_data.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))

def make_str(data):
    return ''.join(str(i) for i in data)
def make_set(data):
    return {make_str(i) for i in data}
def make_list(data):
    return [int(i) for i in data]

def get_ways(l0):
    ways = []
    for i in range(len(l0)):
        l0[i] += 1
        k = make_str(l0)
        if k in X_key_indexs:
            ways.append(X_key_indexs[k])
        l0[i] -= 2
        k = make_str(l0)
        if k in X_key_indexs:
            ways.append(X_key_indexs[k])
        l0[i] += 1
    return tuple(ways)
way_dict = []

X, Y = [], []
for i, val in enumerate(data_is_broken):
    Y.extend([i%2]*val)
#zero_keys, X_keys, X_key_indexs = set(), [], {}
zero_keys, X_key_indexs = set(), {}
n = tuple(range(10))
i = 0
ii = 0
y = []
a = np.empty((0,8), dtype=int)
for i0 in n:
    for i1 in n:
        for i2 in n:
            for i3 in n:
                for i4 in n:
                    for i5 in n:
                        for i6 in n:
                            for i7 in n:
                                if 0 not in [i4, i5, i6]:
                                    l0 = [i0, i1, i2, i3, i4, i5, i6, i7]
                                    X.append(l0)
                                    y.append(Y[i])
                                    key = make_str(l0)
                                    if Y[i] == 0:
                                        zero_keys.add(ii)
                                    #X_keys.append(key)
                                    X_key_indexs[key] = ii
                                    ii += 1
                                    #way_dict[key] = make_set(get_ways(l0))
                                i += 1
    a = np.append(a, np.array(X), axis=0)
    X = []
    print(i0)
    #break


print('!!!')
X, Y = a, np.array(y)
print(X.shape, Y.shape)
print('all', dict(collections.Counter(Y)))

print('way_dict', len(X))
for i in range(len(X)):
    way_dict.append(get_ways(X[i]))
    if not i%1000000:
        print(i)
way_dict = tuple(way_dict)

with open('border_vars/way_dict.p', 'wb') as f:
    pickle.dump(way_dict, f)
with open('border_vars/zero_keys.p', 'wb') as f:
    pickle.dump(zero_keys, f)
with open('border_vars/X_key_indexs.p', 'wb') as f:
    pickle.dump(X_key_indexs, f)
with open('border_vars/X.p', 'wb') as f:
    pickle.dump(X, f)
with open('border_vars/Y.p', 'wb') as f:
    pickle.dump(Y, f)






