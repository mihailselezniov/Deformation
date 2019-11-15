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
    break


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

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

threads = {
    0: [[2,5,6,7,2,5,1,9],[3,9,2,8,2,6,1,6],[4,9,7,3,3,3,3,9],[5,7,9,6,1,4,2,6],[6,5,4,3,3,2,1,2],[7,6,5,4,4,3,2,2],[9,6,1,7,1,2,3,9]],
    1: [[0,0,0,0,1,1,1,0],[2,0,7,4,5,7,7,7],[4,0,3,6,3,4,9,9],[5,8,2,0,5,8,5,3],[6,4,8,8,4,2,7,7],[7,5,0,6,9,1,3,6],[8,4,1,5,3,4,8,3]]
}

for _ in range(721):#64
    print('#', _)
    x_train, y_train = [], []

    for i in threads:
        for thread in threads[i]:
            x_train.append(thread)
            y_train.append(i)

    x_train, y_train = np.array(x_train), np.array(y_train)

    #print(x_train, y_train)


    model = XGBClassifier()
    #model = LogisticRegression()
    #model = LinearSVC(random_state=0, tol=1e-5)
    #model = KNeighborsClassifier(n_neighbors=6)

    model.fit(x_train, y_train)
    y_pred = model.predict(X)
    accuracy = accuracy_score(Y, y_pred)
    print('Accuracy: {}'.format(accuracy))
    #print(y_pred)
    print(dict(collections.Counter(y_pred)))

    if _ in [0, 63, 720]:
        cm = confusion_matrix(Y, y_pred)
        print('Confusion matrix:\n{}'.format(cm))

        print('Precision, recall and f1-score:')
        print(classification_report(Y, y_pred))

        roc = roc_auc_score(Y, y_pred)
        print('ROC AUC: {}'.format(roc))

        pr = average_precision_score(Y, y_pred)
        print('Precision-recall: {}'.format(pr))


    find = 1 if len(threads[0]) > len(threads[1]) else 0



    #print(get_ways([8,8,8,8,2]))

    find_set, border_threads = set(), set()
    find_set = {i for i in range(len(y_pred)) if y_pred[i] == find}
    #find_set = {X_key_indexs(_x) for _x, _y in zip(X_keys, y_pred) if _y == find}
    #    if _y == find:
    #        find_set.add(make_str(_x))
    #    else:
    #        way_set.update(way_dict[key])
    '''
    n = tuple(range(len(X[0])))
    for _x, _y in zip(X, y_pred):
        if _y != find:
            
            for i in n:
                _x[i] += 1
                k = make_str(_x)
                if k in find_set:
                    border_threads.add(k)
                _x[i] -= 2
                k = make_str(_x)
                if k in find_set:
                    border_threads.add(k)
                _x[i] += 1
            #border_threads.update({i for i in get_ways(_x) if i in find_set})
    '''
    print('find_set', len(find_set), len(y_pred))
    #[border_threads.update(find_set.intersection(way_dict[i])) for i in range(len(y_pred)) if y_pred[i] != find]
    for i in range(len(y_pred)):
        if y_pred[i] != find:
            border_threads.update(set(way_dict[i]).intersection(find_set))


    #border_threads = find_set.intersection(way_set)
    print('border_threads', len(border_threads))

    def distance(l0):
        return min([math.sqrt(sum(map(lambda x: (x[0] - x[1])**2, zip(thread, l0)))) for thread in threads[find]])

    distance_threads = [[distance(list(X[i])), i] for i in border_threads]
    distance_threads.sort(reverse=True)
    print(distance_threads[0])
    k = distance_threads[0][1]
    new_thread = list(X[k])
    print(k in zero_keys)
    if k in zero_keys:
        threads[0].append(new_thread)
    else:
        threads[1].append(new_thread)
    print(threads)
    break

x_train, y_train = [], []
for i in threads:
    for thread in threads[i]:
        x_train.append(thread)
        y_train.append(i)
print(x_train, y_train)


'''

'''


