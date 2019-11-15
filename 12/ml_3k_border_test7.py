import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.externals import joblib
import collections
import copy
import math
import pickle
import sys


ml_type = int(sys.argv[1])


def make_str(data):
    return ''.join(str(i) for i in data)
def make_set(data):
    return {make_str(i) for i in data}
def make_list(data):
    return [int(i) for i in data]

way_dict = joblib.load('border_vars/way_dict.j')
zero_keys = joblib.load('border_vars/zero_keys.j')
X_key_indexs = joblib.load('border_vars/X_key_indexs.j')
X = joblib.load('border_vars/X.j')
Y = joblib.load('border_vars/Y.j')
'''
with open('border_vars/way_dict.p', 'rb') as f:
    way_dict = pickle.load(f)
with open('border_vars/zero_keys.p', 'rb') as f:
    zero_keys = pickle.load(f)
with open('border_vars/X_key_indexs.p', 'rb') as f:
    X_key_indexs = pickle.load(f)
with open('border_vars/X.p', 'rb') as f:
    X = pickle.load(f)
with open('border_vars/Y.p', 'rb') as f:
    Y = pickle.load(f)
'''
print(X.shape, Y.shape)
print('all', dict(collections.Counter(Y)))

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

threads = {
    0: [[2,5,6,7,2,5,1,9],[3,9,2,8,2,6,1,6],[4,9,7,3,3,3,3,9],[5,7,9,6,1,4,2,6],[6,5,4,3,3,2,1,2],[7,6,5,4,4,3,2,2],[9,6,1,7,1,2,3,9]],
    1: [[0,0,0,0,1,1,1,0],[2,0,7,4,5,7,7,7],[4,0,3,6,3,4,9,9],[5,8,2,0,5,8,5,3],[6,4,8,8,4,2,7,7],[7,5,0,6,9,1,3,6],[8,4,1,5,3,4,8,3]]
}

for _ in range(715):#72-14=58-1=57 729-14=715-1=714
    print('#', _)
    x_train, y_train = [], []

    for i in threads:
        for thread in threads[i]:
            x_train.append(thread)
            y_train.append(i)

    x_train, y_train = np.array(x_train), np.array(y_train)

    #print(x_train, y_train)

    if ml_type == 1:
        model = XGBClassifier()
    elif ml_type == 2:
        model = LogisticRegression()
    elif ml_type == 3:
        model = LinearSVC(random_state=0, tol=1e-5)
    elif ml_type == 4:
        model = KNeighborsClassifier(n_neighbors=6)

    model.fit(x_train, y_train)
    y_pred = model.predict(X)
    accuracy = accuracy_score(Y, y_pred)
    print('Accuracy: {}'.format(accuracy))
    #print(y_pred)
    print(dict(collections.Counter(y_pred)))

    if _ in [0, 57, 714]:
        print('')
        cm = confusion_matrix(Y, y_pred)
        print('Confusion matrix:\n{}'.format(cm))

        print('Precision, recall and f1-score:')
        print(classification_report(Y, y_pred))

        roc = roc_auc_score(Y, y_pred)
        print('ROC AUC: {}'.format(roc))

        pr = average_precision_score(Y, y_pred)
        print('Precision-recall: {}'.format(pr))
        print('')


    find = 1 if len(threads[0]) > len(threads[1]) else 0



    #print(get_ways([8,8,8,8,2]))

    find_set, border_threads = set(), set()
    find_set = {i for i in range(len(y_pred)) if y_pred[i] == find}
    #print('find_set', len(find_set), len(y_pred))

    for i in range(len(y_pred)):
        if y_pred[i] != find:
            border_threads.update(set(way_dict[i]).intersection(find_set))

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
    #print(threads)
    #break

print('')
x_train, y_train = [], []
for i in threads:
    for thread in threads[i]:
        x_train.append(thread)
        y_train.append(i)
print('x_train =', x_train)
print('y_train =', y_train)


'''

'''


