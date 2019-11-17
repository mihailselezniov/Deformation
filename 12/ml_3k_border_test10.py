import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import collections
import math
import sys


ml_type = int(sys.argv[1])

'''
threads = {
    0: [[2,5,6,7,2,5,1,9],[3,9,2,8,2,6,1,6],[4,9,7,3,3,3,3,9],[5,7,9,6,1,4,2,6],[6,5,4,3,3,2,1,2],[7,6,5,4,4,3,2,2],[9,6,1,7,1,2,3,9]],
    1: [[0,0,0,0,1,1,1,0],[2,0,7,4,5,7,7,7],[4,0,3,6,3,4,9,9],[5,8,2,0,5,8,5,3],[6,4,8,8,4,2,7,7],[7,5,0,6,9,1,3,6],[8,4,1,5,3,4,8,3]]
}
joblib.dump(threads, 'threads_{}.j'.format(ml_type))
'''
threads_f = 'threads/threads_{}'.format(ml_type)
threads_f_name = '{}.j'.format(threads_f)
threads72_f_name = '{}_72.j'.format(threads_f)
threads729_f_name = '{}_729.j'.format(threads_f)
threads = joblib.load(threads_f_name)
len_threads = len(threads[0]) + len(threads[1])
print('threads', len_threads)

way_dict = joblib.load('border_vars/way_dict.j')
X = joblib.load('border_vars/X.j')
Y = joblib.load('border_vars/Y.j')

print(X.shape, Y.shape)
print('all', dict(collections.Counter(Y)))

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


x_train, y_train = [], []
model = ''
def gen_train():
    global x_train
    global y_train
    x_train, y_train = [], []
    for i in threads:
        for thread in threads[i]:
            x_train.append(thread)
            y_train.append(i)
    x_train, y_train = np.array(x_train), np.array(y_train)

while len_threads < 729:
    print('#', len_threads)
    gen_train()

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

    find = 1 if len(threads[0]) > len(threads[1]) else 0

    def distance(l0):
        return min([math.sqrt(sum(map(lambda x: (x[0] - x[1])**2, zip(thread, l0)))) for thread in threads[find]])

    t_id = 0
    max_distance = 0
    max_dis_thread = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            for j in way_dict[i]:
                if y_pred[j] == 1:
                    t_id = j if find else i
                    dis = distance(list(X[t_id]))
                    if dis > max_distance:
                        max_distance = dis
                        max_dis_thread = t_id

    print([max_distance, max_dis_thread])
    print(Y[max_dis_thread])
    threads[Y[max_dis_thread]].append(list(X[max_dis_thread]))
    len_threads = len(threads[0]) + len(threads[1])
    if len_threads == 72:
        joblib.dump(threads, threads72_f_name)
    if len_threads == 729:
        joblib.dump(threads, threads729_f_name)
    joblib.dump(threads, threads_f_name)
    #print(threads)
    #break



