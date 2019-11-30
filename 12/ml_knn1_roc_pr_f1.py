import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import collections
import math
import sys
import time
import pickle
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


ml_type = int(sys.argv[1])

f_name = '4_{}.txt'.format(ml_type)
metrics_f = 'metrics/' + f_name
source_f = 'ml_threads/' + f_name

X = joblib.load('border_vars/X.j')
Y = joblib.load('border_vars/Y.j')
print(X.shape, Y.shape)

model = ''
if ml_type == 1:
    model = XGBClassifier()
elif ml_type == 2:
    model = LogisticRegression()
elif ml_type == 3:
    model = LinearSVC(random_state=0, tol=1e-5)
elif ml_type == 4:
    model = KNeighborsClassifier(n_neighbors=6)

with open(source_f, 'r') as f:
    threads = f.readlines()

x_train, y_train = [], []
for t in threads:
    tr = list(map(int, t.replace('\n', '').split(',')))
    x_train.append(tr[:-1])
    y_train.append(tr[-1])
#print(x_train, y_train)

with open(metrics_f, 'r') as f:
    m = f.readlines()
start_cut_from = len(m)
#print(len(threads), 2+start_from)

for cut in range(2 + start_cut_from, len(threads) + 1):
    x, y = np.array(x_train[:cut]), np.array(y_train[:cut])
    #print(x, y)
    model.fit(x, y)
    y_pred = model.predict(X)

    row_metrics = []

    roc = roc_auc_score(Y, y_pred)
    row_metrics.append(roc)

    pr = average_precision_score(Y, y_pred)
    row_metrics.append(pr)

    f1 = f1_score(Y, y_pred, average=None)
    row_metrics.append(f1[0])

    row_metrics = ','.join(list(map(lambda x: str(round(float(x), 4)), row_metrics)))
    print('#{} ({}) {}'.format(ml_type, cut, row_metrics))

    with open(metrics_f, 'a') as f:
        f.write(row_metrics + '\n')

    #break


'''
'''








