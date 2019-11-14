# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import collections
from allpairspy_master.allpairspy import AllPairs


# -------- PAIRWISE --------
parameters = [
    list(range(10))[::-1],#0 length
    list(range(10))[::-1],#1 diameter
    list(range(10))[::-1],#2 young
    list(range(10))[::-1],#3 density
    list(range(1, 10)),   #4 pressure_time
    list(range(1, 10)),   #5 pressure_radius
    list(range(1, 10)),   #6 pressure_amplitude
    list(range(10))[::-1],#7 strength
]
print("PAIRWISE:")
for i, pairs in enumerate(AllPairs(parameters)):
    print("{:3d}: {}".format(i, pairs))

pairs = list(AllPairs(parameters))
# -------- END PAIRWISE --------


with open('../11/fib_all_data.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))

Y = []
for i, val in enumerate(data_is_broken):
    Y.extend([i%2]*val)

def make_str(data):
    return ''.join(map(str, data))
def make_str_float(data):
    return ''.join(map(str, map(float, data)))
def make_set(data):
    return {make_str_float(i) for i in data}

pairs = make_set(pairs)
#print(pairs)


x_train, x_test, y_train, y_test = [], [], [], []

n = tuple(map(float, range(10)))
i = 0
a = np.empty((0,8), dtype=np.float64)
for i0 in n:
    for i1 in n:
        for i2 in n:
            for i3 in n:
                for i4 in n:
                    for i5 in n:
                        for i6 in n:
                            for i7 in n:
                                if 0 not in [i4, i5, i6]:
                                    l = [i0, i1, i2, i3, i4, i5, i6, i7]
                                    ls = make_str(l)
                                    if ls in pairs:
                                        x_train.append(l)
                                        y_train.append(Y[i])
                                    else:
                                        x_test.append(l)
                                        y_test.append(Y[i])
                                i += 1
    a = np.append(a, np.array(x_test), axis=0)
    x_test = []
    print(i0)
    #break

print('!!!')
x_train, x_test, y_train, y_test = np.array(x_train), a, np.array(y_train), np.array(y_test)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print('all', dict(collections.Counter(y_train)), dict(collections.Counter(y_test)))

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def fit_model(model):
    # fit model on training data
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print('y_pred', dict(collections.Counter(y_pred)))
    # make predictions for test data
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    #print(predictions)
    accuracy = accuracy_score(y_test, predictions)
    print("%.2f%% %s" % (accuracy * 100.0, model.__class__.__name__))

fit_model(XGBClassifier())
fit_model(LogisticRegression())
fit_model(LinearSVC(random_state=0, tol=1e-5))
fit_model(KNeighborsClassifier(n_neighbors=6))


'''
PAIRWISE:
  0: [9, 9, 9, 9, 1, 1, 1, 9]
  1: [8, 8, 8, 8, 2, 2, 2, 9]
  2: [7, 7, 7, 7, 3, 3, 3, 9]
  3: [6, 6, 6, 6, 4, 4, 4, 9]
  4: [5, 5, 5, 5, 5, 5, 5, 9]
  5: [4, 4, 4, 4, 6, 6, 6, 9]
  6: [3, 3, 3, 3, 7, 7, 7, 9]
  7: [2, 2, 2, 2, 8, 8, 8, 9]
  8: [1, 1, 1, 1, 9, 9, 9, 9]
  9: [0, 0, 0, 0, 9, 8, 7, 8]
 10: [0, 1, 2, 3, 6, 5, 4, 7]
 11: [1, 2, 3, 4, 5, 4, 3, 7]
 12: [2, 3, 4, 5, 4, 3, 2, 7]
 13: [3, 4, 5, 6, 3, 2, 1, 7]
 14: [4, 5, 6, 7, 2, 1, 9, 7]
 15: [5, 6, 7, 8, 1, 9, 8, 7]
 16: [6, 7, 8, 9, 8, 7, 6, 7]
 17: [7, 8, 9, 0, 7, 6, 5, 7]
 18: [8, 9, 0, 1, 7, 5, 8, 6]
 19: [9, 0, 1, 2, 6, 4, 2, 6]
 20: [9, 1, 3, 5, 3, 6, 8, 8]
 21: [8, 2, 4, 6, 1, 7, 9, 8]
 22: [7, 3, 5, 9, 9, 4, 8, 5]
 23: [6, 4, 1, 8, 5, 8, 4, 5]
 24: [5, 0, 2, 7, 4, 2, 6, 5]
 25: [4, 9, 8, 0, 4, 9, 3, 4]
 26: [3, 8, 7, 2, 5, 1, 6, 8]
 27: [2, 7, 6, 1, 1, 6, 7, 5]
 28: [1, 6, 0, 3, 2, 3, 5, 5]
 29: [0, 5, 9, 4, 8, 3, 1, 6]
 30: [6, 5, 4, 2, 3, 9, 7, 3]
 31: [0, 8, 3, 1, 4, 7, 5, 3]
 32: [1, 9, 5, 7, 6, 8, 2, 3]
 33: [2, 4, 9, 3, 9, 1, 3, 3]
 34: [3, 0, 8, 5, 1, 5, 9, 3]
 35: [4, 6, 2, 9, 7, 2, 3, 8]
 36: [5, 1, 0, 4, 2, 7, 4, 4]
 37: [7, 2, 6, 8, 6, 9, 1, 2]
 38: [8, 7, 1, 0, 8, 1, 5, 2]
 39: [9, 3, 7, 6, 8, 5, 7, 4]
 40: [4, 3, 9, 8, 3, 8, 6, 1]
 41: [6, 8, 0, 5, 6, 2, 9, 1]
 42: [9, 7, 2, 4, 9, 9, 2, 1]
 43: [8, 1, 5, 2, 4, 3, 7, 1]
 44: [7, 4, 3, 2, 2, 5, 1, 0]
 45: [5, 9, 6, 3, 3, 4, 9, 0]
 46: [3, 2, 1, 9, 2, 6, 4, 1]
 47: [2, 6, 8, 7, 5, 6, 1, 1]
 48: [1, 5, 7, 0, 7, 7, 2, 0]
 49: [0, 7, 4, 8, 7, 4, 1, 4]
 50: [4, 0, 3, 6, 9, 3, 4, 2]
 51: [0, 4, 6, 5, 8, 9, 6, 0]
 52: [1, 8, 2, 6, 1, 1, 8, 4]
 53: [2, 0, 5, 4, 7, 1, 9, 0]
 54: [7, 1, 4, 1, 5, 2, 2, 2]
 55: [9, 5, 0, 8, 4, 6, 3, 0]
 56: [6, 3, 2, 0, 2, 4, 5, 6]
 57: [3, 9, 4, 7, 9, 5, 5, 0]
 58: [5, 2, 9, 1, 6, 3, 7, 0]
 59: [8, 6, 9, 5, 9, 7, 1, 5]
 60: [6, 9, 7, 4, 1, 2, 4, 2]
 61: [9, 4, 8, 1, 7, 8, 4, 2]
 62: [7, 6, 1, 4, 3, 5, 6, 3]
 63: [2, 8, 1, 3, 8, 2, 4, 8]
 64: [0, 3, 8, 2, 1, 2, 3, 0]
 65: [4, 1, 7, 9, 5, 8, 5, 6]
 66: [8, 0, 6, 9, 3, 9, 5, 4]
 67: [5, 7, 3, 0, 6, 2, 8, 3]
 68: [3, 5, 2, 1, 8, 4, 3, 1]
 69: [1, 4, 6, 0, 4, 5, 9, 1]
 70: [5, 8, 4, 9, 9, 8, 3, 6]
 71: [4, 2, 0, 3, 1, 4, 2, 3]
 72: [2, 9, 3, 8, 8, 7, 9, 6]
 73: [7, 5, 8, 6, 6, 1, 4, 6]
 74: [6, 1, 9, 7, 7, 4, 8, 8]
 75: [1, 0, 9, 2, 2, 6, 9, 2]
 76: [0, 2, 5, 0, 3, 1, 2, 5]
 77: [3, 6, 5, 8, 6, 3, 8, 4]
 78: [8, 3, 6, 4, 5, 8, 8, 0]
 79: [9, 6, 4, 0, 2, 8, 6, 7]
 80: [2, 5, 1, 6, 7, 9, 6, 5]
 81: [1, 7, 5, 3, 4, 8, 1, 6]
 82: [6, 0, 7, 1, 2, 6, 1, 3]
 83: [7, 9, 2, 5, 5, 7, 7, 2]
 84: [8, 4, 7, 7, 1, 4, 6, 6]
 85: [0, 9, 1, 6, 2, 6, 8, 1]
 86: [4, 8, 5, 1, 3, 7, 6, 2]
 87: [5, 3, 1, 7, 8, 6, 2, 4]
 88: [9, 2, 6, 7, 9, 2, 5, 6]
 89: [3, 1, 6, 0, 1, 3, 3, 3]
 90: [1, 3, 0, 9, 6, 9, 4, 8]
 91: [2, 1, 0, 6, 5, 1, 5, 3]
 92: [6, 2, 8, 3, 9, 5, 6, 4]
 93: [3, 7, 9, 6, 5, 9, 2, 6]
 94: [4, 7, 1, 5, 2, 5, 3, 0]
 95: [7, 0, 0, 3, 8, 7, 8, 1]
 96: [2, 8, 7, 9, 6, 5, 9, 5]
 97: [5, 4, 8, 2, 7, 3, 5, 8]
 98: [8, 5, 3, 9, 1, 8, 2, 4]
 99: [1, 6, 4, 5, 8, 1, 7, 6]
100: [9, 8, 5, 7, 2, 9, 7, 8]
101: [8, 4, 2, 8, 6, 6, 3, 3]
102: [0, 6, 3, 7, 5, 3, 9, 2]
103: [9, 4, 0, 9, 4, 3, 8, 2]
104: [6, 1, 8, 8, 8, 1, 7, 5]
105: [6, 8, 5, 4, 1, 3, 3, 0]
106: [9, 5, 1, 3, 5, 7, 3, 2]
107: [2, 0, 4, 0, 5, 4, 4, 2]
108: [7, 3, 0, 2, 1, 8, 9, 4]
109: [4, 9, 1, 2, 1, 3, 7, 7]
110: [3, 3, 0, 4, 4, 1, 1, 8]
111: [1, 7, 0, 8, 3, 5, 4, 8]
112: [5, 4, 6, 2, 7, 7, 2, 8]
113: [5, 0, 2, 6, 3, 3, 3, 0]
114: [5, 5, 9, 5, 7, 2, 4, 4]
115: [5, 6, 3, 1, 1, 1, 5, 1]
116: [4, 2, 7, 5, 4, 4, 8, 1]
117: [4, 6, 0, 2, 9, 6, 4, 0]
118: [4, 5, 0, 7, 8, 7, 4, 3]
119: [1, 5, 8, 4, 3, 4, 8, 6]
120: [2, 0, 4, 8, 9, 5, 8, 7]
121: [2, 3, 0, 1, 3, 5, 2, 2]
122: [9, 8, 6, 1, 6, 4, 5, 8]
123: [9, 9, 0, 4, 7, 4, 7, 5]
124: [3, 1, 0, 9, 7, 8, 6, 0]
125: [8, 5, 4, 3, 9, 6, 4, 6]
126: [0, 4, 7, 9, 9, 3, 7, 3]
127: [9, 9, 5, 1, 8, 6, 4, 3]
128: [2, 9, 3, 1, 2, 9, 6, 4]
129: [4, 7, 0, 2, 7, 3, 6, 5]
130: [5, 5, 0, 0, 7, 2, 1, 9]
131: [1, 5, 4, 1, 7, 2, 6, 1]
132: [3, 5, 1, 1, 4, 2, 7, 2]
133: [6, 5, 3, 1, 7, 3, 2, 3]
134: [7, 5, 9, 1, 4, 5, 3, 8]
135: [5, 9, 2, 1, 6, 7, 9, 8]
136: [5, 4, 0, 1, 5, 3, 3, 4]
137: [5, 8, 0, 1, 7, 3, 1, 7]
138: [5, 3, 1, 4, 7, 3, 5, 3]
139: [5, 0, 1, 8, 7, 3, 5, 2]
140: [4, 0, 1, 5, 7, 8, 1, 9]
141: [9, 6, 4, 6, 7, 8, 2, 5]
142: [9, 7, 3, 3, 7, 9, 9, 5]
143: [3, 1, 2, 1, 7, 3, 1, 5]
144: [8, 2, 7, 3, 7, 3, 3, 5]
145: [8, 5, 0, 1, 7, 3, 3, 7]
146: [0, 5, 0, 1, 7, 3, 3, 9]

!!!
(147, 8) (72899853, 8) (147,) (72899853,)
all {1: 128, 0: 19} {1: 63518563, 0: 9381290}
y_pred {1: 66151464, 0: 6748389}
90.71% XGBClassifier
y_pred {1: 62697681, 0: 10202172}
92.42% LogisticRegression
y_pred {1: 63119441, 0: 9780412}
92.75% LinearSVC
y_pred {1: 68893575, 0: 4006278}
90.19% KNeighborsClassifier
'''
