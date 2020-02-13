# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import collections


from numpy import genfromtxt


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

def orig_range():
    print('length', get_list(**par['length']))
    print('diameter', get_list(**par['diameter']))
    print('young', get_list(**par['young']))
    print('density', get_list(**par['density']))
    print('pressure_time', get_list(**par['pressure_time']))
    print('pressure_radius', get_list(**par['pressure_radius']))
    print('pressure_amplitude', get_list(**par['pressure_amplitude']))
    print('strength', get_list(**par['strength']))



def make_test(num):
    data = genfromtxt('data0{}.csv'.format(num), delimiter=';', skip_header=True)
    x_test = []
    for d in data:
        #pressure_time;pressure_radius;pressure_amplitude;young;density;strength;length;diameter;is_broken
        pt, pr, pa, y, de, s, l, di, b = list(map(float, d))
        x_test.append([l, di, y, de, pt, pr, pa, s])
    x_test = np.array(x_test)
    mi = list(map(float, x_test.min(axis=0)))
    ma = list(map(float, x_test.max(axis=0)))
    print('length', '{} - {}'.format(mi[0], ma[0]))
    print('diameter', '{} - {}'.format(mi[1], ma[1]))
    print('young', '{} - {}'.format(mi[2], ma[2]))
    print('density', '{} - {}'.format(mi[3], ma[3]))
    print('pressure_time', '{} - {}'.format(mi[4], ma[4]))
    print('pressure_radius', '{} - {}'.format(mi[5], ma[5]))
    print('pressure_amplitude', '{} - {}'.format(mi[6], ma[6]))
    print('strength', '{} - {}'.format(mi[7], ma[7]))





for num in range(1, 6):
    print('\n\n\n', '#'*5, num, '#'*5)
    orig_range()
    make_test(num)



'''
 ##### 1 #####
length [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
diameter [0.01, 0.06, 0.12, 0.17, 0.23, 0.28, 0.34, 0.39, 0.45, 0.5]
young [60.0, 86.67, 113.33, 140.0, 166.67, 193.33, 220.0, 246.67, 273.33, 300.0]
density [1000.0, 1111.11, 1222.22, 1333.33, 1444.44, 1555.56, 1666.67, 1777.78, 1888.89, 2000.0]
pressure_time [0.0, 11.11, 22.22, 33.33, 44.44, 55.56, 66.67, 77.78, 88.89, 100.0]
pressure_radius [0.0, 0.56, 1.11, 1.67, 2.22, 2.78, 3.33, 3.89, 4.44, 5.0]
pressure_amplitude [0.0, 22.22, 44.44, 66.67, 88.89, 111.11, 133.33, 155.56, 177.78, 200.0]
strength [0.2, 1.29, 2.38, 3.47, 4.56, 5.64, 6.73, 7.82, 8.91, 10.0]
length 40.148557 - 99.935844
diameter 0.100021 - 0.199871
young 125.0 - 125.0
density 1430.0 - 1430.0
pressure_time 1.027277 - 9.97269
pressure_radius 0.111137 - 2.49645
pressure_amplitude 1.01202 - 9.987218
strength 4.0 - 4.0



 ##### 2 #####
length [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
diameter [0.01, 0.06, 0.12, 0.17, 0.23, 0.28, 0.34, 0.39, 0.45, 0.5]
young [60.0, 86.67, 113.33, 140.0, 166.67, 193.33, 220.0, 246.67, 273.33, 300.0]
density [1000.0, 1111.11, 1222.22, 1333.33, 1444.44, 1555.56, 1666.67, 1777.78, 1888.89, 2000.0]
pressure_time [0.0, 11.11, 22.22, 33.33, 44.44, 55.56, 66.67, 77.78, 88.89, 100.0]
pressure_radius [0.0, 0.56, 1.11, 1.67, 2.22, 2.78, 3.33, 3.89, 4.44, 5.0]
pressure_amplitude [0.0, 22.22, 44.44, 66.67, 88.89, 111.11, 133.33, 155.56, 177.78, 200.0]
strength [0.2, 1.29, 2.38, 3.47, 4.56, 5.64, 6.73, 7.82, 8.91, 10.0]
length 40.174803 - 99.981659
diameter 0.100168 - 0.19991
young 125.0 - 125.0
density 1430.0 - 1430.0
pressure_time 1.001345 - 9.996456
pressure_radius 0.101028 - 4.996122
pressure_amplitude 100.116967 - 199.899581
strength 4.0 - 4.0



 ##### 3 #####
length [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
diameter [0.01, 0.06, 0.12, 0.17, 0.23, 0.28, 0.34, 0.39, 0.45, 0.5]
young [60.0, 86.67, 113.33, 140.0, 166.67, 193.33, 220.0, 246.67, 273.33, 300.0]
density [1000.0, 1111.11, 1222.22, 1333.33, 1444.44, 1555.56, 1666.67, 1777.78, 1888.89, 2000.0]
pressure_time [0.0, 11.11, 22.22, 33.33, 44.44, 55.56, 66.67, 77.78, 88.89, 100.0]
pressure_radius [0.0, 0.56, 1.11, 1.67, 2.22, 2.78, 3.33, 3.89, 4.44, 5.0]
pressure_amplitude [0.0, 22.22, 44.44, 66.67, 88.89, 111.11, 133.33, 155.56, 177.78, 200.0]
strength [0.2, 1.29, 2.38, 3.47, 4.56, 5.64, 6.73, 7.82, 8.91, 10.0]
length 80.0 - 80.0
diameter 0.1 - 0.1
young 80.016319 - 179.860993
density 1200.973301 - 1599.97569
pressure_time 1.012113 - 9.993897
pressure_radius 0.103093 - 2.496386
pressure_amplitude 1.005076 - 9.997656
strength 2.000386 - 5.996605



 ##### 4 #####
length [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
diameter [0.01, 0.06, 0.12, 0.17, 0.23, 0.28, 0.34, 0.39, 0.45, 0.5]
young [60.0, 86.67, 113.33, 140.0, 166.67, 193.33, 220.0, 246.67, 273.33, 300.0]
density [1000.0, 1111.11, 1222.22, 1333.33, 1444.44, 1555.56, 1666.67, 1777.78, 1888.89, 2000.0]
pressure_time [0.0, 11.11, 22.22, 33.33, 44.44, 55.56, 66.67, 77.78, 88.89, 100.0]
pressure_radius [0.0, 0.56, 1.11, 1.67, 2.22, 2.78, 3.33, 3.89, 4.44, 5.0]
pressure_amplitude [0.0, 22.22, 44.44, 66.67, 88.89, 111.11, 133.33, 155.56, 177.78, 200.0]
strength [0.2, 1.29, 2.38, 3.47, 4.56, 5.64, 6.73, 7.82, 8.91, 10.0]
length 80.0 - 80.0
diameter 0.1 - 0.1
young 80.005387 - 179.994769
density 1200.006491 - 1599.934591
pressure_time 1.010031 - 9.982748
pressure_radius 0.103978 - 4.993744
pressure_amplitude 100.049927 - 199.916539
strength 2.001763 - 5.998873



 ##### 5 #####
length [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
diameter [0.01, 0.06, 0.12, 0.17, 0.23, 0.28, 0.34, 0.39, 0.45, 0.5]
young [60.0, 86.67, 113.33, 140.0, 166.67, 193.33, 220.0, 246.67, 273.33, 300.0]
density [1000.0, 1111.11, 1222.22, 1333.33, 1444.44, 1555.56, 1666.67, 1777.78, 1888.89, 2000.0]
pressure_time [0.0, 11.11, 22.22, 33.33, 44.44, 55.56, 66.67, 77.78, 88.89, 100.0]
pressure_radius [0.0, 0.56, 1.11, 1.67, 2.22, 2.78, 3.33, 3.89, 4.44, 5.0]
pressure_amplitude [0.0, 22.22, 44.44, 66.67, 88.89, 111.11, 133.33, 155.56, 177.78, 200.0]
strength [0.2, 1.29, 2.38, 3.47, 4.56, 5.64, 6.73, 7.82, 8.91, 10.0]
length 40.05932 - 99.940032
diameter 0.10011 - 0.199931
young 80.024801 - 179.918208
density 1200.216943 - 1598.72198
pressure_time 1.001289 - 9.989583
pressure_radius 0.106349 - 4.994131
pressure_amplitude 1.038218 - 199.934732
strength 2.003763 - 5.994421
'''


