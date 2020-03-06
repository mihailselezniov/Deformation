# -*- coding: utf-8 -*-
import numpy as np


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

#orig_range()

new_parts = 19

def get_new_list(Min, Max):
    return list(map(lambda x: round(x, 2), np.arange(Min, Max+0.01, (Max-Min)/(new_parts-1))))[1::2]


def new_range():
    print('length', get_new_list(**par['length']))
    print('diameter', get_new_list(**par['diameter']))
    print('young', get_new_list(**par['young']))
    print('density', get_new_list(**par['density']))
    print('pressure_time', get_new_list(**par['pressure_time']))
    print('pressure_radius', get_new_list(**par['pressure_radius']))
    print('pressure_amplitude', get_new_list(**par['pressure_amplitude']))
    print('strength', get_new_list(**par['strength']))

#new_range()


def new_range2():
    #pressure_radius := [10]float64{0.0, 0.56, 1.11, 1.67, 2.22, 2.78, 3.33, 3.89, 4.44, 5.0}
    print('length := [9]float64{', ', '.join(list(map(str, get_new_list(**par['length'])))), '}')
    print('diameter := [9]float64{', ', '.join(list(map(str, get_new_list(**par['diameter'])))), '}')
    print('young := [9]float64{', ', '.join(list(map(str, get_new_list(**par['young'])))), '}')
    print('density := [9]float64{', ', '.join(list(map(str, get_new_list(**par['density'])))), '}')
    print('pressure_time := [9]float64{', ', '.join(list(map(str, get_new_list(**par['pressure_time'])))), '}')
    print('pressure_radius := [9]float64{', ', '.join(list(map(str, get_new_list(**par['pressure_radius'])))), '}')
    print('pressure_amplitude := [9]float64{', ', '.join(list(map(str, get_new_list(**par['pressure_amplitude'])))), '}')
    print('strength := [9]float64{', ', '.join(list(map(str, get_new_list(**par['strength'])))), '}')

new_range2()
'''

length [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
diameter [0.01, 0.06, 0.12, 0.17, 0.23, 0.28, 0.34, 0.39, 0.45, 0.5]
young [60.0, 86.67, 113.33, 140.0, 166.67, 193.33, 220.0, 246.67, 273.33, 300.0]
density [1000.0, 1111.11, 1222.22, 1333.33, 1444.44, 1555.56, 1666.67, 1777.78, 1888.89, 2000.0]
pressure_time [0.0, 11.11, 22.22, 33.33, 44.44, 55.56, 66.67, 77.78, 88.89, 100.0]
pressure_radius [0.0, 0.56, 1.11, 1.67, 2.22, 2.78, 3.33, 3.89, 4.44, 5.0]
pressure_amplitude [0.0, 22.22, 44.44, 66.67, 88.89, 111.11, 133.33, 155.56, 177.78, 200.0]
strength [0.2, 1.29, 2.38, 3.47, 4.56, 5.64, 6.73, 7.82, 8.91, 10.0]

length [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0]
diameter [0.04, 0.09, 0.15, 0.2, 0.25, 0.31, 0.36, 0.42, 0.47]
young [73.33, 100.0, 126.67, 153.33, 180.0, 206.67, 233.33, 260.0, 286.67]
density [1055.56, 1166.67, 1277.78, 1388.89, 1500.0, 1611.11, 1722.22, 1833.33, 1944.44]
pressure_time [5.56, 16.67, 27.78, 38.89, 50.0, 61.11, 72.22, 83.33, 94.44]
pressure_radius [0.28, 0.83, 1.39, 1.94, 2.5, 3.06, 3.61, 4.17, 4.72]
pressure_amplitude [11.11, 33.33, 55.56, 77.78, 100.0, 122.22, 144.44, 166.67, 188.89]
strength [0.74, 1.83, 2.92, 4.01, 5.1, 6.19, 7.28, 8.37, 9.46]


length := [9]float64{ 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0 }
diameter := [9]float64{ 0.04, 0.09, 0.15, 0.2, 0.25, 0.31, 0.36, 0.42, 0.47 }
young := [9]float64{ 73.33, 100.0, 126.67, 153.33, 180.0, 206.67, 233.33, 260.0, 286.67 }
density := [9]float64{ 1055.56, 1166.67, 1277.78, 1388.89, 1500.0, 1611.11, 1722.22, 1833.33, 1944.44 }
pressure_time := [9]float64{ 5.56, 16.67, 27.78, 38.89, 50.0, 61.11, 72.22, 83.33, 94.44 }
pressure_radius := [9]float64{ 0.28, 0.83, 1.39, 1.94, 2.5, 3.06, 3.61, 4.17, 4.72 }
pressure_amplitude := [9]float64{ 11.11, 33.33, 55.56, 77.78, 100.0, 122.22, 144.44, 166.67, 188.89 }
strength := [9]float64{ 0.74, 1.83, 2.92, 4.01, 5.1, 6.19, 7.28, 8.37, 9.46 }
'''






