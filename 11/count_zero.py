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

with open('data45k.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))

X, Y, Z = [], [], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

for i, val in enumerate(data_is_broken):
    Y.extend([i%2]*val)

i = 0
for i0, l in enumerate(get_list(**par['length'])):
    for i1, di in enumerate(get_list(**par['diameter'])):
        for i2, y in enumerate(get_list(**par['young'])):
            for i3, de in enumerate(get_list(**par['density'])):
                for i4, pt in enumerate(get_list(**par['pressure_time'])):
                    for i5, pr in enumerate(get_list(**par['pressure_radius'])):
                        for i6, pa in enumerate(get_list(**par['pressure_amplitude'])):
                            for i7, s in enumerate(get_list(**par['strength'])):
                                if not Y[i]:
                                    Z[0][i0] += 1
                                    Z[1][i1] += 1
                                    Z[2][i2] += 1
                                    Z[3][i3] += 1
                                    Z[4][i4] += 1
                                    Z[5][i5] += 1
                                    Z[6][i6] += 1
                                    Z[7][i7] += 1
                                i += 1
    print(i0)
print(Z)
#[[2720416, 2714462, 2715565, 2715269, 2710004, 2715333, 2720025, 2728869, 2723898, 2729683], [2705257, 2715066, 2709424, 2713353, 2712945, 2709375, 2720094, 2724958, 2733360, 2749692], [2717460, 2712604, 2706931, 2715676, 2708920, 2713999, 2724080, 2732766, 2730318, 2730770], [2712466, 2714880, 2727342, 2722086, 2717589, 2712472, 2707480, 2724256, 2731091, 2723862], [9869785, 1935140, 1925353, 1930085, 1934241, 1923986, 1921614, 1918010, 1916669, 1918641], [10000000, 1969487, 1906525, 1903128, 1903122, 1902640, 1902416, 1902219, 1902055, 1901932], [10000000, 1980669, 1902775, 1901440, 1901440, 1901440, 1901440, 1901440, 1901440, 1901440], [2711315, 2720103, 2720221, 2720245, 2720261, 2720268, 2720269, 2720275, 2720282, 2720285]]



