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

# par['pressure_time'] = i1      # i1: 0 100
# par['pressure_radius'] = i2    # i2: 0 5
# par['pressure_amplitude'] = i3 # i3: 0 200
# par['length'] = i4             # i4: 10 100
# par['diameter'] = i5/100       # i5: 1 50
# par['young'] = i6              # i6: 60 300
# par['density'] = i7            # i7: 1000 2000
# par['strength'] = i8/10        # i8: 2 100

def get_list(Min, Max):
    return list(map(lambda x: round(x, 2), np.arange(Min, Max+0.01, (Max-Min)/(parts-1))))

# length, diameter, young, density, pressure_time, pressure_radius, pressure_amplitude, strength
# for l in get_list(**par['length']):
#     print(l)
# for di in get_list(**par['diameter']):
#     print(di)
# for y in get_list(**par['young']):
#     print(y)
# for de in get_list(**par['density']):
#     print(de)
# for pt in get_list(**par['pressure_time']):
#     print(pt)

# ---

# for pr in get_list(**par['pressure_radius']):
#     print(pr)
# for pa in get_list(**par['pressure_amplitude']):
#     print(pa)
# for s in get_list(**par['strength']):
#     print(s)

# all_input = set()

# for l in get_list(**par['length']):
#     for di in get_list(**par['diameter']):
#         for y in get_list(**par['young']):
#             for de in get_list(**par['density']):
#                 for pt in get_list(**par['pressure_time']):
#                     all_input.add(','.join(map(str, (l, di, y, de, pt, pr))))
# print(','.join(map(str, (l, di, y, de, pt, pr))))
# print(len(all_input))

print(', '.join(map(str, get_list(**par['pressure_radius']))))
print(', '.join(map(str, get_list(**par['pressure_amplitude']))))
print(', '.join(map(str, get_list(**par['strength']))))

