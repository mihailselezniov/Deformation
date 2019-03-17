# -*- coding: utf-8 -*-
import csv
from fiber_test2 import Test_drive_fiber

params = {'length': 10,
          'diameter': 0.1,
          'young': 125,
          'density': 1430,
          'strength': 4,
          'pressure_time': 10,
          'pressure_radius': 1,
          'pressure_amplitude': 25,
          'POINTS_PER_FIBER': 200,
          'NUMBER_OF_FRAMES': 3000}
is_broken = []


# get all params
def all_params(d):
    p = params.copy()
    p.update(d)
    return p

def save(data):
    fields = [data[key] for key in sorted(data.keys())]
    with open('data2.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)



def process_data(list_params):
    for p in list_params:
        results = Test_drive_fiber(p).get_results()
        save(results)
        is_broken.append(results['is_broken'])
        print(len(is_broken))



# list_params = [all_params({'pressure_time': i}) for i in range(1, 20)]
# list_params = [all_params({'pressure_radius': i}) for i in range(1, 20)]
# list_params = [all_params({'pressure_amplitude': i}) for i in range(10, 30)]
# list_params = [all_params({'length': i}) for i in range(1, 20)]
# list_params = [all_params({'diameter': i/100}) for i in range(1, 20)]
# list_params = [all_params({'young': i}) for i in range(115, 135)]
# list_params = [all_params({'density': i}) for i in range(1420, 1440)]
# list_params = [all_params({'strength': i}) for i in range(1, 10)]
# process_data(list_params)
# if len(is_broken) > 1:
#     print('^_^' if xor(is_broken) else '=(')

step = 9
par = {}
list_params = []
for i1 in range(1, 21, step):
    par['pressure_time'] = i1
    for i2 in range(1, 21, step):
        par['pressure_radius'] = i2
        for i3 in range(10, 31, step):
            par['pressure_amplitude'] = i3
            for i4 in range(1, 21, step):
                par['length'] = i4
                for i5 in range(1, 21, step):
                    par['diameter'] = i5/100
                    for i6 in range(115, 136, step):
                        par['young'] = i6
                        for i7 in range(1420, 1441, step):
                            par['density'] = i7
                            for i8 in range(1, 11, step):
                                par['strength'] = i8
                                list_params.append(all_params(par))

max_ = len(list_params)
print(max_)#4374
process_data(list_params)
