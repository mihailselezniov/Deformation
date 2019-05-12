# -*- coding: utf-8 -*-
import numpy as np
from os import walk

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


tasks = {}
f = []
for (dirpath, dirnames, filenames) in walk('./'):
    f.extend(filenames)
    break
fib_data_filenames = [i for i in f if 'fib_data' in i]
for filename in fib_data_filenames:
    with open(filename, 'r') as f:
        for line in f:
            task, val = line.split(':')
            tasks[task] = val

print(len(tasks))
all_num = 0
tmp = 0
for l in get_list(**par['length']):
    for di in get_list(**par['diameter']):
        for y in get_list(**par['young']):
            for de in get_list(**par['density']):
                for pt in get_list(**par['pressure_time']):
                    task = ','.join(map(str, (l, di, y, de, pt)))
                    print(task, tasks[task])
                    nums = tasks[task].split(',')
                    nums = list(map(int, nums))
                    if tmp:
                        nums[0] += tmp
                        tmp = 0
                    if len(nums) % 2:
                        nums, tmp = nums[:-1], nums[-1]
                    break
#                     with open('fib_all_data.txt', 'a') as f:
#                         for num in nums:
#                             f.write('{}\n'.format(num))
#                             all_num += num
# if tmp:
#     with open('fib_all_data.txt', 'a') as f:
#         f.write('{}\n'.format(tmp))

# print(all_num)



