# -*- coding: utf-8 -*-
import numpy as np
from os import walk
from flask import Flask
app = Flask(__name__)

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

tasks = set()

for l in get_list(**par['length']):
    for di in get_list(**par['diameter']):
        for y in get_list(**par['young']):
            for de in get_list(**par['density']):
                for pt in get_list(**par['pressure_time']):
                    tasks.add(','.join(map(str, (l, di, y, de, pt))))

print(len(tasks))

from os import walk

f = []
for (dirpath, dirnames, filenames) in walk('./'):
    f.extend(filenames)
    break
fib_data_filenames = [i for i in f if 'fib_data' in i]
for filename in fib_data_filenames:
    with open(filename, 'r') as f:
        for line in f:
            task = line.split(':')[0]
            tasks.remove(task)

print(len(tasks))

@app.route("/")
def get_task():
    print(len(tasks))
    return tasks.pop()

if __name__ == "__main__":
   app.run()
