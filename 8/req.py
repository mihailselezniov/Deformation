import requests
from time import time as t

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

url = 'https://mipt{}-mihailselezniov.c9users.io/test_drive_fiber'.format(0)

t1 = t()
#print({'params': str([params]*10)})
r = requests.post(url, json={'params': str([params]*10)})
print(t()-t1)