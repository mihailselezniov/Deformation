# -*- coding: utf-8 -*-
import collections
from allpairspy_master.allpairspy import AllPairs


# -------- PAIRWISE --------
parameters = [
    list(range(10)),   #0 length
    list(range(10)),   #1 diameter
    list(range(10)),   #2 young
    list(range(10)),   #3 density
    list(range(1, 10)),#4 pressure_time
    list(range(1, 10)),#5 pressure_radius
    list(range(1, 10)),#6 pressure_amplitude
    list(range(10)),   #7 strength
]
print("PAIRWISE:")
#for i, pairs in enumerate(AllPairs(parameters)):
#    print("{:3d}: {}".format(i, pairs))

pairs = list(AllPairs(parameters))
# -------- END PAIRWISE --------

for col in zip(*pairs):
    print(dict(collections.Counter(col)))


'''
PAIRWISE:
{0: 17, 1: 14, 2: 12, 3: 13, 4: 20, 5: 16, 6: 14, 7: 15, 8: 14, 9: 12}
{0: 15, 1: 14, 2: 13, 3: 14, 4: 22, 5: 15, 6: 14, 7: 12, 8: 13, 9: 15}
{0: 12, 1: 12, 2: 13, 3: 13, 4: 13, 5: 15, 6: 14, 7: 13, 8: 17, 9: 25}
{0: 14, 1: 14, 2: 13, 3: 13, 4: 13, 5: 14, 6: 14, 7: 14, 8: 25, 9: 13}
{1: 16, 2: 14, 3: 14, 4: 14, 5: 15, 6: 15, 7: 29, 8: 15, 9: 15}
{1: 14, 2: 15, 3: 26, 4: 16, 5: 16, 6: 15, 7: 15, 8: 16, 9: 14}
{1: 15, 2: 16, 3: 20, 4: 18, 5: 16, 6: 16, 7: 15, 8: 16, 9: 15}
{0: 12, 1: 16, 2: 13, 3: 15, 4: 16, 5: 14, 6: 17, 7: 16, 8: 13, 9: 15}
'''
