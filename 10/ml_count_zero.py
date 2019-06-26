# -*- coding: utf-8 -*-


with open('data_45k/fib_all_data1.3.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))

X, Y = [], []

for i, val in enumerate(data_is_broken):
    Y.extend([i%2]*val)

print(Y.count(0)*100/10**8)
# 0 - 27%
# 1 - 72%
 