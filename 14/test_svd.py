# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import svd

a = [
[0, 0],
[0, 1],
[1, 0],
[1, 1]]

b = [
[1, 0, 0, 0],
[0, 1, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, 1]]

b = np.array(b)
u, s, vh = svd(b)

print(u)
print(vh)
'''
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
 
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
'''

a = [
[0, 0, 0],
[0, 0, 1],
[0, 1, 0],
[0, 1, 1],
[1, 0, 0],
[1, 0, 1],
[1, 1, 0],
[1, 1, 1]]

b = [
[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
[0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]
b = np.array(b)
u, s, vh = svd(b)

#print(u)
for i in u:
    print(list(map(lambda x: round(x, 2), list(i))))

'''
[-0.35, -0.0, 0.0, 0.61, 0.0, 0.61, -0.0, -0.35]
[-0.35, -0.47, 0.33, 0.2, 0.44, -0.2, 0.38, 0.35]
[-0.35, 0.52, 0.25, 0.2, 0.11, -0.2, -0.57, 0.35]
[-0.35, 0.05, 0.58, -0.2, -0.55, -0.2, 0.19, -0.35]
[-0.35, -0.05, -0.58, 0.2, -0.55, -0.2, 0.19, 0.35]
[-0.35, -0.52, -0.25, -0.2, 0.11, -0.2, -0.57, -0.35]
[-0.35, 0.47, -0.33, -0.2, 0.44, -0.2, 0.38, -0.35]
[-0.35, -0.0, 0.0, -0.61, 0.0, 0.61, 0.0, 0.35]
'''


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
x = enc.fit_transform(a).toarray()
print(x.shape)
#print(x)

'''
(8, 6)
[[1. 0. 1. 0. 1. 0.]
 [1. 0. 1. 0. 0. 1.]
 [1. 0. 0. 1. 1. 0.]
 [1. 0. 0. 1. 0. 1.]
 [0. 1. 1. 0. 1. 0.]
 [0. 1. 1. 0. 0. 1.]
 [0. 1. 0. 1. 1. 0.]
 [0. 1. 0. 1. 0. 1.]]
 '''
u, s, vh = svd(x)
for i in u:
    print(list(map(lambda x: round(x, 2), list(i))))




