# -*- coding: utf-8 -*-
import math


def gen_shift(rang):
    rs = []
    rs.append(rang)

    def dev2(rang):
        mid = (rang[1] - rang[0])/2 +rang[0]
        #print(mid)
        mids.append(mid)
        return mid, [rang[0], mid], [mid, rang[1]]

    def run_dev(rangs):
        rs = []
        for r in rangs:
            d = dev2(r)
            #print(d)
            rs.extend(d[1:])
        return rs

    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def mix(rs):
        rs2 = list(split(rs, 2))
        #print(rs2)
        rs2_pair = list(zip(*rs2))
        #print(rs2_pair)
        rs = []
        for r1, r2 in rs2_pair:
            rs.append(r1)
            rs.append(r2)
        return rs

    while 1:
        mids = []
        rs = run_dev(rs)
        #print([i[0] for i in rs])
        for mid in mids:
            yield mid
        count_mix = int(math.log2(len(rs))-1)
        #print(len(rs), 'mix', count_mix)
        for i in range(count_mix):
            rs = mix(rs)
        #print([i[0] for i in rs])


if __name__ == '__main__':
    rang = [0, 1]
    g = gen_shift(rang)
    for i in range(500):
        print(g.__next__())


"""
import math

rs = [[0, 1],]

def dev2(rang):
  mid = (rang[1] - rang[0])/2 +rang[0]
  print(mid)
  return mid, [rang[0], mid], [mid, rang[1]]

def run_dev(rangs):
  rs = []
  for r in rangs:
    d = dev2(r)
    #print(d)
    rs.extend(d[1:])
  return rs

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def mix(rs):
  rs2 = list(split(rs, 2))
  #print(rs2)
  rs2_pair = list(zip(*rs2))
  #print(rs2_pair)
  rs = []
  for r1, r2 in rs2_pair:
    rs.append(r1)
    rs.append(r2)
  return rs

for i in range(5):
  rs = run_dev(rs)
  print([i[0] for i in rs])

  count_mix = int(math.log2(len(rs))-1)
  print(len(rs), 'mix', count_mix)
  for i in range(count_mix):
    rs = mix(rs)
  print([i[0] for i in rs])
"""