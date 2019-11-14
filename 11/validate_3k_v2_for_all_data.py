
with open('fib_all_data.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))

def make_str(data):
    return ''.join(map(str, data))

l, Y = set(), []
for i, val in enumerate(data_is_broken):
    Y.extend([i%2]*val)
n = list(range(10))
i = 0
a, b = 0, 0
for i0 in n:
    for i1 in n:
        for i2 in n:
            for i3 in n:
                for i4 in n:
                    for i5 in n:
                        for i6 in n:
                            for i7 in n:
                                if 0 not in [i4, i5, i6]:
                                    if not Y[i]:
                                        b += 1
                                        l.add(make_str([i0, i1, i2, i3, i4, i5, i6, i7]))
                                    a += 1
                                i += 1
    print(i0)
print(a, b)


#def cut_pressure_time(data):
#    return list(map(lambda i: [i[0],i[1],i[2],i[3],i[7]], data))


def make_set(data):
    return {make_str(i) for i in data}
def make_list(data):
    return list(map(int, data))

ls = l
l = list(ls)

print(len(l))

shift = []
n = 2
r = list(range(n))
for i0 in r:
    for i1 in r:
        for i2 in r:
            for i3 in r:
                for i4 in r:
                    for i5 in r:
                        for i6 in r:
                            for i7 in r:
                                shift.append([i0, i1, i2, i3, i4, i5, i6, i7])
del shift[0]
print(len(shift))
shift = [i for i in shift if sum(i) == 1]
shift.extend([list(map(lambda x: -x, i)) for i in shift])

print(len(shift))

def get_ways(l0):
    ways = []
    for i in shift:
        ways.append(list(map(lambda x: abs(sum(x)), zip(i, l0))))
    return [i for i in ways if (10 not in i) and (11 not in i)]


all_ways = set()
for i in l:
    all_ways.update(make_set(get_ways(make_list(i))))

result = ls.intersection(all_ways)
print(len(ls))
print(len(result))
print(len(all_ways))

'''
without cut
72900000 9381309
9381309
255
16
9381309
9381276
24097874

cut pressure time
1762216
127
14
1762216
1762211
3667983

cut all pressure params
77693
31
10
77693
77693
94205
'''


