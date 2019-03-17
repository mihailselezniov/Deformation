# -*- coding: utf-8 -*-
import collections
import csv
import grequests

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
number_of_servers = 5
# server for data processing
serv_urls = ['https://mipt{}-mihailselezniov.c9users.io/test_drive_fiber'.format(i) for i in range(number_of_servers)]

# split list into smaller lists
split_list = lambda A, n=3: [A[i:i+n] for i in range(0, len(A), n)]
# xor of element of a list/tuple
xor = lambda bit: 1 if len(set(bit))>1 else 0 
# get all params
def all_params(d):
    p = params.copy()
    p.update(d)
    return p

def save(data):
    fields = [data[key] for key in sorted(data.keys())]
    with open('data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)



def process_data(list_params):
    for cup_params in split_list(list_params, number_of_servers):
        rs = (grequests.post(url, json=param) for param, url in zip(cup_params, serv_urls))
        res = grequests.map(rs)
        for r in res:
            if r.ok:
                results = r.json()['results']
                save(results)
                is_broken.append(results['is_broken'])
                #print('.', end='')
            else:
                print(r.status_code)
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

if __name__ == '__main__':
    step = 10
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
                                for i8 in range(1, 21, step):
                                    par['strength'] = i8
                                    list_params.append(all_params(par))

    print(len(list_params))
    process_data(list_params)



