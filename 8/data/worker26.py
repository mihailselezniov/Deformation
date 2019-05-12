import requests
from time import time as t
from time import gmtime, strftime, sleep

num = 26
url = 'https://mipt{}-mihailselezniov.c9users.io/'.format(num)
url_task = 'http://127.0.0.1:5000/'

retry = 0

while 1:
    if retry:
        task = retry
    else:
        r = requests.get(url_task)
        task = r.text
    t1 = t()
    r = requests.get(url + task)
    if r.status_code == 200:
        with open('fib_data{}.txt'.format(num), 'a') as f:
            f.write('{}:{}\n'.format(task, r.text))
        retry = 0
    else:
        retry = task
        print('=(')
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()), round(t()-t1), 'sec')
    sleep(10)
