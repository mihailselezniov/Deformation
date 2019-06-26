import requests
from time import time as t
from time import gmtime, strftime, sleep
import os

filename = os.path.basename(__file__)
num = filename.split('.')[0].split('worker')[1]

url = 'https://mipt{}-mihailselezniov.c9users.io/'.format(num)
url_task = 'http://127.0.0.1:5000/'

retry = 0
t1 = t()
while 1:
    if retry:
        task = retry
    else:
        r = requests.get(url_task)
        task = r.text

    try:
        r = requests.get(url + task)
    except:
        retry = task
        print('X', end='', flush=True)
        sleep(5)
        continue
        
    if r.status_code == 200 and r.text:
        with open('fib_data{}.txt'.format(num), 'a') as f:
            f.write('{}:{}\n'.format(task, r.text))
        retry = 0
        print(strftime("\n%d %H:%M:%S", gmtime()), round((t()-t1)/60, 1), 'min')
        t1 = t()
    elif r.status_code != 200:
        retry = task
        print('X', end='', flush=True)
        sleep(5)
    else:
        retry = task
        print('.', end='', flush=True)
        sleep(5)
    
