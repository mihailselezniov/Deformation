from shutil import copyfile

for i in range(71, 101):
    copyfile('worker0.py', 'worker{}.py'.format(i))