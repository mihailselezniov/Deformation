#import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#sns.set(style="ticks", color_codes=True)

with open('../../11/fib_all_data.txt', 'r') as f:
    data_is_broken = f.readlines()
data_is_broken = list(map(int, data_is_broken))

X, Y = [], []
for i, val in enumerate(data_is_broken):
    Y.extend([i%2]*val)


n = tuple(map(float, range(10)))
i = 0
y = []
a = np.empty((0,8), dtype=np.float64)
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
                                        X.append([i0, i1, i2, i3, i4, i5, i6, i7])
                                i += 1
    a = np.append(a, np.array(X), axis=0)
    X = []
    print(i0)
    break

print(a.shape)
df = pd.DataFrame(a)



#g = sns.pairplot(df)
g = sns.PairGrid(df)
g = g.map_diag(sns.kdeplot)#, color="r")
g = g.map_offdiag(sns.kdeplot)#, color="r")
g.savefig('pairplot_all.png')
#export QT_QPA_PLATFORM='offscreen'
