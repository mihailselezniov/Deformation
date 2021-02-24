#import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
#sns.set(style="ticks", color_codes=True)

with open('../Y/Y.txt', 'r') as f:
    y_str = f.readlines()
y = [int(i) for i in y_str[0]]
#print(len(Y))
Y = np.array(y)

def cartesian_coord(*arrays):
    grid = np.meshgrid(*arrays)        
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points

a = np.arange(100)
X = cartesian_coord(*4*[a])
X = X[::10]
X = X[Y[::10] == 0]

print(X.shape)
df = pd.DataFrame(X)



#g = sns.pairplot(df)
g = sns.PairGrid(df)
g = g.map_diag(sns.kdeplot)#, color="r")
g = g.map_offdiag(sns.kdeplot)#, color="r")
g.savefig('pairplot_all2.png')
#export QT_QPA_PLATFORM='offscreen'
