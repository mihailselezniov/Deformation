# -*- coding: utf-8 -*-
import numpy as np
from sklearn.externals import joblib

model = joblib.load('dump_models/KNN7.j')

x_test = np.array([[1.00000e+01, 5.00000e-01, 3.00000e+02, 1.77778e+03, 7.77800e+01, 5.60000e-01, 1.33330e+02, 4.56000e+00], [1.00000e+01, 5.00000e-01, 3.00000e+02, 2.00000e+03, 1.00000e+02, 5.00000e+00, 4.44400e+01, 6.73000e+00]])

y_pred = model.predict(x_test)
print(y_pred)



'''
[1 1]
'''


