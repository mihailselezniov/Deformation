import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier

from data_analysis import get_data


data = get_data()
print(data[0])

X, y = [], []
x_params = ['density', 'diameter', 'length', 'pressure_amplitude', 'pressure_radius', 'pressure_time', 'strength', 'young']
for row in data:
    X.append([row[param] for param in x_params])
    y.append(row['is_broken'])

X, y = np.array(X), np.array(y)
print(X[0], y[0])
# split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

### Linear Regression 89.61% ###
clf = LinearRegression(normalize=True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(r2_score(y_test, y_pred)) #0.5099823026228494

predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


### XGBoost 97.30% ###
# fit model no training data
model = XGBClassifier()
model.fit(x_train, y_train)
# make predictions for test data
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
