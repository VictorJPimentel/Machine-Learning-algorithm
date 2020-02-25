import pandas as pd
import numpy as np
import sklearn
import sys
from sklearn import  linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep =";")


data = data[["G1", "G2", "G3", "Walc", "studytime", "failures", "absences", "age"]]
print(data.head())

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
"""
best = 0.96;
for _ in range(15030):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        print("We found it")
        with open("student.pickle", "wb") as f:
            pickle.dump(linear, f)
        sys.exit()
"""



pickle_in = open("student.pickle", "rb")
linear = pickle.load(pickle_in)

print('Co:  \n', linear.coef_)
print('Intercept:  \n', linear.intercept_)

prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(int(prediction[x]), x_test[x], y_test[x])

p = 'G2'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()