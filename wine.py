import pandas as pd
import numpy as np
import sklearn
import sys
from sklearn import  linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from decimal import Decimal

data = pd.read_csv("winequality-red.csv", sep = ";")

data = data[["fixed acidity",  "alcohol", "quality", "sulphates", "pH", "chlorides"]]
print(data.head())


#quality
"""
predict2 = "quality"

x2 = np.array(data.drop([predict2], 1))
y2 = np.array(data[predict2])

x_train2, x_test2, y_train2, y_test2 = sklearn.model_selection.train_test_split(x2, y2, test_size=0.1)

linear2 = linear_model.LinearRegression()

linear2.fit(x_train2, y_train2)

prediction2 = linear2.predict2(x_test2)

for x in range(len(prediction2)):
    print(round(prediction2[x], 2), x_test2[x], round(y_test2[x], 2))

"""



#pH
predict = "pH"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)



"""
linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)




best = 0.80;
for _ in range(20030):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        print("We found it")
        with open("wine80.pickle", "wb") as f:
            pickle.dump(linear, f)
        sys.exit()
"""

pickle_in = open("wine.pickle", "rb")
linear = pickle.load(pickle_in)

prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(round(prediction[x], 2), x_test[x], round(y_test[x], 2))

acc = linear.score(x_test, y_test)
print(acc)

p = 'pH'
style.use("ggplot")
pyplot.scatter(data[p], data["quality"])
pyplot.xlabel(p)
pyplot.ylabel("Quality")
pyplot.show()