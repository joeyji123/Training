import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen_der, rosen_hess

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

class ep1OOP:
    pva1 = "potato"

    def __init__(self, fileName, headersCleaning, rangeCleaning, inputCol):
        self.fileName = fileName #store the file name we wish to read in
        self.input = pd.read_csv(fileName, sep=",")
        #data cleaning portion, delete rows with in valid data, out of range data
        self.input.dropna(inplace=True)
        self.input = self.input.loc[(self.input[headersCleaning[0]] >= rangeCleaning[0]) & (self.input[headersCleaning[0]] <= rangeCleaning[1]) &\
                                    (self.input[headersCleaning[1]] >= rangeCleaning[2]) & (self.input[headersCleaning[1]] <= rangeCleaning[3])]

        #assign
        self.size = len(self.input) #number of rows in the data
        self.input["Assignment"] = np.random.uniform(0, 1, self.size)
        self.trainingSet = self.input[self.input["Assignment"] <= 0.8]
        self.validationSet = self.input[self.input["Assignment"] > 0.8]
        self.inputCol = inputCol

    def show(self):
        #print(pd.read_csv(self.fileName, sep=","))
        #print(self.input)

        print(self.trainingSet.iloc[:, :2])

    def polyTransform(self, deg, x):
        poly = PolynomialFeatures(degree=deg)
        return poly.fit_transform(x)

    def curveFit(self, deg):
        x = self.polyTransform(deg, self.trainingSet.iloc[:, :self.inputCol].to_numpy())
        y = self.trainingSet.iloc[:, self.inputCol:len(self.input.columns)-1].to_numpy()
        clf = linear_model.LinearRegression()
        clf.fit(x, y)
        return clf

    def R2(self, deg):
        x = self.polyTransform(deg, self.validationSet.iloc[:, :self.inputCol].to_numpy())
        y = self.validationSet.iloc[:, self.inputCol:len(self.input.columns) - 1].to_numpy()
        clf = self.curveFit(2)
        m, n = y.shape
        avg = np.mean(y) * np.ones(m)
        sstot = ((y.reshape(1, m) - avg) ** 2) @ np.ones(m)
        ssres = ((y - clf.predict(x)) ** 2).reshape(1, m) @ np.ones(m)
        return 1 - ssres / sstot



#changed one thing

#Create Surrogate Model
"""

input = pd.read_csv("branin_data.csv", sep=",")
size = len(input)
print(input)
input["Assignment"] = np.random.uniform(0, 1, size)
#clean the data, strategy is to eliminate the row with NaN
input.dropna(inplace = True)
trainingSet = input[input["Assignment"] <= 0.8]
validationSet = input[input["Assignment"] > 0.8]
print(input)
print(input.to_numpy())

x = np.linspace(0, 1, 12).reshape(4, 3)
y = np.linspace(0, 1, 4).reshape(4, 1)
predict= [[0.49, 0.18, 0.19]]

poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(x)
predict_ = poly.fit_transform(predict)

clf = linear_model.LinearRegression()
clf.fit(X_, y)
print("X_ = ",X_)
print("predict_ = ",predict_)
print("Prediction = ",clf.predict(predict_))

"""