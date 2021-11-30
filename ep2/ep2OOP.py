import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn import linear_model

class ep2OOP:
    pva1 = "potato"

    def __init__(self, fileName, headersCleaning, rangeCleaning, inputCol, deg):
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
        self.rangeCleaning = rangeCleaning
        self.deg = deg

        x = self.polyTransform(self.trainingSet.iloc[:, :self.inputCol].to_numpy())
        y = self.trainingSet.iloc[:, self.inputCol:len(self.input.columns) - 1].to_numpy()
        self.clf = linear_model.LinearRegression()
        self.clf.fit(x, y)

        x = self.trainingSet.iloc[:, :self.inputCol].to_numpy()
        y = self.trainingSet.iloc[:, self.inputCol:len(self.input.columns) - 1].to_numpy()
        self.scalerX = StandardScaler()
        self.scalerY = StandardScaler()
        self.mlp = MLPRegressor(hidden_layer_sizes=(500, 500, 500, 500))
        self.mlp.fit(self.scalerX.fit_transform(x), self.scalerY.fit_transform(y))

    def polyTransform(self, x):
        poly = PolynomialFeatures(degree = self.deg)
        return poly.fit_transform(x)

    def R2(self):
        x = self.polyTransform(self.validationSet.iloc[:, :self.inputCol].to_numpy())
        y = self.validationSet.iloc[:, self.inputCol:len(self.input.columns) - 1].to_numpy()
        m, n = y.shape
        avg = np.mean(y) * np.ones(m)
        sstot = ((y.reshape(1, m) - avg) ** 2) @ np.ones(m)
        ssres = ((y - self.clf.predict(x)) ** 2).reshape(1, m) @ np.ones(m)
        return 1 - ssres / sstot

    def plot3D(self, size):
        fig = plt.figure(figsize=plt.figaspect(0.5))
        x1 = np.linspace(self.rangeCleaning[0], self.rangeCleaning[1], size)
        x2 = np.linspace(self.rangeCleaning[2], self.rangeCleaning[3], size)
        #changed this porton to accomadate for 3rd variable
        x3 = np.ones(size) * self.input["cl"][0]
        X1, X2, X3 = np.meshgrid(x1, x2, x3)
        x = self.polyTransform(np.concatenate((X1.reshape(size **3, 1), X2.reshape(size **3, 1), X3.reshape(size **3, 1)), axis = -1))
        y = self.clf.predict(x)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(X1.reshape(size, size**2), X2.reshape(size, size**2), y.reshape(size, size**2), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set(xlabel='thickness', ylabel='camber', zlabel='Cd' , title='xfoil_DoE, Cl = 0.5')
        plt.show()

    def RMSE(self):
        x = self.polyTransform(self.validationSet.iloc[:, :self.inputCol].to_numpy())
        yprime = self.validationSet.iloc[:, self.inputCol:len(self.input.columns) - 1].to_numpy()
        yhat = self.clf.predict(x)
        return np.sqrt(np.sum((np.absolute(yprime - yhat)) ** 2) / x.size)

    def actualByPredicted(self):
        fig = plt.figure()
        x = self.polyTransform(self.validationSet.iloc[:, :self.inputCol].to_numpy())
        yprime = self.validationSet.iloc[:, self.inputCol:len(self.input.columns) - 1].to_numpy()
        yhat = self.clf.predict(x)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(yprime, yhat, s=5, c=np.random.rand(yhat.size))
        ax1.set(xlabel='Validation Data', ylabel='Predicted Data', title='Third Order Polynomial Curvefit')
        ax1.grid()
        x = np.linspace(*ax1.get_xlim())
        ax1.plot(x, x)

        x = self.validationSet.iloc[:, :self.inputCol].to_numpy()
        x_norm = self.scalerX.transform(x)
        print("*****************")
        yhat = self.scalerY.inverse_transform(self.mlp.predict(x_norm))
        print(yhat)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(yprime, yhat.reshape(yhat.size, 1), s=5, c=np.random.rand(yhat.size))
        ax2.set(xlabel='Validation Data', ylabel='Predicted Data', title='Four Layers Neural Network with 1000 nodes each layer')
        ax2.grid()
        x = np.linspace(*ax2.get_xlim())
        ax2.plot(x, x)
        plt.show()

    def cfFunc(self, x):
        return self.clf.predict(self.polyTransform(x.reshape(-1, 3)))
    def optimizer(self):
        # x0 = 0.5* np.ones((2, 1))
        x0 = np.array([0.07, 0.02, 0.5])
        bounds = Bounds((0.06, 0.01, 0.5), (0.12, 0.04, 0.5))
        res = minimize(self.cfFunc, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True}, bounds=bounds)
        #res = minimize(self.cfFunc, x0, method='SLSQP', bounds=bounds)
        print('optimized function value ', self.cfFunc(res.x) * (-2/3))
        print('optimized input settings ', res.x)
        return res.x