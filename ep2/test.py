from numpy import meshgrid
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ep2OOP import ep2OOP
import numpy as np

test = ep2OOP("xfoil_doe.txt", ["thickness", "camber", "cl"], [0.06, 0.12, 0.01, 0.04], 3, 3)
#print('test')
print("R2 is ", test.R2())
print("RMSE is ", test.RMSE())
test.plot3D(25)
test.actualByPredicted()
test.optimizer()
"""

size = 10
a = np.linspace(1, 10, size)
b = a**2
x = np.concatenate((a.reshape(size, 1), b.reshape(size, 1)), axis=1)
#print(a.reshape(size, 1) + b.reshape(size, 1))

scalerX = StandardScaler()
scalerY = StandardScaler()
x = np.concatenate((a.reshape(size, 1), b.reshape(size, 1)), axis = 1)
x_norm = scalerX.fit_transform(x)
print(x)
print(scalerX.inverse_transform(x_norm))

y_norm = scalerY.fit_transform((a+b).reshape(size, 1))
mlp = MLPRegressor(hidden_layer_sizes=(1000, 1000, 1000))
mlp.fit(x_norm, y_norm)
#print((a.reshape(size, 1) + b.reshape(size, 1)).reshape(1, size))
print(y_norm.reshape(size))
print(mlp.predict(x_norm))
print("*********************************************")
print((a+b))
print(scalerY.inverse_transform(mlp.predict(x_norm)))

"""