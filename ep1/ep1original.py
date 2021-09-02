import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen_der, rosen_hess

size = 20

#Generate Data
initial = np.random.rand(size, 3)
epsilon = np.random.normal(0, 1, size).reshape(size, 1)
cd =3 * np.ones((size, 1)) + .2*initial[:, 0].reshape(size, 1) + 1.5*initial[:, 1].reshape(size, 1) + 0.5*initial[:, 1].reshape(size, 1)*initial[:, 0].reshape(size, 1) +0.1*initial[:, 0].reshape(size, 1)**3 + np.cos(initial[:, 1].reshape(size, 1)) + epsilon + initial[:, 2].reshape(size, 1)
#assign some random error in the table ~5% NaN
cd[2::16] = np.nan
table = np.hstack((initial, cd))
np.savetxt("outputTable.csv", table, delimiter=",")

#Create Surrogate Model
input = pd.read_csv("outputTable.csv", sep=",", names=["Camber", "thickness", "CL", "CD"])
input["Assignment"] = np.random.uniform(0, 1, size)
#clean the data, strategy is to eliminate the row with NaN
input.dropna(inplace = True)
trainingSet = input[input["Assignment"] <= 0.8]
validationSet = input[input["Assignment"] > 0.8]
c = trainingSet["Camber"].to_numpy()
t = trainingSet["thickness"].to_numpy()
l = trainingSet["CL"].to_numpy()
d = trainingSet["CD"].to_numpy()
#curvefit
A = np.column_stack([np.ones(trainingSet["Camber"].size), c, t, l, c*t, c*l, t*l, c**2, t**2, l**2])
B = d
coefficient = np.linalg.lstsq(A, B, rcond = None)
curvefit = coefficient[0] #1, c, t, l, c*t, c*l, t*l, c**2, t**2, l**2
print("Curvefit is : ")
print(curvefit)
#R2
cv = validationSet["Camber"].to_numpy()
tv = validationSet["thickness"].to_numpy()
lv = validationSet["CL"].to_numpy()
dv = validationSet["CD"].to_numpy()
A = np.column_stack([np.ones(validationSet["Camber"].size), cv, tv, lv, cv*tv, cv*lv, tv*lv, cv**2, tv**2, lv**2])
B = dv
avg = np.ones(cv.size) * np.mean(dv)
sstot = ((B - avg)**2) @ np.ones(cv.size)
ssres = ((B - A @ curvefit)**2) @ np.ones(cv.size)
R2 = 1 - ssres / sstot
print("R2 is : ")
print(R2)

#3D plot of the surrogate
#restrict CL, three plots with CL = 0.2, 0.3, 0.5
num = size
CL = np.array([0.2, 0.3, 0.5])
pltNum = 1
fig = plt.figure(figsize=plt.figaspect(0.5))
for k in CL:
   z = np.ones(num) * k
   def f(x, y):
      return curvefit[0] + curvefit[1] * x + curvefit[2] * y + curvefit[3] * z + curvefit[4] * x * y + \
             curvefit[5] * x * z + curvefit[6] * y * z + curvefit[7] * x**2 + curvefit[8] * y**2 + curvefit[9] * z**2
   x = np.linspace(0, 1, num)
   y = x
   X, Y = np.meshgrid(x, y)
   Z = f(X, Y)
   ax = fig.add_subplot(1, CL.size, pltNum, projection='3d')
   ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
   ax.set_title('CL = ' + str(k))
   ax.set_xlabel('Camber')
   ax.set_ylabel('thickness')
   ax.set_zlabel('CD')
   pltNum = pltNum + 1
#calculating RMSE
x = validationSet["Camber"].to_numpy()
y = validationSet["thickness"].to_numpy()
z = validationSet["CL"].to_numpy()
yhat = curvefit[0] + curvefit[1] * x + curvefit[2] * y + curvefit[3] * z + curvefit[4] * x * y + curvefit[5] * x * z +\
       curvefit[6] * y * z + curvefit[7] * x**2 + curvefit[8] * y**2 + curvefit[9] * z**2
yprime = validationSet["CD"].to_numpy()
rmse = np.sqrt(np.sum((np.absolute(yprime - yhat))**2)/x.size)
print("RMSE is : ")
print(rmse)

#finding optimum of the surrogate model
def cfFunc(x):
   return curvefit[0] + curvefit[1] * x[0] + curvefit[2] * x[1] + curvefit[3] * x[2] + curvefit[4] * x[0] *\
          x[1] + curvefit[5] * x[0] * x[2] + curvefit[6] * x[1] * x[2] + curvefit[7] * x[0] ** 2 +\
          curvefit[8] * x[1] ** 2 + curvefit[9] * x[2] ** 2
def rosen(x):
   """The Rosenbrock function"""
   return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
x0 = np.array([0.5, 0.5, 0.5])
res = minimize(cfFunc, x0, method='Newton-CG', jac=rosen_der, hess=rosen_hess, options={'xtol': 1e-8, 'disp': True})
print(res.x)
print(rosen(res.x))


plt.show()