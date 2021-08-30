import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


size = 500

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
c = input["Camber"].to_numpy()
t = input["thickness"].to_numpy()
l = input["CL"].to_numpy()
d = input["CD"].to_numpy()
print(input)
#curvefit
A = np.column_stack([np.ones(input["Camber"].size), c, t, l, c*t, c*l, t*l, c**2, t**2, l**2])
B = d
coefficient = np.linalg.lstsq(A, B, rcond = None)
curvefit = coefficient[0] #1, c, t, l, c*t, c*l, t*l, c**2, t**2, l**2
print("Curvefit is : ")
print(curvefit)
#R2
avg = np.ones(c.size) * np.mean(d)
sstot = ((d - avg)**2) @ np.ones(c.size)
ssres = ((d - A @ curvefit)**2) @ np.ones(c.size)
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
x = input["Camber"].to_numpy()
y = input["thickness"].to_numpy()
z = input["CL"].to_numpy()
yhat = curvefit[0] + curvefit[1] * x + curvefit[2] * y + curvefit[3] * z + curvefit[4] * x * y + \
             curvefit[5] * x * z + curvefit[6] * y * z + curvefit[7] * x**2 + curvefit[8] * y**2 + curvefit[9] * z**2
yprime = input["CD"].to_numpy()
rmse = np.sqrt(np.sum((np.absolute(yprime - yhat))**2)/x.size)
print("RMSE is : ")
print(rmse)

plt.show()