import numpy as np
import pandas as pd
from kapteyn import kmpfit

size = 50

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
print(input[input.columns[1:4]])

def model(CD, camber, thickness, cl):
   a,b,c,d,e,f,g,h,i,j,k = CD
   return  a*camber**2 + b*thickness**2 + c*cl**2 + d*camber*thickness + e*camber*cl + f*thickness*cl + g*camber + h*thickness + i*cl + k

def residuals(CD, data):
   camber, thickness, cl, z = data
   a,b,c,d,e,f,g,h,i,j,k = CD
   return (z-model(CD, camber, thickness, cl))

par0 = [1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
fitobj = kmpfit.Fitter(residuals=input[input.columns[1:4]].to_numpy(), data=input["CD"].to_numpy())
fitobj.fit(params0=par0)
print(input)

