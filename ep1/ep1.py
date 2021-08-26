import numpy as np
import pandas as pd

#Generate Data
camber = np.random.rand(50, 1)
thickness = np.random.rand(50, 1)
cl = np.random.rand(50, 1)
epsilon = np.random.normal(0, 1, 50).reshape(50, 1)
cd =3 * np.ones((50, 1)) + .2*camber + 1.5*thickness + 0.5*thickness*camber +0.1*camber**3 + np.cos(thickness) + epsilon
#assign some random error in the table ~5% NaN
cd[2::16] = np.nan
table = np.hstack((camber, thickness, cl, cd))
np.savetxt("outputTable.csv", table, delimiter=",")
print(table)

#Create Surrogate Model
input = pd.read_csv("outputTable.csv", sep=",", names=["Camber", "thickness", "CL", "CD"])
assignment = np.random.uniform(0, 1, 50)
print(assignment)
print(input)

