import os
import subprocess
import numpy as np

Re = 200000  # Reynolds Number
cl = 0.5  # cl

# input and optimization domain
thickness_dom = [0.06, 0.12]
camber_dom = [0.01, 0.04]
#x_max_thick_dom = [0.2, 0.5]
#x_max_camber_dom = [0.2, 0.5]

#(-3*cl)/(cd*2)
# *********************************

# define functions of interest
def run_xfoil(X_xfoil):
    thickness = X_xfoil[0]
    max_camber = X_xfoil[1]
    Cl = X_xfoil[2]
    # print(X_xfoil)
    report_name = 'xfoil_out.txt'
    if os.path.exists(report_name):
        os.remove(report_name)
    xfoilscript = open('xfscript', 'w+')
    xfoilscript.write("""
plop
g

naca 2412

gdes
tset {} {}
eXec

oper
iter 500
v {}
pacc
{}

cl {}

quit
""".format(thickness, max_camber, Re, report_name, Cl))

    xfoilscript.close()
    subprocess.run("xfoil < xfscript 2>&1 xfoil.out", shell = True, check = True)
    return report_name

#DoE creating a set of xfoil data for analysis
size = 10
file_name = "xfoil_doe.txt"
if os.path.exists(file_name):
    os.remove(file_name)
fh = open(file_name, 'w+')
fh.write('thickness,camber,cl,-3CL/2CD' + "\n")
list = []
for i in np.linspace(thickness_dom[0], thickness_dom[1], size):
    for j in np.linspace(camber_dom[0], camber_dom[1], size):
        f_name = run_xfoil([i, j, cl])
        xfoilh = open(f_name, 'r')
        lines = xfoilh.readlines()
        output = np.nan
        has_found = False
        for line in lines:
            line = line.strip() #remove white spaces at the front and back of the line
            print(line)
            if has_found == True:
                counter = 0
                for item in line.split():
                    item = item.strip()
                    if counter == 1:
                        CL = item
                    elif counter == 2:
                        output = -3 * float(CL) / (2 * float(item))
                        break
                    counter = counter + 1
                break
            if '-' in line and has_found == False:
                has_found = True
                continue
        xfoilh.close()
        fh.write(str(i) + ',' + str(j) + ',' + str(cl) + ',' + str(output) + '\n')
fh.close()
print(list)





