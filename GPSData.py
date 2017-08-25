import numpy as np
import math
from scipy import interpolate

# def interpolate(a, b, alpha):
#     return a * (1 - alpha) + b * alpha

filename = "EQ7/GPSRecord_Raw.txt"

X0 = []
X1 = []
X2 = []
Y0 = []
Y1 = []
Y2 = []

with open(filename, "r") as f:
  while 1:
    line = f.readline() 
    if not line:
        break
        pass
    line = line[:line.find('\t\n')]
    frame_i, x0, y0, x1, y1, x2, y2 = [float(i) for i in line.split('\t')]
    if frame_i < 3267: #3285
        continue
    if frame_i > 4299:
        break
    X0.append(x0)
    X1.append(x1)
    X2.append(x2)
    Y0.append(y0)
    Y1.append(y1)
    Y2.append(y2)


para = 3286

x= np.arange(1, 3 * len(X0),3)

from scipy import interpolate

s = interpolate.InterpolatedUnivariateSpline(x, X0)
xnew = np.arange(1, 3 * len(X0))
X0 = s(xnew)

s = interpolate.InterpolatedUnivariateSpline(x, X1)
xnew = np.arange(1, 3 * len(X0))
X1 = s(xnew)

s = interpolate.InterpolatedUnivariateSpline(x, X2)
xnew = np.arange(1, 3 * len(X0))
X2 = s(xnew)

s = interpolate.InterpolatedUnivariateSpline(x, Y0)
xnew = np.arange(1, 3 * len(X0))
Y0 = s(xnew)

s = interpolate.InterpolatedUnivariateSpline(x, Y1)
xnew = np.arange(1, 3 * len(X0))
Y1 = s(xnew)

s = interpolate.InterpolatedUnivariateSpline(x, Y2)
xnew = np.arange(1, 3 * len(X0))
Y2 = s(xnew)

filename = "EQ7/GPSRecord_Prcss.txt"

with open(filename, "w") as f:
    for i in range(len(X0) - 1):

        if i + 3267 <= para :
            continue

        f.write(str(i + 3267)+"\t"+\
            str(X0[i])+"\t"+str(Y0[i])+"\t"+\
            str(X1[i])+"\t"+str(Y1[i])+"\t"+\
            str(X2[i])+"\t"+str(Y2[i])+"\t"+\
            "\n")

        if i + 3267 > 4250:
            break
