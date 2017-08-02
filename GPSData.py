import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import cv2  
import math

filename = "EQ7/GPSRecord_Raw.txt"

X0 = []
X1 = []
X2 = []
# X3 = []
Y0 = []
Y1 = []
Y2 = []
# Y3 = []

with open(filename, "r") as f:
  while 1:
    line = f.readline() 
    if not line:
        break
        pass
    line = line[:line.find('\t\n')]
    frame_i, x0, y0, x1, y1, x2, y2 = [float(i) for i in line.split('\t')]
    if frame_i < 3285:
        continue

    X0.append(x0)
    X0.append(x0)
    X0.append(x0)
    X1.append(x1)
    X1.append(x1)
    X1.append(x1)
    X2.append(x2)
    X2.append(x2)
    X2.append(x2)
    # X3.append(x3)
    Y0.append(y0)
    Y0.append(y0)
    Y0.append(y0)
    Y1.append(y1)
    Y1.append(y1)
    Y1.append(y1)
    Y2.append(y2)
    Y2.append(y2)
    Y2.append(y2)
    # Y3.append(y3)
    pass

filename = "EQ7/GPSRecord_Prcss.txt"

with open(filename, "w") as f:
    # f.write("720\t"+\
    #     str(X0[i])+"\t"+str(Y0[i])+"\t"+\
    #     str(X1[i])+"\t"+str(Y1[i])+"\t"+\
    #     str(X2[i])+"\t"+str(Y2[i])+"\t"+\
    #     "\n")
    for i in range(len(X0)):
        f.write(str(i+3267)+"\t"+\
            str(X0[i])+"\t"+str(Y0[i])+"\t"+\
            str(X1[i])+"\t"+str(Y1[i])+"\t"+\
            str(X2[i])+"\t"+str(Y2[i])+"\t"+\
            # str(X3[i])+"\t"+str(Y3[i])+\
            "\n")
        if i+3267 > 4225:
            break
