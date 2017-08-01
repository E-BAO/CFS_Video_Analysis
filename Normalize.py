import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import cv2  
import math

filename = "replacementGPS.txt"

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
    X0.append(x0)
    X1.append(x1)
    X2.append(x2)
    # X3.append(x3)
    Y0.append(y0)
    Y1.append(y1)
    Y2.append(y2)
    # Y3.append(y3)
    pass


filename = "replacement.txt"

StdX0 = []
StdX1 = []
StdX2 = []
StdY0 = []
StdY1 = []
StdY2 = []
# Y3 = []

with open(filename, "r") as f:
  while 1:
    line = f.readline() 
    if not line:
        break
        pass
    line = line[:line.find('\t\n')]
    frame_i, x0, y0, x1, y1, x2, y2, x3, y3 = [float(i) for i in line.split('\t')]
    StdX0.append(x0)
    StdX1.append(x1)
    StdX2.append(x2)
    # X3.append(x3)
    StdY0.append(y0)
    StdY1.append(y1)
    StdY2.append(y2)
    # Y3.append(y3)
    pass


lengthX = math.sqrt(math.pow(StdX1[0] - StdX0[0],2) + math.pow(StdY1[0] - StdY0[0],2))

lengthY = math.sqrt(math.pow(StdX1[0] - StdX2[0],2) + math.pow(StdY1[0] - StdY2[0],2))

centerX, centerY = 800,460
filename = "replacementGPS_normalize.txt"

with open(filename, "w") as f:
    # f.write("720\t"+\
    #     str(X0[i])+"\t"+str(Y0[i])+"\t"+\
    #     str(X1[i])+"\t"+str(Y1[i])+"\t"+\
    #     str(X2[i])+"\t"+str(Y2[i])+"\t"+\
    #     "\n")
    for i in range(len(X0)):
        curLengthX = math.sqrt(math.pow(StdX1[i] - StdX0[i],2) + math.pow(StdY1[i] - StdY0[i],2))
        curLengthY = math.sqrt(math.pow(StdX1[i] - StdX2[i],2) + math.pow(StdY1[i] - StdY2[i],2))
        rateX = lengthX/curLengthX
        rateY = lengthY/curLengthY
        X0[i] = centerX - (centerX - X0[i]) * rateX
        X1[i] = centerX + (X1[i] - centerX) * rateX
        # X3[i] = centerX - (centerX - X3[i]) * rateX
        X2[i] = centerX + (X2[i] - centerX) * rateX

        Y0[i] = centerY + (Y0[i] - centerY) * rateY
        Y1[i] = centerY - (centerY - Y1[i]) * rateY
        Y2[i] = centerY + (Y2[i] - centerY) * rateY
        # Y3[i] = centerY + (Y3[i] - centerY) * rateY
        f.write(str(i+721)+"\t"+\
            str(X0[i])+"\t"+str(Y0[i])+"\t"+\
            str(X1[i])+"\t"+str(Y1[i])+"\t"+\
            str(X2[i])+"\t"+str(Y2[i])+"\t"+\
            # str(X3[i])+"\t"+str(Y3[i])+\
            "\n")