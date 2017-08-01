import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import cv2  

filename = "replacementGPS_normalize.txt"

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
    frame_i, x0, y0, x1, y1, x2, y2= [float(i) for i in line.split('\t')]
    X0.append(x0)
    X1.append(x1)
    X2.append(x2)
    Y0.append(y0)
    Y1.append(y1)
    Y2.append(y2)
    pass

filename = "velocityGPS.txt"

fps = 30
dt = 1/fps

vx0 = []
vy0 = []
vx1 = []
vy1 = []
vx2 = []
vy2 = []

with open(filename, 'w') as f:
    # f.write("721\t0\t0\t0\t0\t0\t0\t\n")
    for i in range(len(X0) - 1):
        vx0.append ((X0[i + 1] - X0[i]) * fps)
        vy0.append ((Y0[i + 1] - Y0[i]) * fps)
        vx1.append ((X1[i + 1] - X1[i]) * fps)
        vy1.append ((Y1[i + 1] - Y1[i]) * fps)
        vx2.append ((X2[i + 1] - X2[i]) * fps)
        vy2.append ((Y2[i + 1] - Y2[i]) * fps)
        f.write(str(i+721)+"\t"+\
            str(vx0[i])+"\t"+str(vy0[i])+"\t"+\
            str(vx1[i])+"\t"+str(vy1[i])+"\t"+\
            str(vx2[i])+"\t"+str(vy2[i])+"\t"+\
            "\n")
        pass

filename = "accelerateGPS.txt"

ax0 = []
ay0 = []
ax1 = []
ay1 = []
ax2 = []
ay2 = []

with open(filename, 'w') as f:
    # f.write("721\t0\t0\t0\t0\t0\t0\t\n")
    for i in range(len(vx0) - 1):
        ax0.append ((vx0[i + 1] - vx0[i]) * fps)
        ay0.append ((vy0[i + 1] - vy0[i]) * fps)
        ax1.append ((vx1[i + 1] - vx1[i]) * fps)
        ay1.append ((vy1[i + 1] - vy1[i]) * fps)
        ax2.append ((vx2[i + 1] - vx2[i]) * fps)
        ay2.append ((vy2[i + 1] - vy2[i]) * fps)
        f.write(str(i+721)+"\t"+\
            str(ax0[i])+"\t"+str(ay0[i])+"\t"+\
            str(ax1[i])+"\t"+str(ay1[i])+"\t"+\
            str(ax2[i])+"\t"+str(ay2[i])+"\t"+\
            "\n")
        pass