import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import cv2  
import math

filename = "EQ7/displacementGPS.txt"

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

x = np.linspace(1, len(X0), len(X0))

rft = np.fft.rfft(X0)
rft[100:] = 0
X0 = np.fft.irfft(rft)
X0 = np.append(X0, X0[len(X0) - 1:])

rft = np.fft.rfft(X1)
rft[100:] = 0
X1 = np.fft.irfft(rft)
X1 = np.append(X1, X1[len(X1) - 1:])

rft = np.fft.rfft(X2)
rft[100:] = 0
X2 = np.fft.irfft(rft)
X2 = np.append(X2, X2[len(X2) - 1:])

rft = np.fft.rfft(Y0)
rft[100:] = 0
Y0 = np.fft.irfft(rft)
Y0 = np.append(Y0, Y0[len(Y0) - 1:])

rft = np.fft.rfft(Y1)
rft[100:] = 0
Y1 = np.fft.irfft(rft)
Y1 = np.append(Y1, Y1[len(Y1) - 1:])

rft = np.fft.rfft(Y2)
rft[100:] = 0
Y2 = np.fft.irfft(rft)
Y2 = np.append(Y2, Y2[len(Y2) - 1:])

filename = "EQ7/displacementGPS_Denoise.txt"

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


