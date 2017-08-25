import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
import cv2

def drawlines(img1,pts1):
    r = img1.shape[0]
    c = img1.shape[1]
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    for pt1 in pts1:
        color = tuple(np.random.randint(0,255,3).tolist())
        cv2.circle(img1,tuple((int(pt1[0]),int(pt1[1]))),5,(1.0,1.0,0.0),-1)
    return img1

def drawlines2(img1,pts1):
    r = img1.shape[0]
    c = img1.shape[1]
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    for pt1 in pts1:
        color = tuple(np.random.randint(0,255,3).tolist())
        cv2.circle(img1,tuple((int(pt1[0]),int(pt1[1]))),5,color,-1)
    return img1

filename = "EQ7/displacementGPS_Denoise.txt"

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

filename = "EQ7/displacementGPS_Pixel.txt"

RX0 = []
RX1 = []
RX2 = []
RY0 = []
RY1 = []
RY2 = []

with open(filename, "r") as f:
  while 1:
    line = f.readline() 
    if not line:
        break
        pass
    line = line[:line.find('\t\n')]
    frame_i, x0, y0, x1, y1, x2, y2= [float(i) for i in line.split('\t')]
    RX0.append(x0)
    RX1.append(x1)
    RX2.append(x2)
    RY0.append(y0)
    RY1.append(y1)
    RY2.append(y2)
    pass

baseX = 0#X0[0] 
baseY = 0#Y0[0] 

# to image pixel    
for i in range(len(RX0)):
    RX0[i] = baseX - RX0[i] + X0[0]
    RX1[i] = baseX - RX1[i] + X1[0]
    RX2[i] = baseX - RX2[i] + X2[0]
    RY0[i] = baseY - RY0[i] + Y0[0]
    RY1[i] = baseY - RY1[i] + Y1[0]
    RY2[i] = baseY - RY2[i] + Y2[0]

frame_i = 3266

# color = [(255, 0, 0),(0, 0, 255),(0, 255, 0),(0, 0, 255),(255, 0, 0),(0, 255, 255),(0, 255, 0),(0, 0, 255)]
for i in range(len(X0)):
    frame_i = frame_i + 1
    print frame_i
    frame = cv2.imread("EQ7/Preprocess3/frame%d.png"%(frame_i))

    if frame is None:
        break

    r,c,a = frame.shape
    frame = frame[200:r - 100,0:c - 600]

    img = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

    x3 = X1[i] + X2[i] - X0[i]
    y3 = Y1[i] + Y2[i] - Y0[i]
    rx3 = RX1[i] + RX2[i] - RX0[i]
    ry3 = RY1[i] + RY2[i] - RY0[i]

    src_pts = np.float32([[X0[i], Y0[i]],[X1[i], Y1[i]],[x3, y3], [X2[i], Y2[i]]]).reshape(-1,1,2)
    dst_pts = np.float32([[RX0[i], RY0[i]],[RX1[i], RY1[i]],[rx3, ry3], [RX2[i], RY2[i]]]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    # img = drawlines2(img,[[X0[i], Y0[i]],[X1[i], Y1[i]],[x3, y3],[X2[i], Y2[i]]])
    # img = drawlines(img,[[RX0[i], RY0[i]],[RX1[i], RY1[i]],[rx3, ry3],[RX2[i], RY2[i]]])
    

    cv2.imwrite("EQ7/GPSPreprocess/frame%d.png"%(frame_i),img)

    img = cv2.resize(img, None, fx = 0.5, fy= 0.5)
    
    cv2.imshow('img',img)
    
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

    
