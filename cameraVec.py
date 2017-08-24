import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation
from matplotlib import pyplot as plt
from settings import *
import utilities as ut
from utilities import Arrow3D
import global_figure as gf

filename = "EQ5/CameraPoseDetectepnp.txt"

max_unit_length = 250

pattern_image = cv2.imread("EQ5/ReferenceGround.jpg")
size = pattern_image.shape
pattern_image = cv2.resize(pattern_image, None, fx = 0.1, fy=0.1)

# pattern_image = cv2.flip(cv2.transpose(pattern_image), 0)

# Plot pattern
height = pattern_image.shape[0]
width = pattern_image.shape[1]
xx, yy = np.meshgrid(np.linspace(0, width * 0.5, width), 
    np.linspace(height * 0.5,0, height))

X = xx
Y = yy
Z = 0

camera_pose = [0,0,0]
camera_orientation = [0,0,0]

R_vectorX = []
R_vectorY = []
R_vectorZ = []
T_vectorX = []
T_vectorY = []
T_vectorZ = []

frame_i = 0
srcSize = 0

with open("EQ7/CameraPoseDetect1.txt", "r") as f:
    IM = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
    line = f.readline() 
    frame_i = int(line)
    for r in range(3):
        line = f.readline() 
        line = line[:line.find('\t\n')]
        IM[r][0],IM[r][1],IM[r][2] = [float(i) for i in line.split('\t')]

camera_intrinsic_matrix = np.array(IM, dtype = "double")

print "camera_intrinsic_matrix=\n",camera_intrinsic_matrix

srcIntrinsic_matrix = camera_intrinsic_matrix
dstIntrinsic_matrix = camera_intrinsic_matrix
Kd = np.row_stack((dstIntrinsic_matrix, np.float32([0,0,0])))
Kd = np.column_stack((Kd, np.float32([0,0,0,1])))

Ks = np.row_stack((srcIntrinsic_matrix, np.float32([0,0,0])))
Ks = np.column_stack((Ks, np.float32([0,0,0,1])))    


with open(filename, "r") as f:
    Rvec = [0, 0, 0]
    Tvec = [0, 0, 0]
    while 1:
        line = f.readline() 
        if not line:
            break
        line = line[:line.find('\t\n')]
        frame_i, Rvec[0], Rvec[1], Rvec[2],\
        Tvec[0], Tvec[1], Tvec[2] = [float(i) for i in line.split('\t')]

        R_vectorX.append(Rvec[0])
        R_vectorY.append(Rvec[1])
        R_vectorZ.append(Rvec[2])
        T_vectorX.append(Tvec[0])
        T_vectorY.append(Tvec[1])
        T_vectorZ.append(Tvec[2])




def smoothFFT1(Y):
    x = np.linspace(1, len(Y), len(Y))
    y = [(Y[k]) for k in range(len(Y))]
    rft = np.fft.rfft(y)
    rft[10:] = 0
    y_smooth = np.fft.irfft(rft)
    a = y_smooth[len(y_smooth) - 1:]
    y_smooth = np.append(y_smooth,a)  
    return y_smooth

def smoothFFT2(Y):
    x = np.linspace(1, len(Y), len(Y))
    y = [(Y[k]) for k in range(len(Y))]
    rft = np.fft.rfft(y)
    rft[10:] = 0
    y_smooth = np.fft.irfft(rft)
    a = y_smooth[len(y_smooth) - 1:]
    y_smooth = np.append(y_smooth,a)  
    return y_smooth

R_vectorX = smoothFFT1(R_vectorX)
R_vectorY = smoothFFT1(R_vectorY)
R_vectorZ = smoothFFT1(R_vectorZ)

T_vectorX = smoothFFT2(T_vectorX)
T_vectorY = smoothFFT2(T_vectorY)
T_vectorZ = smoothFFT2(T_vectorZ)

frame_start = 720

srcImage = cv2.imread("EQ5/Frames/frame720.png")
srcSize = srcImage.shape
r,c,a = srcImage.shape
fps = 30
# video  = cv2.VideoWriter('EQ7/Calibration/video.avi', -1, fps, (c, r))

with open("EQ5/Calibration/CalibrationDenoise.txt", "w") as f:
    for i in range(len(R_vectorX)):

        frame_i = frame_start + i

        print frame_i

        Rvec = [R_vectorX[i],R_vectorY[i],R_vectorZ[i]]
        Tvec = [T_vectorX[i],T_vectorY[i],T_vectorZ[i]]

        rotation_vector = np.array([[Rvec[0]],[Rvec[1]],[Rvec[2]]])
        rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)

        translation_vector = np.array([[Tvec[0]],[Tvec[1]],[Tvec[2]]])

        Ox = np.matmul(rotation_matrix.T, np.array([1, 0, 0]).T)
        Oy = np.matmul(rotation_matrix.T, np.array([0, 1, 0]).T)
        Oz = np.matmul(rotation_matrix.T, np.array([0, 0, 1]).T)


        C = np.matmul(-rotation_matrix.transpose(), translation_vector)

        camera_pose = C.squeeze()
        R = rotation_matrix
        T = translation_vector.squeeze() # - np.dot(srcR, srcC)
        srcExtrinsic_matrix = np.column_stack((R, T))

        srcImage = cv2.imread("EQ5/Frames/frame%d.png"%(frame_i))

        srcC = C.squeeze()
        print "cameraPose = ",srcC

        model_height = 0
        camera_height = 13062

        centerX, centerY = size[1]/2., size[0]/2.
        dstC = np.float32([centerX,centerY,camera_height])
        dstR = np.float32([[1,0,0],[0,1,0],[0,0,-1]])
        dstT = - np.dot(dstR, dstC)
        dstExtrinsic_matrix = np.column_stack((dstR, dstT))

        Es = np.row_stack((srcExtrinsic_matrix, np.float32([0,0,0,1])))
        Ed = np.row_stack((dstExtrinsic_matrix, np.float32([0,0,0,1])))

        Ksinv = np.linalg.inv(Ks)
        KddotEd = np.dot(Kd, Ed)
        KsdotEs = np.dot(Ks, Es)
        
        Esinv = np.linalg.inv(Es)
        EsIdotKsI = np.linalg.inv(KsdotEs)
        PM = np.dot(KddotEd, EsIdotKsI)

        pixel_corners = np.float32([(-10,10),(10,10),(10,-10),(-10,-10)])

        scene_corners = []

        row3 = np.matmul(EsIdotKsI.T, np.array([0, 0, 1, 0]).T)

        scale = camera_height / (camera_height - model_height) 
        scale = scale * scale

        # print "dstC = ",dstC.squeeze(),camera_height, scale
        for c in pixel_corners:
            Cscreen = np.float32([c[0],c[1],1,0])
            Res = np.dot(row3, Cscreen)
            Cz = model_height #############################################################################
            t3 = row3[3]
            s =  Res / (Cz - t3) 

            c = np.float32([c[0],c[1],1,s])

            Cworld = np.dot(EsIdotKsI, c)
            Cworld = Cworld / Cworld[3]
            d = np.dot(KddotEd, Cworld)
            # d = np.dot(PM, c)
            d = d/d[2]
            # d[0] = (d[0] - centerX) / scale + centerX
            # d[1] = (d[1] - centerX) / scale + centerY
            scene_corners.append( (d[0] , d[1] ) )

        print "points________\n",pixel_corners,"\n",np.float32(scene_corners)

        src_pts = np.float32([ k for k in pixel_corners ]).reshape(-1,1,2)
        dst_pts = np.float32([ k for k in scene_corners ]).reshape(-1,1,2)

        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)    
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        srcCorner = [(25,srcSize[0]),(srcSize[1],srcSize[0]),(srcSize[1],0),(0,0)]

        img2 = cv2.warpPerspective(srcImage, M, (srcSize[1], srcSize[0]))

        font = cv2.FONT_HERSHEY_SIMPLEX
        string = "("+str(int(srcC[0])) + "," + str(int(srcC[1])) + "," + str(int(-srcC[2])) + ")"
        cv2.putText(img2,string,(50,200), font, 2,(255,255,255),8)
        # string = "("+str(float(Ox[0])) + "," + str(float(Ox[1])) + "," + str(float(Ox[2])) + ")"
        # cv2.putText(img2,string,(50,400), font, 2,(255,255,255),8)
        # string = "("+str(float(Oy[0])) + "," + str(float(Oy[1])) + "," + str(float(Oy[2])) + ")"
        # cv2.putText(img2,string,(50,600), font, 2,(255,255,255),8)
        # string = "("+str(float(Oz[0])) + "," + str(float(Oz[1])) + "," + str(float(Oz[2])) + ")"
        # cv2.putText(img2,string,(50,800), font, 2,(255,255,255),8)

        f.write(str(frame_i) + "\t")
        f.write(str(srcC[0]) + "\t"  + str(srcC[1]) + "\t"  + str(srcC[2]) + "\t")
        f.write(str(Ox[0]) + "\t"  + str(Ox[1]) + "\t"  + str(Ox[2]) + "\t")
        f.write(str(Oy[0]) + "\t"  + str(Oy[1]) + "\t"  + str(Oy[2]) + "\t")
        f.write(str(Oz[0]) + "\t"  + str(Oz[1]) + "\t"  + str(Oz[2]) + "\t\n")


        cv2.imwrite("EQ5/Calibration/frame%d.png"%(frame_i),img2)
        # video.write(img2)

        img2 = cv2.resize(img2, None, fx =0.2, fy=0.2)
        cv2.imshow("kkk",img2)
        img2 = cv2.resize(srcImage, None, fx =0.2, fy=0.2)
        cv2.imshow("hhh",img2)

        cv2.waitKey(1)