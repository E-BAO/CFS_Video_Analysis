import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation
from matplotlib import pyplot as plt
from settings import *
import utilities as ut
from utilities import Arrow3D
import global_figure as gf

filename = "EQ7/CameraPoseDetect1.txt"

max_unit_length = 250

pattern_image = cv2.imread("EQ7/frame3300_origin.png")
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


def camPoseEstimate(image):
    '''
    Get estimated camera position and orientation in 3D world coordinates.

    Input:
        image_path: Input image path - string
    Output:
        Coordinates of camera in 3D world coordinates and its orientation matrix - numpy.array, numpy.array
    '''
    size = image.shape
    pattern_points = np.array(QRCode.detectQRCode(image), dtype='double')
    model_points = np.array([   (-QRCodeSide / 2, QRCodeSide / 2, 0.0), 
                                (QRCodeSide / 2, QRCodeSide / 2, 0.0), 
                                (QRCodeSide / 2, -QRCodeSide / 2, 0.0), 
                                (-QRCodeSide / 2, -QRCodeSide / 2, 0.0), 
                            ])
    focal_length = size[1]
    camera_center = (size[1] / 2, size[0] / 2)
    camera_intrinsic_matrix = np.array([[focal_length, 0, camera_center[0]],
                                        [0, focal_length, camera_center[1]],
                                        [0, 0, 1]
                                        ], dtype = "double")
    dist_coeffs = np.zeros((4, 1))
    flag, rotation_vector, translation_vector = cv2.solvePnP(   model_points, 
                                                                pattern_points, 
                                                                camera_intrinsic_matrix, 
                                                                dist_coeffs, 
                                                                flags=cv2.CV_ITERATIVE  )
    rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)
    srcC = np.matmul(-rotation_matrix.transpose(), translation_vector)
    srcC = srcC.squeeze()

    # Orientation vector
    
    Ox = np.matmul(rotation_matrix.T, np.array([1, 0, 0]).T)
    Oy = np.matmul(rotation_matrix.T, np.array([0, 1, 0]).T)
    Oz = np.matmul(rotation_matrix.T, np.array([0, 0, 1]).T)

    transM = np.array([  [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1] ])

    print rotation_matrix

    rotation_matrix = np.dot(transM, rotation_matrix)
    print rotation_matrix

    srcR = rotation_matrix
    srcT = - np.dot(srcR, srcC)
    srcExtrinsic_matrix = np.column_stack((srcR, srcT))

    return camera_intrinsic_matrix, srcExtrinsic_matrix

IM = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]

frame_i = 0
srcSize = 0

with open(filename, "r") as f:
    IM = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
    line = f.readline() 
    frame_i = int(line)
    srcImage = cv2.imread("EQ7/Frames/frame%d.png"%(frame_i))
    srcSize = srcImage.shape
    for r in range(3):
        line = f.readline() 
        line = line[:line.find('\t\n')]
        IM[r][0],IM[r][1],IM[r][2] = [float(i) for i in line.split('\t')]

camera_intrinsic_matrix = np.array(IM, dtype = "double")

print "camera_intrinsic_matrix=\n",camera_intrinsic_matrix

with open(filename, "r") as f:

    srcIntrinsic_matrix = camera_intrinsic_matrix
    dstIntrinsic_matrix = camera_intrinsic_matrix
    Kd = np.row_stack((dstIntrinsic_matrix, np.float32([0,0,0])))
    Kd = np.column_stack((Kd, np.float32([0,0,0,1])))

    Ks = np.row_stack((srcIntrinsic_matrix, np.float32([0,0,0])))
    Ks = np.column_stack((Ks, np.float32([0,0,0,1])))    

    while 1:
        line = f.readline() 
        if not line:
            break
        frame_i = int(line)
        print frame_i

        srcImage = cv2.imread("EQ7/Frames/frame%d.png"%(frame_i))
        
        for r in range(3):
            line = f.readline() 

        EM = [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
        for r in range(3):
            line = f.readline() 
            line = line[:line.find('\t\n')]
            EM[r][0],EM[r][1],EM[r][2],EM[r][3] = [float(i) for i in line.split('\t')]

        srcExtrinsic_matrix = np.array(EM, dtype = "double")

        print "srcExtrinsic_matrix = \n",srcExtrinsic_matrix

        rotation_matrix = srcExtrinsic_matrix[:,0:3]
        translation_vector = srcExtrinsic_matrix[:,3:]
        srcC = np.matmul(-rotation_matrix.transpose(), translation_vector)
        srcC = srcC.squeeze()
        print "cameraPose = ",srcC

        model_height = 0
        camera_height = 13062
        scale = camera_height / (camera_height - model_height) 

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

        print "PM = ", PM

        pixel_corners = np.float32([(-10,10),(10,10),(10,-10),(-10,-10)])

        scene_corners = []

        row3 = np.matmul(EsIdotKsI.T, np.array([0, 0, 1, 0]).T)
        print "__________",EsIdotKsI,row3

        scale = camera_height / (camera_height - model_height) 
        scale = scale * scale

        # print "dstC = ",dstC.squeeze(),camera_height, scale
        for c in pixel_corners:
            Cscreen = np.float32([c[0],c[1],1,0])
            Res = np.dot(row3, Cscreen)
            print row3,"dot",Cscreen,"=",Res
            Cz = model_height #############################################################################
            t3 = row3[3]
            s =  Res / (Cz - t3) 
            print Res," / ",(Cz - t3),"=",s

            c = np.float32([c[0],c[1],1,s])

            Cworld = np.dot(EsIdotKsI, c)
            Cworld = Cworld / Cworld[3]
            print "Cworld = ", Cworld
            d = np.dot(KddotEd, Cworld)
            # d = np.dot(PM, c)
            d = d/d[2]
            # d[0] = (d[0] - centerX) / scale + centerX
            # d[1] = (d[1] - centerX) / scale + centerY
            scene_corners.append( (d[0], d[1]) )

        print "points________\n",pixel_corners,"\n",np.float32(scene_corners)

        src_pts = np.float32([ k for k in pixel_corners ]).reshape(-1,1,2)
        dst_pts = np.float32([ k for k in scene_corners ]).reshape(-1,1,2)

        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)    
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        print " M=\n", M

        srcCorner = [(25,srcSize[0]),(srcSize[1],srcSize[0]),(srcSize[1],0),(0,0)]

        img2 = cv2.warpPerspective(srcImage, M, (srcSize[1], srcSize[0]))

        img2 = cv2.resize(img2, None, fx =0.2, fy=0.2)
        cv2.imshow("kkk",img2)
        img2 = cv2.resize(srcImage, None, fx =0.2, fy=0.2)
        cv2.imshow("hhh",img2)

        cv2.waitKey(30)


        