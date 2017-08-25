import numpy as np
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import drawMatches as dm
import PerspecCut as ps
from settings import *
import utilities as ut
from utilities import Arrow3D
import global_figure as gf

def is_point_in(x, y, points,offset = 3):
    count = 0
    x1, y1 = points[0]
    x1_part = (y1 > y) or ((x1 - x > 0) and (y1 == y)) 
    x2, y2 = '', ''  
    points.append((x1, y1))
    for point in points[1:]:
        x2, y2 = point
        x2_part = (y2 > y) or ((x2 > x) and (y2 == y)) 
        if x2_part == x1_part:
            x1, y1 = x2, y2
            continue
        mul = (x1 - x)*(y2 - y) - (x2 - x)*(y1 - y)
        if mul > 0:  
            count += 1
        elif mul < 0:
            count -= 1
        x1, y1 = x2, y2
        x1_part = x2_part
    if count == 2 or count == -2:
        return True
    else:
        return False

def expendPts(pts,offset = 3):
    ctX = np.sum(pts[:,0])/len(pts)
    ctY = np.sum(pts[:,1])/len(pts)
    points = []
    for point in pts:
        x, y = point
        dx,dy = x - ctX, y - ctY
        t = offset / np.sqrt(dx * dx + dy * dy)
        pt = (x + t * dx * 2.0, y + t * dy * 1.2)
        points.append(pt)
    return points

def findCenter(img, pts, color):
    if len(pts) > 0:
        ctX = np.sum(pts[:,0])/len(pts)
        ctY = np.sum(pts[:,1])/len(pts)
        cv2.circle(img,(int(ctX), int(ctY)),2,color,-1)
    else:
        print("Points less than 2")
    return ctX, ctY

def drawlines(img1,pts1):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    for pt1 in pts1:
        color = tuple(np.random.randint(0,255,3).tolist())
        cv2.circle(img1,tuple(pt1),4,color,-1)

    return img1

# cap = cv2.VideoCapture('EQ7/EQ7_Top.avi')
# fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
# start_time = 0
# end_time = 32
# start_frame =  3266
frame_i = 720

MIN_MATCH_COUNT = 4
file_pattern = 'EQ5/ReferenceGround.jpg'
img1 = cv2.imread(file_pattern,0)          # queryImage

# pattern_image = cv2.imread("EQ5/ReferenceGround.png")
# pattern_image = cv2.resize(pattern_image, None, fx = 0.1, fy=0.1)
# pattern_image = cv2.flip(cv2.transpose(pattern_image), 0)

# Plot pattern
# height = pattern_image.shape[0]
# width = pattern_image.shape[1]
# xx, yy = np.meshgrid(np.linspace(0, width * 0.5, width), 
#     np.linspace(height * 0.5,0, height))

# Establish global figure
# gf.fig = plt.figure()
# gf.ax = gf.fig.add_subplot(111, projection='3d')

# Set axes labels
# gf.ax.set_xlabel('X')
# gf.ax.set_ylabel('Y')
# gf.ax.set_zlabel('Z')

# max_unit_length = 250

# gf.ax.set_xlim3d(0, 150)
# gf.ax.set_ylim3d(0, 150)
# gf.ax.set_zlim3d(0, 250)

# plt.gca().set_aspect('equal', adjustable='box')

sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)

with open("EQ5/CameraPoseDetectepnpIt.txt", "w") as f:
    while(frame_i <= 1678):
        print frame_i

        frame_p = cv2.imread("EQ5/Frames/frame%d.png"%(frame_i))

        # gf.ax.cla()

        # gf.ax.set_xlabel('X')
        # gf.ax.set_ylabel('Y')
        # gf.ax.set_zlabel('Z')
        # gf.ax.set_xlim3d(0, 150)
        # gf.ax.set_ylim3d(0, 150)
        # gf.ax.set_zlim3d(0, 250)
        # plt.gca().set_aspect('equal', adjustable='box')

        img2 = cv2.cvtColor(frame_p, cv2.COLOR_RGB2GRAY)

        print "pre......."
        img2_Cut,pts2,corp = ps.preProcess_origin(img2)
        # array([[ 184,  409],
        #    [ 170, 1347],
        #    [1508, 1371],
        #    [1527,  431]])
        pts2 = expendPts(pts2,10)

        print "matching..."
        #Initiate SIFT detector
        kp2, des2 = sift.detectAndCompute(img2_Cut,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                pt = kp2[m.trainIdx].pt
                if not is_point_in(pt[0],pt[1],pts2):
                    good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
            depth = np.zeros(len(dst_pts))
            dst_pts = np.c_[dst_pts, depth]

            src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])
            src_pts[:,0] = [x + corp[2] for x in src_pts[:,0]]
            src_pts[:,1] = [y + corp[0] for y in src_pts[:,1]]

            print "actually_______________",len(src_pts),len(dst_pts)
            
            img2 = drawlines(img2, src_pts)
            imgshow = cv2.resize(img2, None, fx = 0.3, fy= 0.3)

            cv2.imshow("frame_preprocess",imgshow)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

            print "R T ....."
            # depth = np.zeros(len(src_pts))
            # src_pts = np.c_[src_pts, depth]

            size = img2.shape
            focal_length = size[1]
            camera_center = (size[1] / 2, size[0] / 2)

            sw = 6.16
            sh = 4.62
            cw = 4000
            ch = 3000
            dx = sw/cw
            dy = sh/ch
            ff = 20
            fx = ff/dx
            fy = ff/dy
            u0 = size[1] / 2 
            v0 = size[0] / 2

            # Initialize approximate camera intrinsic matrix
            camera_intrinsic_matrix = np.array([[fx, 0, u0],
                                                [0, fy, v0],
                                                [0, 0, 1]
                                                ], dtype = "double")

            # Assume there is no lens distortion
            dist_coeffs = np.zeros((4, 1))


            flag, rotation_vector, translation_vector = cv2.solvePnP(   dst_pts, 
                                                                        src_pts, 
                                                                        camera_intrinsic_matrix, 
                                                                        dist_coeffs, 
                                                                        flags=cv2.CV_ITERATIVE   )

            # Convert 3x1 rotation vector to rotation matrix for further computation

            print rotation_vector

            rotation_vector = rotation_vector.squeeze()
            translation_vector = translation_vector.squeeze()
            f.write(str(frame_i) + "\t")

            for i in range(3):
                f.write(str(rotation_vector[i]) + "\t")
            for i in range(3):
                f.write(str(translation_vector[i]) + "\t")

            f.write("\n")


            # rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)

            # # C = -R.transpose() * T
            # Ox = np.matmul(rotation_matrix.T, np.array([1, 0, 0]).T)
            # Oy = np.matmul(rotation_matrix.T, np.array([0, 1, 0]).T)
            # Oz = np.matmul(rotation_matrix.T, np.array([0, 0, 1]).T)

            # C = np.matmul(-rotation_matrix.transpose(), translation_vector)

            # camera_pose = C.squeeze()
            # R = rotation_matrix
            # T = translation_vector.squeeze() # - np.dot(srcR, srcC)
            # Extrinsic_matrix = np.column_stack((R, T))
            # print "Extrinsic_matrix_______________\n",Extrinsic_matrix

            # # K = np.row_stack((Intrinsic_matrix, np.float32([0,0,0])))
            # # K = np.column_stack((K, np.float32([0,0,0,1])))

            # # E = np.row_stack((Extrinsic_matrix, np.float32([0,0,0,1])))

            # # KdotE = np.dot(K, E)

            # # print "KdotE_______________\n",KdotE

            # # Ox = np.matmul(rotation_matrix.T, np.array([1, 0, 0]).T)
            # # Oy = np.matmul(rotation_matrix.T, np.array([0, 1, 0]).T)
            # # Oz = np.matmul(rotation_matrix.T, np.array([0, 0, 1]).T)
            # # Equal the unit scale, some embellishment
            # f.write(str(frame_i) + "\n")

            # for i in range(3):
            #     for j in range(3):
            #         f.write(str(camera_intrinsic_matrix[i][j]) + "\t")
            #     f.write("\n")

            # # f.write("\n")

            # for i in range(3):
            #     for j in range(4):
            #         f.write(str(Extrinsic_matrix[i][j]) + "\t")
            #     f.write("\n")

            # f.write("\n")
            # +\
            #     "\t"+str(camera_pose[0])+"\t"+str(camera_pose[1])+"\t"+str(camera_pose[2])+\
            #     "\t"+str(Oz[0])+"\t"+str(Oz[1])+"\t"+str(Oz[2])+"\t\n")

            frame_i = frame_i + 1

        else:
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            matchesMask = None

# Show the plots
# plt.show()
