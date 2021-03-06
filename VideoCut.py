import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import Matching

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for line,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        # x0,y0 = map(int, [0, -line[2]/line[1] ])
        # x1,y1 = map(int, [c, -(line[2]+line[0]*c)/line[1] ])
        # if abs(y0) > sys.maxsize or abs(y1) > sys.maxsize:
        #     continue
        #cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pt1),2,color,-1)
        cv2.circle(img2,tuple(pt2),2,color,-1)

    return img1,img2

cap = cv2.VideoCapture('EQ7/OFGPS.avi')

ret, frame = cap.read()
# img1 = Matching.preProcess(frame)
r,c,a = frame.shape
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
video  = cv2.VideoWriter('EQ7/PPT/OFGPS.mov', -1, fps, (c, r))


start_time = 5
end_time = 20
start_frame =  0
frame_i = 0#start_frame + int(fps * start_time)

while(1):
    ret, frame_p = cap.read()
    if frame_p is None:
        break
    print frame_i
    frame_i = frame_i + 1
    if (frame_i < start_frame + int(fps * start_time)):
        continue
    if (frame_i > start_frame + int(fps * end_time)):
        break
    video.write(frame_p)
    # cv2.imshow('video',frame_p)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
    
    # img2 = Matching.preProcess(frame_p)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # cv2.imwrite("EQ7/img2/frame%d.png"%(frame_i),img2)
    
    
    # orb = cv2.ORB()

    # # find the keypoints and descriptors with ORB
    # kp1, des1 = orb.detectAndCompute(img1,None)
    # kp2, des2 = orb.detectAndCompute(img2,None)

    # # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # # Match descriptors.
    # matches = bf.match(des1,des2)

    # good = []
    # pts1 = []
    # pts2 = []

    # D_MATCH_THRES = 65.0
    # for m in matches:
    #     if m.distance < D_MATCH_THRES:
    #         good.append(m)
    #         pts2.append(kp2[m.trainIdx].pt)
    #         pts1.append(kp1[m.queryIdx].pt)

    # pts1 = np.float32(pts1)
    # pts2 = np.float32(pts2)

    # # compute F
    # F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC) #cv2.FM_LMEDS

    # # We select only inlier points
    # pts1 = pts1[mask.ravel()==1]
    # pts2 = pts2[mask.ravel()==1]

    # # Find epilines corresponding to points in right image (second image) and
    # # drawing its lines on left image
    # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    # lines1 = lines1.reshape(-1,3)
    # img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    # # Find epilines corresponding to points in left image (first image) and
    # # drawing its lines on right image
    # lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    # lines2 = lines2.reshape(-1,3)
    # img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    # # img3 = cv2.resize(img3, None, fx = 0.3, fy=0.3)
    # cv2.imshow("img3",img3)
    #out.write(img3)

    #img1 = img2

    

# video.release()
cap.release()
cv2.destroyAllWindows()

# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()