import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import Matching

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

def drawlines(img1,pts1):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    for pt1 in pts1:
        color = tuple(np.random.randint(0,255,3).tolist())
        cv2.circle(img1,tuple(pt1),2,color,-1)

    return img1

cap = cv2.VideoCapture('video.avi')

ret, frame = cap.read()
img1 = Matching.preProcess(frame)
r,c = img1.shape
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)

frame_i = 0
start_time = 13
end_time = 55

while(1):
    ret, frame_p = cap.read()
    frame_i = frame_i + 1
    if (frame_i < fps * start_time):
        continue
    if (frame_i > fps * end_time):
        break
    print frame_i
    
    img2 = Matching.preProcess(frame_p)



    img2 = cv2.cvtColor(frame_p,cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    good = []
    pts1 = []
    pts2 = []

    D_MATCH_THRES = 65.0
    for m in matches:
        if m.distance < D_MATCH_THRES:
            pt = kp2[m.trainIdx].pt
            if is_point_in(pt[0],pt[1],points):
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    # compute F
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC) #cv2.FM_LMEDS

    #print F

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    # lines1 = lines1.reshape(-1,3)
    # img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    cv2.imshow("img3",img3)

    img1 = img2

    k = cv2.waitKey(0) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()