import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import PerspecCut as ps

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
        cv2.circle(img1,tuple(pt1),2,color,-1)

    return img1

def cutConponent(img1,img2):
    cutMap = img2.copy()
    MIN_MATCH_COUNT = 10
    #img2 = cv2.imread('test.png',0) # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(cutMap,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)

    points = []
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        cv2.fillPoly(cutMap,[np.int32(dst)],0)

        for ptArray in dst.tolist():
            points.append(ptArray[0])

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    return cutMap, np.float32(points)

cap = cv2.VideoCapture('EQ5_Top.MOV')

# ret, frame = cap.read()
img1 = cv2.imread("frame394.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
r,c = img1.shape
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
#video  = cv2.VideoWriter('Preprocess2/video07_07.avi', -1, fps, (c, r));

# img1 = cv2.imread('Ground.png',0)          # queryImage

frame_i = 0
start_time = 25 - 1
end_time = 55 + 1
MIN_MATCH_COUNT = 10

while(1):
    ret, frame_p = cap.read()
    if frame_p is None:
        break

    frame_i = frame_i + 1
    if (frame_i < fps * start_time):
        continue
    if (frame_i > fps * end_time):
        break
    print frame_i
    

    # img2 = cv2.cvtColor(frame_p, cv2.COLOR_GRAY2RGB)
    img2,img2_Cut,pts2 = ps.preProcess(frame_p)
    pts2 = expendPts(pts2,40)
    # cv2.imshow("frame2",img2)
    #img2 = cv2.cvtColor(frame_p, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("img3",img2)

    #Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
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
        dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        img2 = drawlines(img2,np.float32([ kp2[m.trainIdx].pt for m in good ]))

        # matchesMask = mask.ravel().tolist()
        # h,w = img1.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv2.perspectiveTransform(pts,M)
        #pts_draw = np.array(pts2, np.int32)
        #pts_draw = pts_draw.reshape((-1,1,2))
        #cv2.polylines(img2,[pts_draw],True,255,1,cv2.CV_AA)

        img2 = cv2.warpPerspective(img2, M, (img2.shape[1], img2.shape[0]))
        #img11 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    #imgshow = cv2.resize(img2, None, fx = 0.2, fy= 0.2)
    #cv2.imshow("frame_Persp",imgshow)
    cv2.imwrite("Preprocess2/frame%d.png"%(frame_i),img2)
    #video.write(img2)
    # img1 = img2
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

#video.release()
cap.release()
cv2.destroyAllWindows()


# cap = cv2.VideoCapture('Preprocess/video07_07.avi')

# marker0 = cv2.imread("Marker/Marker0.png",0)
# marker1 = cv2.imread("Marker/Marker1.png",0)
# marker2 = cv2.imread("Marker/Marker2.png",0)

# frame_i = 0

# filename = "replacement.txt"

# with open(filename, "w") as f:
#     while(1):
#         frame_i = frame_i + 1
#         print frame_i
#         ret, frame_p = cap.read()

#         k = cv2.waitKey(30) & 0xff
#         if k == 27:
#             break

#         img1 = cv2.cvtColor(frame_p,cv2.COLOR_RGB2GRAY)

#         img_cut0, pts0 = cutConponent(marker0,img1)
#         img_cut1, pts1 = cutConponent(marker1,img1)
#         img_cut2, pts2 = cutConponent(marker2,img1)

#         img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#         cntX0, cntY0 = findCenter(img1,pts0,color = (0,255,0))
#         cntX1, cntY1 = findCenter(img1,pts1,color = (255,0,0))
#         cntX2, cntY2 = findCenter(img1,pts2,color = (0,0,255))

#         f.write(str(frame_i)+"\t"+\
#                 str(cntX0)+"\t"+str(cntY0)+"\t"+ \
#                 str(cntX1)+"\t"+str(cntY1)+"\t"+ \
#                 str(cntX2)+"\t"+str(cntY2)+"\t"+ "\n")

#         # imgshow = cv2.resize(img1, None, fx = 0.5, fy= 0.5)
#         # cv2.imshow("img_Marker",imgshow)

# cap.release()
# cv2.destroyAllWindows()

# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()