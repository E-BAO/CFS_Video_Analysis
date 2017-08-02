import matplotlib.image as mping
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    for pt1 in pts1:
        color = tuple(np.random.randint(0,255,3).tolist())
        # x0,y0 = map(int, [0, -line[2]/line[1] ])
        # x1,y1 = map(int, [c, -(line[2]+line[0]*c)/line[1] ])
        # if abs(y0) > sys.maxsize or abs(y1) > sys.maxsize:
        #     continue
        #cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pt1),2,color,-1)

    return img1

img1 = cv2.imread("img/frame403.png",0)
img2 = cv2.imread("img/frame390.png",0)


# cv2.imshow("frame2",img2)
#img2 = cv2.cvtColor(frame_p, cv2.COLOR_RGB2GRAY)
# cv2.imshow("img3",img2)

#Initiate SIFT detector
sift = cv2.SIFT()
MIN_MATCH_COUNT = 10
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)


# store all the good matches as per Lowe's ratio test.
good = []
points = [(100,100),(300,80),(400,200),(80,230)]

x0 = 100
x1 = 300

for m,n in matches:
    if m.distance < 0.7*n.distance:
        pt = kp2[m.trainIdx].pt
        if is_point_in(pt[0],pt[1],points):
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
    #img2 = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
    #img11 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
    pts1 = np.array([(100,100),(300,80),(400,200),(80,230)], np.int32)
    pts1 = pts1.reshape((-1,1,2))
    cv2.polylines(img2,[pts1],True,255,3,cv2.CV_AA)
    cv2.imshow("img2",img2)
    cv2.waitKey(0)
else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
