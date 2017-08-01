import numpy as np
import cv2
from matplotlib import pyplot as plt
import drawMatches

def expendPts(pts,offset = 40):
    ctX = np.sum(pts[:,0])/len(pts)
    ctY = np.sum(pts[:,1])/len(pts)
    points = []
    for point in pts:
        x, y = point
        dx,dy = x - ctX, y - ctY
        t = offset / np.sqrt(dx * dx + dy * dy)
        x = x + t * dx * 0.2
        if dy > 0:
            y = y + t * dy * 6.0
        pt = (x, y)
        points.append(pt)
    return points

def cutConponent(img1,img2):
    MIN_MATCH_COUNT = 10
    #img2 = cv2.imread('test.png',0) # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT()

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
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,100],[200,100],[200,0] ]).reshape(-1,1,2)
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        # points = []
        # for ptArray in np.float32(dst).tolist():
        #     points.append(ptArray[0])

        # pts = expendPts(dst.reshape(1,-1,2)[0])

        # dst = np.float32(pts).reshape(-1,1,2)

        cv2.fillPoly(img2,[np.int32(dst)],0)
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    return img2

def preProcess(img):
    #img = cv2.resize(img, None, fx = 1.0, fy= 1.0)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    r,c = img.shape
    # img = img[180:r - 150,315:c - 300]
    # img = img[180:r - 250,545:c - 300]
    # img = img[360:r - 500,1090:c - 600]
    # img = img[360:r - 300,1090:c - 600]
    img = img[360:r - 430,980:c - 600]
    #img = img[360:r - 160,980:c - 600]
    #img = img[360:r - 180,1000:c - 1500]

    img_roof = cv2.imread('roof.png',0)  # queryImage
    img_framedown = cv2.imread('RD.png',0)
    # img_framedown = cv2.imread('frames.png',0)
    # img_frameTop = cv2.imread('top_frame.png',0)
    # img_frameLeft = cv2.imread('leftFrame.png',0)
    #img_right = cv2.imread('right_reference.png',0)  # queryImage
    img = cutConponent(img_roof,img)
    #img = cutConponent(img_framedown,img)
    #img = cutConponent(img_frameLeft,img)
    #img = cutConponent(img_frameTop,img)
    return img