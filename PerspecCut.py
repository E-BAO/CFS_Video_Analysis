import numpy as np
import cv2

def expendPts(pts,offset = 40):
    ctX = np.sum(pts[:,0])/len(pts)
    ctY = np.sum(pts[:,1])/len(pts)
    points = []
    for point in pts:
        x, y = point
        dx,dy = x - ctX, y - ctY
        t = offset / np.sqrt(dx * dx + dy * dy)
        x = x + t * dx
        # if dy > 0:
        y = y + t * dy * 0.5
        pt = (x, y)
        points.append(pt)
    return points

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
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        pts = expendPts(dst.reshape(1,-1,2)[0], 30)

        dst = np.float32(pts).reshape(-1,1,2)

        cv2.fillPoly(cutMap,[np.int32(dst)],0)

        # cv2.fillPoly(cutMap,[np.int32(dst)],0)
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    points = []
    for ptArray in np.int32(dst).tolist():
        points.append(ptArray[0])

    return cutMap, np.int32(points)


def preProcess(img):
    #img = cv2.resize(img, None, fx = 0.5, fy=0.5)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    r,c = img.shape
    #img = img[180:r - 150,315:c - 200]
    # img = img[360:r - 500,1090:c - 600]
    #img = img[200:r,800:c - 600]
    # img[360:r - 430,980:c - 600] #EQ5
    img = img[250:r - 400,1000:c - 600] #EQ7
    #img = img[100:r - 100,250:c - 250]
    img_roof = cv2.imread('EQ7/roof.png',0)  # queryImage
    
    # img_frameTop = cv2.imread('top_frame.png',0)
    # img_frameLeft = cv2.imread('leftFrame.png',0)
    #img_right = cv2.imread('right_reference.png',0)  # queryImage
    img_cut, pts = cutConponent(img_roof,img)
    #img = cutConponent(img_framedown,img)
    #img = cutConponent(img_frameLeft,img)
    #img = cutConponent(img_frameTop,img)

    return img, img_cut, pts