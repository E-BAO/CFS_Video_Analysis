import numpy as np
import cv2
from matplotlib import pyplot as plt
import drawMatches as dm

MIN_MATCH_COUNT = 10

img1 = cv2.imread('roof.png',0)          # queryImage
img2 = cv2.imread('test.png',0) # trainImage

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
    if m.distance < 0.3*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    print img2.shape
    print [np.int32(dst)]
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
    cv2.polylines(img2,[np.int32(dst)],True,(0,255,0),3,cv2.CV_AA)
    #cv2.fillPoly(img2,[np.int32(dst)],255)
    print img2.shape

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)


# img3 = dm.drawMatches(img1,kp1,img2,kp2,good)

plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
plt.show()