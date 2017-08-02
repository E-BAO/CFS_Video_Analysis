import numpy as np
import matplotlib.pyplot as plt
import cv2

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 20,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,3))

old_frame = cv2.imread("Preprocess/frame720.png")
r,c,a = old_frame.shape
old_frame = old_frame[280:r - 60,100:c - 630]
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


X = np.random.rand(100, 1000)
xs = p0[:,0,0]
ys = p0[:,0,1]

fig, ax = plt.subplots()
ax.set_title('click on point to plot time series')
line, = ax.plot(xs, ys, 'o', picker=5)  # 5 points tolerance

NUM = 3
N = 0
p1 = p0[:NUM,:,:]

def onpick(event):
    global N
    global p1
    if N > NUM - 1:
        print("enough pts selected",p1)
        return

    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick points:', ind, points)
    p1[N,:,:] = p0[ind[0],:,:]
    N = N + 1

fig.canvas.mpl_connect('pick_event', onpick)
plt.imshow(old_frame)
plt.show()

cv2.waitKey(0)