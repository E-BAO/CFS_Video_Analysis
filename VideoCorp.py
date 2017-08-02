import numpy as np
import cv2

cap = cv2.VideoCapture('EQ7/EQ7_Top.avi')
fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
c = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
r = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.cv.CV_FOURCC(*'XVID')
# video = cv2.VideoWriter('EQ7/EQ7_Top_UnPressed.avi',-1, fps, (int(c),int(r)))

frame_i = 3266
start_time = 0
end_time = 32

while(1):
    ret, frame_p = cap.read()
    if frame_p is None:
        break

    frame_i = frame_i + 1
    # if (frame_i <= fps * start_time):
    #     continue
    # if (frame_i > fps * end_time):
    #     break
    print frame_i

    # frame_p = cv2.cvtColor(frame_p, cv2.COLOR_BGR2GRAY)
    # frame_p = cv2.resize(frame_p, None, fx = 0.5, fy= 0.5)
    # cv2.imshow('img',frame_p)
    # cv2.waitKey(30)
    # video.write(frame_p)
    cv2.imwrite("EQ7/Frames/frame%d.png"%(frame_i),frame_p)

cap.release()