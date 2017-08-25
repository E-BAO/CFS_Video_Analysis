import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import cv2  

filename = "EQ7/Calibration/CalibrationDenoiseGPS_Inch.txt"

X0 = []
X1 = []
X2 = []
Y0 = []
Y1 = []
Y2 = []

with open(filename, "r") as f:
  while 1:
    line = f.readline() 
    if not line:
        break
        pass
    line = line[:line.find('\t\n')]
    frame_i, x0, y0, x1, y1, x2, y2= [float(i) for i in line.split('\t')]
    X0.append(x0)
    X1.append(x1)
    X2.append(x2)
    Y0.append(y0)
    Y1.append(y1)
    Y2.append(y2)
    pass

def denoise(X0):
    x = np.linspace(1, len(X0), len(X0))
    y = [X0[k] for k in range(len(X0))]
    rft = np.fft.rfft(y)
    rft[20:] = 0
    y_smooth = np.fft.irfft(rft)
    a = y_smooth[len(y_smooth) - 1:]
    y_smooth = np.append(y_smooth,a)  
    return y_smooth


X0 = denoise(X0)
X1 = denoise(X1)
X2 = denoise(X2)
# X3 = denoise(X3)

Y0 = denoise(Y0)
Y1 = denoise(Y1)
Y2 = denoise(Y2)

filename = "EQ7/GPSRecord_Prcss.txt"

RX0 = []
RX1 = []
RX2 = []
RY0 = []
RY1 = []
RY2 = []

with open(filename, "r") as f:
  while 1:
    line = f.readline() 
    if not line:
        break
        pass
    line = line[:line.find('\t\n')]
    frame_i, x0, y0, x1, y1, x2, y2= [float(i) for i in line.split('\t')]
    RX0.append(x0)
    RX1.append(x1)
    RX2.append(x2)
    RY0.append(y0)
    RY1.append(y1)
    RY2.append(y2)
    pass

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=30, metadata=metadata)

font = {'weight' : 'normal','size' : 4}
matplotlib.rc('font', **font)

fig = plt.figure()

fig.suptitle('Video Data Analysis of Roof Level', fontsize=12)

ax1 = fig.add_subplot(2,2,1,xlim=(1, len(X0)), ylim=(-20, 40)) 
ax1.set_title('X Relative Displacement',fontweight="bold", size=6) # Title
ax1.set_ylabel('Deplacement(inch)', fontsize = 6) # Y label
ax1.set_xlabel('time(frame)', fontsize = 6) # X label

line0, = ax1.plot([], [], lw=1,label='VIDEO',color = (1,0,0))   
line1, = ax1.plot([], [], lw=1,label='GPS',color = (0,0,1))   
# line2, = ax1.plot([], [], lw=1,label='BottomRight',color = (0,1,0))   
# line3, = ax1.plot([], [], lw=1,label='BottomLeft')    
ax1.legend(loc='upper right')
text1 = ax1.text(40, 35, str(i), fontsize=6)


ax2 = fig.add_subplot(2,2,2,xlim=(1, len(X0)), ylim=(-10, 20))
ax2.set_title('Y Relative Displacement',fontweight="bold", size=6) # Title
ax2.set_ylabel('Deplacement(inch)', fontsize = 6) # Y label
ax2.set_xlabel('time(frame)', fontsize = 6) # X label

line4, = ax2.plot([], [], lw=1,label='VIDEO',color = (1,0,0))   
line5, = ax2.plot([], [], lw=1,label='GPS',color = (0,0,1))   
# line6, = ax2.plot([], [], lw=1,label='BottomRight',color = (0,1,0))   
# line7, = ax2.plot([], [], lw=1,label='BottomLeft')    
ax2.legend(loc='upper right')
text2 = ax2.text(40, 18, str(i), fontsize=6)

frame_i = 3267

frame = cv2.imread("EQ7/PPT/frame3267.png")

im00 = fig.add_subplot(2,2,3,title = "Original Video")#xlim=(0, 3840), ylim=(0, 2160)
im00.axis('off')
im00.set_title('Original',fontweight="bold", size=6) # Title
im0 = plt.imshow(frame, cmap=plt.get_cmap('jet'), vmin=0, vmax=855)

im11 = fig.add_subplot(2,2,4,title = "Processed Video")#xlim=(0, 1500), ylim=(0, 970),
im11.axis('off')
im11.set_title('Processed',fontweight="bold", size=6) # Title
im1 = plt.imshow(frame, cmap=plt.get_cmap('jet'), vmin=0, vmax=855)

plt.tight_layout(pad=10.08, w_pad=3.0, h_pad=3.0)

cap0 = cv2.VideoCapture("EQ7/EQ7_Top.avi")

# cap1 = cv2.VideoCapture('EQ7/OFGPS.avi')

with writer.saving(fig, "EQ7/Calibration/OFGPSct.mp4", len(X0)):
    for i in range(len(X0)):
        text1.set_text(str(X1[i]))
        text2.set_text(str(Y1[i]))

        x = np.linspace(1, len(X0), len(X0))
        temp = [X0[0] for k in range(len(X0))]
        y1 = [(X0[k] - temp[k]) for k in range(len(X0))]
        n = len(X0) - i
        # y1[i:] = [0 for k in range(n)]
        line0.set_data(x, y1)   

        # temp = [X1[0] for k in range(len(X0))]
        # y = [X1[k] - temp[k] for k in range(len(X0))]
        # y[i:] = [0 for k in range(n)]
        temp = [RX0[0] for k in range(len(X0))]
        y2 = [ - RX0[k] + temp[k] for k in range(len(X0))]
        # y2[i:] = [0 for k in range(n)]
        line1.set_data(x, y2)   

        # y = [(y1[k] - y2[k]) for k in range(len(X0))]
        # line2.set_data(x, y)   

        # temp = [X3[0] for k in range(len(X0))]
        # y = [X3[k] - temp[k] for k in range(len(X0))]
        # y[i:] = [0 for k in range(n)]
        # line3.set_data(x, y)   

        temp = [Y0[0] for k in range(len(X0))]
        y = [ (- Y0[k] + temp[k]) for k in range(len(X0))]
        # y[i:] = [0 for k in range(n)]
        line4.set_data(x, y)   

        temp = [RY0[0] for k in range(len(X0))]
        y = [- RY0[k] + temp[k] for k in range(len(X0))]
        # y[i:] = [0 for k in range(n)]
        line5.set_data(x, y)   

        # temp = [RX2[0] for k in range(len(X0))]
        # y = [- RX2[k] + temp[k] for k in range(len(X0))]
        # y[i:] = [0 for k in range(n)]
        # line6.set_data(x, y)   

        # temp = [Y3[0] for k in range(len(X0))]
        # y = [Y3[k] - temp[k] for k in range(len(X0))]
        # y[i:] = [0 for k in range(n)]
        # line7.set_data(x, y)  

        frame_i = 3267 + i
        print frame_i
        frame1 = cv2.imread("EQ7/PPT/frame%d.png"%(frame_i))
        # ret, frame1 = cap1.read()
        if frame1 is None:
            break
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        ret, frame0 = cap0.read()
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        # frame = cv2.imread("Preprocess2/frame%d.png"%(frame_i))
        im0.set_array(frame0)
        im1.set_array(frame1)

        writer.grab_frame()
    # return (line0,line1,line2,line3) ,[im]  

cap.release()
# fig = plt.figure() # make figure

# # make axesimage object
# # the vmin and vmax here are very important to get the color map correct
# r,c,a = frame.shape

# im = plt.imshow(frame, cmap=plt.get_cmap('jet'), vmin=0, vmax=855)

# # function to update figure
# def updatefig(j):
#     # set the data in the axesimage object
#     frame_i = 720 + j
#     print frame_i
#     frame = cv2.imread("Preprocess2/frame%d.png"%(frame_i))
#     im.set_array(frame)
#     # return the artists set
#     return [im]
# # kick off the animation
# ani = animation.FuncAnimation(fig, updatefig, frames=range(len(X)), interval=30, blit=True)

# plt.show()


