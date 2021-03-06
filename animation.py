import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import cv2  

filename = "EQ7/Calibration/Calibrate_Noise3_Inch.txt"

X0 = []
X1 = []
X2 = []
X3 = []
Y0 = []
Y1 = []
Y2 = []
Y3 = []

with open(filename, "r") as f:
  while 1:
    line = f.readline() 
    if not line:
        break
        pass
    line = line[:line.find('\t\n')]
    frame_i, x0, y0, x1, y1, x2, y2 = [float(i) for i in line.split('\t')]
    X0.append(x0)
    X1.append(x1)
    X2.append(x2)
    # X3.append(x3)
    Y0.append(y0)
    Y1.append(y1)
    Y2.append(y2)
    # Y3.append(y3)
    pass

# first set up the figure, the axis, and the plot element we want to animate   

# fig = plt.figure() # make figure

# # make axesimage object
# # the vmin and vmax here are very important to get the color map correct
# r,c,a = frame.shape

# im = plt.imshow(frame, cmap=plt.get_cmap('jet'), vmin=0, vmax=855)

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=30, metadata=metadata)

font = {'weight' : 'normal','size' : 4}
matplotlib.rc('font', **font)

fig = plt.figure()

fig.suptitle('Video Data Analysis of Roof Level', fontsize=12)

ax1 = fig.add_subplot(2,2,1,xlim=(0, len(X0)/30.), ylim=(-30, 50)) 
ax1.set_title('X Relative Displacement',fontweight="bold", size=6) # Title
ax1.set_ylabel('Deplacement(inch)', fontsize = 6) # Y label
ax1.set_xlabel('time(sec)', fontsize = 6) # X label

line0, = ax1.plot([], [], lw=1,label='BottomLeft',color = (0,0,1))   
# line3, = ax1.plot([], [], lw=1,label='BottomRight', color = (0.8,0.8,0))    
line1, = ax1.plot([], [], lw=1,label='TopLeft',color = (1,0,0))   
line2, = ax1.plot([], [], lw=1,label='TopRight',color = (0,1,0))   
ax1.legend(loc='upper right')
# text1 = ax1.text(80, 60, str(i), fontsize=6)


ax2 = fig.add_subplot(2,2,2,xlim=(0, len(X0)/30.), ylim=(-30, 50))
ax2.set_title('Y Relative Displacement',fontweight="bold", size=6) # Title
ax2.set_ylabel('Deplacement(inch)', fontsize = 6) # Y label
ax2.set_xlabel('time(sec)', fontsize = 6) # X label

line4, = ax2.plot([], [], lw=1,label='BottomLeft',color = (0,0,1))   
# line7, = ax2.plot([], [], lw=1,label='BottomRight', color = (0.8,0.8,0))    
line5, = ax2.plot([], [], lw=1,label='TopLeft',color = (1,0,0))   
line6, = ax2.plot([], [], lw=1,label='TopRight',color = (0,1,0))   
ax2.legend(loc='upper right')
# text2 = ax2.text(80, 40, str(i), fontsize=6)

frame_i = 3267

# cap1 = cv2.VideoCapture("EQ7/OFCneter1.avi")
frame1 = cv2.imread("EQ7/CalibrationDenoise3t/frame3267.png")
# ret, frame1 = cap1.read()
r, c, a = frame1.shape
# frame1 = frame1[220:r -120,150:c - 200,:]

cap0 = cv2.VideoCapture("EQ7/EQ7_Top.avi")

# for i in range(3266):
ret, frame0 = cap0.read()

# frame0 = cv2.imread("EQ7/Frames/frame3267.png")

im00 = fig.add_subplot(2,2,3,title = "Original Video")#xlim=(0, 3840), ylim=(0, 2160)
im00.axis('off')
im00.set_title('Original',fontweight="bold", size=6) # Title
im0 = plt.imshow(frame0, cmap=plt.get_cmap('jet'), vmin=0, vmax=855)

im11 = fig.add_subplot(2,2,4,title = "Processed Video")#xlim=(0, 1500), ylim=(0, 970),
im11.axis('off')
im11.set_title('Processed Video',fontweight="bold", size=6) # Title
im1 = plt.imshow(frame1, cmap=plt.get_cmap('jet'), vmin=0, vmax=855)

plt.tight_layout(pad=10.08, w_pad=3.0, h_pad=3.0)

# def init():   
#     line1.set_data([], [])     
#     return line1,[im]
  
# # animation function.  this is called sequentially     
# def animate(i):  
#     #i = 500
#     x = np.linspace(1, len(X0), len(X0))
#     temp = [X0[0] for k in range(len(X0))]
#     y = [X0[k] - temp[k] for k in range(len(X0))]
#     n = len(X0) - i
#     y[i:] = [0 for k in range(n)]
#     line0.set_data(x, y)   

#     temp = [X1[0] for k in range(len(X0))]
#     y = [X1[k] - temp[k] for k in range(len(X0))]
#     y[i:] = [0 for k in range(n)]
#     line1.set_data(x, y)   

#     temp = [X2[0] for k in range(len(X0))]
#     y = [X2[k] - temp[k] for k in range(len(X0))]
#     y[i:] = [0 for k in range(n)]
#     line2.set_data(x, y)   

#     temp = [X3[0] for k in range(len(X0))]
#     y = [X3[k] - temp[k] for k in range(len(X0))]
#     y[i:] = [0 for k in range(n)]
#     line3.set_data(x, y)   

#     frame_i = 720 + i
#     print frame_i
#     frame = cv2.imread("Preprocess2/frame%d.png"%(frame_i))
#     im.set_array(frame)
#     return (line0,line1,line2,line3) ,[im]
  
# anim1=animation.FuncAnimation(fig, animate, init_func=init,  frames=len(X0), interval=30)  

# cap1 = cv2.VideoCapture('EQ7/OFCneter_denoise.avi')

# centerX, centerY = 800,460

with writer.saving(fig, "EQ7/Calibration/OFGPScali3.mp4", len(X0)):
    x = np.linspace(0, len(X0)/30., len(X0))

    for i in range(len(X0)):    
        # for i in range(len(X0)):
            # text1.set_text(str(X0[i0]))
            # text2.set_text(str(Y0[i]))
            
        # xMax = len(X0)
        # xN = len(X0)
        temp = [X0[0] for k in range(len(X0))]
        y = [X0[k] - temp[k] for k in range(len(X0))]
        n = len(X0) - i
        y[i:] = [0 for k in range(n)]
        line0.set_data(x, y)   

        temp = [X1[0] for k in range(len(X0))]
        y = [X1[k] - temp[k] for k in range(len(X0))]
        y[i:] = [0 for k in range(n)]
        line1.set_data(x, y)   

        temp = [X2[0] for k in range(len(X0))]
        y = [X2[k] - temp[k] for k in range(len(X0))]
        y[i:] = [0 for k in range(n)]
        line2.set_data(x, y)   

        # temp = [X3[0] for k in range(len(X0))]
        # y = [X3[k] - temp[k] for k in range(len(X0))]
        # y[i:] = [0 for k in range(n)]
        # line3.set_data(x, y)   

        temp = [Y0[0] for k in range(len(X0))]
        y = [Y0[k] - temp[k] for k in range(len(X0))]
        y[i:] = [0 for k in range(n)]
        line4.set_data(x, y)   

        temp = [Y1[0] for k in range(len(X0))]
        y = [Y1[k] - temp[k] for k in range(len(X0))]
        y[i:] = [0 for k in range(n)]
        line5.set_data(x, y)   

        temp = [Y2[0] for k in range(len(X0))]
        y = [Y2[k] - temp[k] for k in range(len(X0))]
        y[i:] = [0 for k in range(n)]
        line6.set_data(x, y)   

        # temp = [Y3[0] for k in range(len(X0))]
        # y = [Y3[k] - temp[k] for k in range(len(X0))]
        # y[i:] = [0 for k in range(n)]
        # line7.set_data(x, y)  

        frame_i = 3267 + i
        print frame_i
        frame1 = cv2.imread("EQ7/CalibrationDenoise3t/frame%d.png"%(frame_i))
        # ret, frame1 = cap1.read()
        if frame1 is None:
            break
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        # frame1 = frame1[220:r -120,150:c - 200,:]

        ret, frame0 = cap0.read()
        # frame = cv2.imread("EQ7/Frames/frame%d.png"%(frame_i))
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        im0.set_array(frame0)
        im1.set_array(frame1)

        # plt.tight_layout(pad=3.08, h_pad=2.0)

        # plt.show()


        writer.grab_frame()
    # return (line0,line1,line2,line3) ,[im]  

# cap.release()
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


