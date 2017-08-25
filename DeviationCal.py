import numpy as np
import matplotlib
import matplotlib.pyplot as plt

filename = "EQ7/displacementGPS_normalize.txt"

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

# font = {'weight' : 'normal','size' : 4}
# matplotlib.rc('font', **font)

fig = plt.figure()

fig.suptitle('Deplacement Deviation of GPS', fontsize=12)

ax1 = fig.add_subplot(221,xlim=(1, len(X0)), ylim=(-10, 15)) 
ax1.set_title('X Deviation',fontweight="bold", size=6) # Title
ax1.set_ylabel('Deviation(inch)', fontsize = 6) # Y label
ax1.set_xlabel('time(frame)', fontsize = 6) # X label

line0, = ax1.plot([], [], lw=1,label='Center',color = (1,0,0)) 
line1, = ax1.plot([], [], lw=1,label='Right Top',color = (0,0,1)) 
line2, = ax1.plot([], [], lw=1,label='Right Bottom',color = (0,1,0)) 
lineBase, = ax1.plot([], [], lw=1, color = (0,0,0)) 

x = np.linspace(1, len(X0), len(X0))
y = [(X0[k] - X0[0] + RX0[k] - RX0[0]) for k in range(len(X0))]
line0.set_data(x, y)   

y = [(X1[k] - X1[0] + RX1[k] - RX1[0]) for k in range(len(X0))]
line1.set_data(x, y)   

y = [(X2[k] - X2[0] + RX2[k] - RX2[0]) for k in range(len(X0))]
line2.set_data(x, y)   

y = [0 for k in range(len(X0))]
lineBase.set_data(x, y)   

ax1.legend()

ax2 = fig.add_subplot(222,xlim=(1, len(X0)), ylim=(-15, 10)) 
ax2.set_title('Y Deviation',fontweight="bold", size=6) # Title
ax2.set_ylabel('Deviation(inch)', fontsize = 6) # Y label
ax2.set_xlabel('time(frame)', fontsize = 6) # X label

line3, = ax2.plot([], [], lw=1,label='Center',color = (1,0,0)) 
line4, = ax2.plot([], [], lw=1,label='Right Top',color = (0,0,1)) 
line5, = ax2.plot([], [], lw=1,label='Right Bottom',color = (0,1,0)) 
lineBase, = ax2.plot([], [], lw=1, color = (0,0,0)) 

x = np.linspace(1, len(X0), len(X0))
y = [(Y0[k] - Y0[0] - RY0[k] + RY0[0]) for k in range(len(X0))]
line3.set_data(x, y)   

y = [(Y1[k] - Y1[0] - RY1[k] + RY1[0]) for k in range(len(X0))]
line4.set_data(x, y)   

y = [(Y2[k] - Y2[0] - RY2[k] + RY2[0]) for k in range(len(X0))]
line5.set_data(x, y)   

y = [0 for k in range(len(X0))]
lineBase.set_data(x, y)  

ax2.legend()

ax3 = fig.add_subplot(223,xlim=(1, len(X0)), ylim=(-10, 15)) 
ax3.set_title('X Deviation',fontweight="bold", size=6) # Title
ax3.set_ylabel('Deviation(inch)', fontsize = 6) # Y label
ax3.set_xlabel('time(frame)', fontsize = 6) # X label

line0, = ax3.plot([], [], lw=1,label='Center',color = (1,0,0)) 
line1, = ax3.plot([], [], lw=1,label='Right Top',color = (0,0,1)) 
line2, = ax3.plot([], [], lw=1,label='Right Bottom',color = (0,1,0)) 
lineBase, = ax3.plot([], [], lw=1, color = (0,0,0)) 

x = np.linspace(1, len(X0), len(X0))
y = [(X0[k] - X0[0] + RX0[k] - RX0[0]) for k in range(len(X0))]
# z1 = np.polyfit(x, y, 20)
# p1 = np.poly1d(z1)
rft = np.fft.rfft(y)
rft[30:] = 0
y_smooth = np.fft.irfft(rft)
a = y_smooth[len(y_smooth) - 1:]
y_smooth = np.append(y_smooth,a)  
line0.set_data(x, y_smooth)   

y = [(X1[k] - X1[0] + RX1[k] - RX1[0]) for k in range(len(X0))]
# z1 = np.polyfit(x, y, 20)
# p1 = np.poly1d(z1)
rft = np.fft.rfft(y)
rft[30:] = 0
y_smooth = np.fft.irfft(rft)
a = y_smooth[len(y_smooth) - 1:]
y_smooth = np.append(y_smooth,a)
line1.set_data(x, y_smooth)   

y = [(X2[k] - X2[0] + RX2[k] - RX2[0]) for k in range(len(X0))]
rft = np.fft.rfft(y)
rft[30:] = 0
y_smooth = np.fft.irfft(rft)
a = y_smooth[len(y_smooth) - 1:]
y_smooth = np.append(y_smooth,a) 
line2.set_data(x, y_smooth)  

y = [0 for k in range(len(X0))]
lineBase.set_data(x, y)   

ax3.legend()

ax4 = fig.add_subplot(224,xlim=(1, len(X0)), ylim=(-15, 10)) 
ax4.set_title('Y Deviation',fontweight="bold", size=6) # Title
ax4.set_ylabel('Deviation(inch)', fontsize = 6) # Y label
ax4.set_xlabel('time(frame)', fontsize = 6) # X label

line3, = ax4.plot([], [], lw=1,label='Center',color = (1,0,0)) 
line4, = ax4.plot([], [], lw=1,label='Right Top',color = (0,0,1)) 
line5, = ax4.plot([], [], lw=1,label='Right Bottom',color = (0,1,0)) 
lineBase, = ax4.plot([], [], lw=1, color = (0,0,0)) 

x = np.linspace(1, len(X0), len(X0))
y = [(Y0[k] - Y0[0] - RY0[k] + RY0[0]) for k in range(len(X0))]
rft = np.fft.rfft(y)
rft[30:] = 0
y_smooth = np.fft.irfft(rft)
a = y_smooth[len(y_smooth) - 1:]
y_smooth = np.append(y_smooth,a)
line3.set_data(x, y_smooth)   

y = [(Y1[k] - Y1[0] - RY1[k] + RY1[0]) for k in range(len(X0))]
rft = np.fft.rfft(y)
rft[30:] = 0
y_smooth = np.fft.irfft(rft)
a = y_smooth[len(y_smooth) - 1:]
y_smooth = np.append(y_smooth,a)
line4.set_data(x, y_smooth)  

y = [(Y2[k] - Y2[0] - RY2[k] + RY2[0]) for k in range(len(X0))]
rft = np.fft.rfft(y)
rft[30:] = 0
y_smooth = np.fft.irfft(rft)
a = y_smooth[len(y_smooth) - 1:]
y_smooth = np.append(y_smooth,a)
line5.set_data(x, y_smooth)    

y = [0 for k in range(len(X0))]
lineBase.set_data(x, y)  

ax4.legend()

plt.tight_layout(pad=3.08, h_pad=2.0)

plt.show()

