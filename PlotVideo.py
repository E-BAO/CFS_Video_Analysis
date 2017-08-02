# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# import matplotlib.animation as animation


# frame = cv2.imread("Preprocess2/frame720.png")

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
# ani = animation.FuncAnimation(fig, updatefig, frames=range(958), interval=1, blit=True)
# plt.show()


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
l, = plt.plot([], [], 'k-o')

plt.xlim(-5, 5)
plt.ylim(-5, 5)

x0, y0 = 0, 0

with writer.saving(fig, "writer_test.mp4", 100):
    for i in range(100):
        x0 += 0.1 * np.random.randn()
        y0 += 0.1 * np.random.randn()
        l.set_data(x0, y0)
        writer.grab_frame()