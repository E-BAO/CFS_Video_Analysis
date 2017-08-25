import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation
from matplotlib import pyplot as plt
from settings import *
import utilities as ut
from utilities import Arrow3D
import global_figure as gf

filename = "EQ7/Calibration/CalibrationDenoise.txt"

# Establish global figure
gf.fig = plt.figure()
gf.ax = gf.fig.add_subplot(111, projection='3d')


# Set axes labels
gf.ax.set_xlabel('X')
gf.ax.set_ylabel('Y')
gf.ax.set_zlabel('Z')
gf.ax.set_xlim3d(0, 25)
gf.ax.set_ylim3d(0, 25)
gf.ax.set_zlim3d(0, 50)

plt.gca().set_aspect('equal', adjustable='box')

pattern_image = cv2.imread("EQ7/frame3300_origin.png")
pattern_image = cv2.resize(pattern_image, None, fx = 0.1, fy=0.1)

# pattern_image = cv2.flip(cv2.transpose(pattern_image), 0)

# Plot pattern
height = pattern_image.shape[0]
width = pattern_image.shape[1]
xx, yy = np.meshgrid(np.linspace(0, width * 0.05, width), 
    np.linspace(height * 0.05,0, height))

X = xx
Y = yy
Z = 0

ax_image = gf.ax.plot_surface(X, Y, Z, rstride=2, cstride=2, facecolors=pattern_image / 255., shade=False)

camera_pose = [0,0,0]
camera_orientation = [0,0,0]

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=30, metadata=metadata)

flag = 0
with writer.saving(gf.fig, "EQ7/CameraMovement_denoise2.mp4", dpi = 100):
    with open(filename, "r") as f:
        ax_text = None

        while 1:
            line = f.readline() 
            if not line:
                break
                pass

            line = line[:line.find('\t\n')]

            # part1, part2 = [str(i) for i in line.split('\t\t')]
            frame_i, camera_pose[0], camera_pose[1], camera_pose[2],\
             _,_,_,_,_,_,\
             camera_orientation[0], camera_orientation[1], camera_orientation[2] \
             = [float(i) for i in line.split('\t')]
             # = [float(i) for i in part2.split('\t')]
            
            # frame_i, camera_pose[0], camera_pose[1], camera_pose[2],\
            #     camera_orientation[0], camera_orientation[1], camera_orientation[2]\
            #     = [float(i) for i in line.split('\t')]

            print frame_i

            # gf.ax.cla()

            if flag:
                # gf.ax.texts.remove(ax_text)
                gf.ax.collections.remove(ax_camera)
                gf.ax.artists.remove(arrow)

            pixel2inch = 0.345

            # label = '%d (%5d, %5d, %5d)' % (frame_i, camera_pose[0], camera_pose[1], camera_pose[2])
            # ax_text = gf.ax.text(camera_pose[0], camera_pose[1], camera_pose[2], label)

            camera_pose = [i/200 for i in camera_pose]
            camera_pose[2] = - camera_pose[2]
            camera_orientation[2] = -camera_orientation[2]

            # max_unit_length = max(30, max(camera_pose[:3])) + 30

            # Decompose the camera coordinate
            arrow_length =  - camera_pose[2] / camera_orientation[2]
            xs = [camera_pose[0], camera_pose[0] + camera_orientation[0] * arrow_length]
            ys = [camera_pose[1], camera_pose[1] + camera_orientation[1] * arrow_length]
            zs = [camera_pose[2], 0]

            # Plot camera location
            ax_camera = gf.ax.scatter([camera_pose[0]], [camera_pose[1]], [camera_pose[2]],color='blue')
            # item.append(ax_camera)
            # item.append(ax_label)

            arrow = Arrow3D(xs, ys, zs, mutation_scale=5, lw=1, arrowstyle="-|>", color="r")
            gf.ax.add_artist(arrow)

            # Prepare pattern image
            # To cater to the settings of matplotlib, 
            # we need the second line here to rotate the image 90 degree counterclockwise
            # ax_image = gf.ax.plot_surface(X, Y, Z, rstride=2, cstride=2, facecolors=pattern_image / 255., shade=False)

            flag = 1
            # plt.pause(.0001)

            writer.grab_frame()