from matplotlib import pyplot as plt
from celluloid import Camera

"""
fig = plt.figure()
camera = Camera(fig)
for i in range(4):
	if i !=0:
		circle.remove()
	circle = plt.Circle((0.5, 0.5), i*0.1, color='blue', fill=False)
	fig.add_artist(circle)
	camera.snap()
animation = camera.animate()
animation.save('celluloid_minimal.gif', writer = 'PillowWriter')
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import types
import time

from fourier import *
from image_processing import *

t0 = time.time()
timed = False

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(5, 5)

scale = 500
ax = plt.axes(xlim=(-scale, scale), ylim=(-scale, scale))


def add_epicycle(self, epicycle):
    self.add_patch(epicycle.circle)
    self.add_line(epicycle.line)


ax.add_epicycle = types.MethodType(add_epicycle, ax)

radius = 5
patch = plt.Circle((0, 0), radius, fc="y", fill=False)
# line = plt.Line2D((0,radius),(0,0))


class epicycle:
    def __init__(self, r, w, x=0, y=0, theta=0):
        self.circle = plt.Circle((x, y), r, fill=False, linewidth=0.4)
        self.line = plt.Line2D(
            (x, x + r * np.sin(theta)),
            (y, y + r * np.cos(theta)),
            linewidth=0.4,
        )
        self.r = r
        self.w = w
        self.theta = theta

    def get_artists(self):
        return [self.circle, self.line]

    def set_centre(self, xy):
        x = xy[0]
        y = xy[1]
        self.circle.set_center((x, y))
        x_data = self.line.get_xdata()
        y_data = self.line.get_ydata()
        self.line.set_xdata((x, x + x_data[1] - x_data[0]))
        self.line.set_ydata((y, y + y_data[1] - y_data[0]))

    def set_angle(self, theta):
        x_data = self.line.get_xdata()
        y_data = self.line.get_ydata()
        x_end = x_data[0] + np.cos(theta) * self.r
        y_end = y_data[0] + np.sin(theta) * self.r
        self.line.set_xdata((x_data[0], x_end))
        self.line.set_ydata((y_data[0], y_end))

    def rotate(self, i):
        self.set_angle(self.theta + self.w * np.radians(i))

    def get_centre(self):
        return self.circle.get_center()

    def get_end(self):
        return [self.line.get_xdata()[1], self.line.get_ydata()[1]]


def coeff_bin_to_epicycles(coeff_bins):
    epicycles = [
        epicycle(abs(coeff), freq, theta=np.angle(coeff))
        for coeff, freq in coeff_bins
    ]
    epicycles_sorted = sorted(epicycles, key=lambda x: x.r, reverse=True)
    return epicycles_sorted


image_new = cv2.imread("picture.png", 0)
xdata, ydata = greedy_tour(image_new, start=[82, 91])
xdata_downsampled, ydata_downsampled = downsample(xdata, ydata, target_num=300)
invert_ydata_downsampled = [-y for y in ydata_downsampled]
points = np.array(list(zip(xdata_downsampled, invert_ydata_downsampled)))
co_bi, norm_x, norm_y = points_to_coeff_bin(points)
epi_list = coeff_bin_to_epicycles(co_bi)
"""
test_epi = epicycle(3,2,theta = np.pi*0.5)
second_epi = epicycle(2,7)
line = plt.Line2D([],[])
epi_list = []
epi_list.append(test_epi)
epi_list.append(second_epi)
"""
line = plt.Line2D([], [], color="r")
artists = []

artists.append(line)
for epi in epi_list:
    artists = artists + epi.get_artists()


def init():
    ax.add_epicycle(epi_list[0])
    # ax.scatter(norm_x,norm_y,color='r')

    for i in range(len(epi_list) - 1):
        next_epi = epi_list[i + 1]
        current_epi = epi_list[i]
        next_epi.set_centre(current_epi.get_end())
        ax.add_epicycle(next_epi)

    ax.add_line(line)

    return artists


def animate(i):

    # test_epi.set_centre((np.sin(np.radians(i)),np.cos(np.radians(i))))
    epi_list[0].rotate(i)

    for j in range(len(epi_list) - 1):
        epi_list[j + 1].set_centre(epi_list[j].get_end())
        epi_list[j + 1].rotate(i)

    x_new, y_new = epi_list[-1].get_end()
    xdata = line.get_xdata() + [x_new]
    ydata = line.get_ydata() + [y_new]
    line.set_data(xdata, ydata)

    global timed
    if not (timed):
        print("completed in ", time.time() - t0)
        timed = True
    return artists


anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=360, interval=20, blit=True
)


plt.show()
