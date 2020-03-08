from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import cv2
import numpy as np

"""
threshold = 128

image_new = cv2.imread('picture.png',0)
image_new = image_new > threshold

# Invert the horse image
image = invert(image_new)

# perform skeletonization
skeleton = skeletonize(image)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
#plt.show()
"""

adj_directions = [
    [-1, -1],
    [0, -1],
    [1, -1],
    [-1, 0],
    [1, 0],
    [1, -1],
    [1, 0],
    [1, 1],
]

adj_directions_2 = []
for i in range(-2, 3):
    for j in range(-2, 3):
        adj_directions_2.append([i, j])
adj_directions_2 = [
    x for x in adj_directions_2 if max([abs(y) for y in x]) == 2
]


def part_of_skel(skel, point):
    try:
        return skel[point[1], point[0]] == True
    except IndexError:
        return False


def find_adj_points(skel, current_point):
    adj_points = []
    for direction in adj_directions:
        next_point = current_point + np.array(direction)
        if part_of_skel(skel, next_point):
            adj_points.append(next_point)

    if len(adj_points) == 0:
        for direction in adj_directions_2:
            next_point = current_point + np.array(direction)
            if part_of_skel(skel, next_point):
                adj_points.append(next_point)
    return adj_points


class FalsePixelError(Exception):
    pass


def shortest_tour(
    skel, start=np.array([0, 0]), unvisited_branches=[], visited_points=[]
):
    # print(start)

    if not part_of_skel(skel, start):
        raise FalsePixelError

    skel[start[1], start[0]] = False
    visited_points.append(start)

    unvisited_branches = find_adj_points(skel, start) + unvisited_branches
    if len(unvisited_branches) == 0:
        return skel, None, unvisited_branches, visited_points, True
    else:
        next_point = np.array(unvisited_branches[0])
        unvisited_branches = [
            x for x in unvisited_branches if not np.array_equal(x, next_point)
        ]
        return skel, next_point, unvisited_branches, visited_points, False


# print(len(shortest_tour(skeleton,start=np.array([91,82]))))


def greedy_tour(image, start=np.array([0, 0])):
    threshold = 128
    image_new = image > threshold

    # Invert the horse image
    image = invert(image_new)

    # perform skeletonization
    skeleton = skeletonize(image)

    found = False
    i = start[1]
    while not found:
        try:
            finished = False
            (
                skeleton,
                next_point,
                unvisited_branches,
                visited_points,
                finished,
            ) = shortest_tour(skeleton, start=np.array([start[0], i]))

            while not finished:
                (
                    skeleton,
                    next_point,
                    unvisited_branches,
                    visited_points,
                    finished,
                ) = shortest_tour(
                    skeleton,
                    start=next_point,
                    unvisited_branches=unvisited_branches,
                    visited_points=visited_points,
                )

            xdata = [x for x, y in visited_points]
            ydata = [y for x, y in visited_points]

            """
			ax[1].imshow(skeleton, cmap=plt.cm.gray)
			ax[1].axis('off')
			ax[1].set_title('skeleton', fontsize=20)

			fig.tight_layout()
			plt.show()
			"""

            found = True
        except FalsePixelError:
            i += 1

    return xdata, ydata


def downsample(xdata, ydata, target_num=100):
    sample_size = len(xdata)
    fraction = sample_size / target_num
    downsample_rate = int(fraction)
    if fraction < 1:
        print("no need to downsample, using m = {}".format(sample_size))
        return xdata, ydata
    print(
        "downsampling by factor {}, using m = {}".format(
            fraction, int(sample_size / int(fraction))
        )
    )
    xdata_downsampled, ydata_downsampled = [], []
    for i in range(sample_size):
        if i % downsample_rate == 0:
            xdata_downsampled.append(xdata[i])
            ydata_downsampled.append(ydata[i])
    return xdata_downsampled, ydata_downsampled


"""
image_new = cv2.imread('picture.png',0)

xdata,ydata = greedy_tour(image_new,start=[82,91])
xdata_downsampled,ydata_downsampled = downsample(xdata,ydata)
ax[1].plot(xdata_downsampled,ydata_downsampled, linewidth=0.7)
ax[1].set_title('greedy tour', fontsize=20)
fig.tight_layout()
plt.show()
"""
