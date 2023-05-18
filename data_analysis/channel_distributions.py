import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/CellTracking")

from CellTracking import get_file_embcode, read_img_with_resolution
import os

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/Casp3/movies/'
path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/Casp3/CellTrackObjects/'

files = os.listdir(path_data)

file, embcode = get_file_embcode(path_data, 0)

IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)
IMGS_flaten1 = IMGS.flatten()
IMGS_flaten1 = IMGS_flaten1[IMGS_flaten1]

IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=0)
IMGS_flaten0 = IMGS.flatten()
IMGS_flaten0 = IMGS_flaten0[IMGS_flaten0]


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(IMGS_flaten0, bins=100)
ax.hist(IMGS_flaten1, bins=100)

ax.set_xlim([2,100])
plt.show()

import numpy as np
np.mean(IMGS_flaten1)