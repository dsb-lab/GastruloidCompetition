import sys
sys.path.insert(0, "/home/pablo/Desktop/PhD/projects/EmbryoSegmentation")

from embryosegmentation import EmbryoSegmentation, get_file_embcode, load_ES, save_ES, read_img_with_resolution
import os

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/movies_registered/'
path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/CellTrackObjects/'

file, embcode = get_file_embcode(path_data, 0)

IMGS_ch1, xyres, zres = read_img_with_resolution(path_data+file, channel=1)
IMGS_ch2, xyres, zres = read_img_with_resolution(path_data+file, channel=0)

IMGS = IMGS_ch1 + IMGS_ch2
# EmbSeg = EmbryoSegmentation(IMGS, ksize=5, ksigma=3, binths=[20,5], checkerboard_size=6, num_inter=100, smoothing=5, trange=None, zrange=None, mp_threads=15, apply_biths_to_zrange_only=False)
# EmbSeg(IMGS)

# save_ES(EmbSeg, path_save, embcode)

EmbSeg = load_ES(path_save, embcode)

EmbSeg.plot_segmentation([0,9],[33,33])

