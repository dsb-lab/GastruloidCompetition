from qlivecell import tif_reader_5D, cellSegTrack, extract_fluoro, get_file_names, get_file_name, get_intenity_profile, construct_RGB, EmbryoSegmentation
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

weight_th=2
path_data = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/merge/2024_07_22_DMSOmerged_Casp3/"

channel_names = ["A12", "Casp3", "F3", "DAPI"]

files = get_file_names(path_data)
files = [file for file in files if ".tif" in file]

files_WT_48_6 = [file for file in files if all(map(file.__contains__, ["48+06", "F3+WT"]))]
files_KO_48_6 = [file for file in files if all(map(file.__contains__, ["48+06", "F3+KO"]))]
files_WT_48_24 = [file for file in files if all(map(file.__contains__, ["48+24", "F3+WT"]))]
files_KO_48_24 = [file for file in files if all(map(file.__contains__, ["48+24", "F3+KO"]))]

files_WT_72_6 = [file for file in files if all(map(file.__contains__, ["72+06", "F3+WT"]))]
files_KO_72_6 = [file for file in files if all(map(file.__contains__, ["72+06", "F3+KO"]))]
files_WT_72_24 = [file for file in files if all(map(file.__contains__, ["72+24", "F3+WT"]))]
files_KO_72_24 = [file for file in files if all(map(file.__contains__, ["72+24", "F3+KO"]))]

files_current = files_WT_72_6
file = files_current[0]

path_data_file = path_data+file
            
ch_F3 = channel_names.index("F3")
ch_A12 = channel_names.index("A12")
ch_Casp3 = channel_names.index("Casp3")

hyperstack, metadata = tif_reader_5D(path_data_file)

stack_f3_mean = np.mean(hyperstack[0,:,ch_F3], axis=0)
stack_a12_mean = np.mean(hyperstack[0,:,ch_A12], axis=0)
stack_casp3_mean = np.mean(hyperstack[0,:,ch_Casp3], axis=0)

hstack = hyperstack[:,:,ch_F3].astype("float32")+hyperstack[:,:,ch_A12].astype("float32")
z_plot = np.rint(hyperstack[0,:,ch_F3].shape[0]/2).astype("int64")
z_plot = 24 
ES = EmbryoSegmentation(
        hstack,
        ksize=5,
        ksigma=20,
        binths=8,
        apply_biths_to_zrange_only=False,
        checkerboard_size=10,
        num_inter=100,
        smoothing=20,
        trange=None,
        zrange=range(z_plot, z_plot+1),
        mp_threads=None,
    )

ES(hstack)
ES.plot_segmentation(0, z_plot, plot_background=True)
ES.Backmask[0][0].shape

