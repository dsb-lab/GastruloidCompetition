### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, get_file_names, correct_path, save_4Dstack
import numpy as np
from scipy import stats

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')


### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/96hr/WT/'
path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/96hr/test/'

try: 
    files = get_file_names(path_save_dir)
except: 
    import os
    os.mkdir(path_save_dir)

### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data_dir)

channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
if "96hr" in path_data_dir:
    channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

zplanes = []
for f, file in enumerate(files):

    path_data = path_data_dir+file
    file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
    path_save = correct_path(path_save_dir+embcode)
    try: 
        files = get_file_names(path_save)
    except: 
        import os
        os.mkdir(path_save)

    ### DEFINE ARGUMENTS ###
    segmentation_args={
        'method': 'stardist2D', 
        'model': model, 
        'blur': None, 
        # 'n_tiles': (2,2),
    }

    concatenation3D_args = {
        'distance_th_z': 3.0, # microns
        'relative_overlap':False, 
        'use_full_matrix_to_compute_overlap':True, 
        'z_neighborhood':2, 
        'overlap_gradient_th':0.3, 
        'min_cell_planes': 2
    }


    error_correction_args = {
        'backup_steps': 10,
        'line_builder_mode': 'points',
    }

    ch_F3 = channel_names.index("F3")

    batch_args = {
        'name_format':"ch"+str(ch_F3)+"_{}",
        'extension':".tif",
    }
    plot_args = {
        'plot_layout': (1,1),
        'plot_overlap': 1,
        'masks_cmap': 'tab10',
        # 'plot_stack_dims': (256, 256), 
        'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
        'channels':[ch_F3],
        'min_outline_length':75,
    }

    chans = [ch_F3]
    for _ch in range(len(channel_names)):
        if _ch not in chans:
            chans.append(_ch)
    CT_F3 = CellTracking(
        path_data,
        path_save,
        segmentation_args=segmentation_args,
        concatenation3D_args=concatenation3D_args,
        error_correction_args=error_correction_args,
        plot_args=plot_args,
        batch_args=batch_args,
        channels=chans
    )


    CT_F3.run()
    # CT_F3.plot_tracking()

    len(CT_F3.blocked_cells)
    ch_A12 = channel_names.index("A12")
    batch_args = {
        'name_format':"ch"+str(ch_A12)+"_{}",
        'extension':".tif",
    }
    plot_args = {
        'plot_layout': (1,1),
        'plot_overlap': 1,
        'masks_cmap': 'tab10',
        # 'plot_stack_dims': (256, 256), 
        'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
        'channels':[ch_A12],
        'min_outline_length':75,
    }

    chans = [ch_A12]
    for _ch in range(len(channel_names)):
        if _ch not in chans:
            chans.append(_ch)

    CT_A12 = CellTracking(
        path_data,
        path_save,
        segmentation_args=segmentation_args,
        concatenation3D_args=concatenation3D_args,
        error_correction_args=error_correction_args,
        plot_args=plot_args,
        batch_args=batch_args,
        channels=chans
    )

    CT_A12.run()
    # CT_A12.plot_tracking()


    for cell in CT_F3.jitcells:
        zplanes.append(len(cell.zs[0]))
        
    for cell in CT_A12.jitcells:
        zplanes.append(len(cell.zs[0]))
        

import matplotlib.pyplot as plt
binss, be, _ = plt.hist(zplanes, bins=np.max(zplanes))
plt.xticks(range(np.max(zplanes)))
plt.show()

from skimage.filters import threshold_triangle

threshold_triangle(zplanes, nbins=np.max(zplanes))

zplanes = np.array(zplanes)
len(np.where(zplanes >9)[0]) / len(zplanes)


a = [2,3,4,5,6,7]
mid_plane = np.rint(len(a)/2).astype("int64")
ids1 = [i for i in range(mid_plane)]
ids2 = [i for i in range(mid_plane, len(a))]