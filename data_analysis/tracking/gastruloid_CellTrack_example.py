### LOAD PACKAGE ###
from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, load_CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/'
path_save='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/'

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)
### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data)

file, embcode = get_file_embcode(path_data, 2, returnfiles=False)


### LOAD HYPERSTACKS ###
channel = 0
IMGS_A12, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
### LOAD HYPERSTACKS ###
channel = 1
IMGS_F3, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
### LOAD HYPERSTACKS ###
channel = 2
IMGS_Casp3, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
### LOAD HYPERSTACKS ###
channel = 4
IMGS_DAPI, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)


### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    'blur': [5,1], 
    # 'scale': 3
}
          
concatenation3D_args = {
    'distance_th_z': 3.0, 
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
    'min_cell_planes': 3,
}

tracking_args = {
    'time_step': 5, # minutes
    'method': 'greedy', 
    'z_th':5, 
    'dist_th' : 10.0,
}

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    'plot_stack_dims': (512, 512), 
    'plot_centers':[True, True]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'lasso',
}


### CREATE CELLTRACKING CLASS ###
CT = CellTracking(
    IMGS_Casp3, 
    path_save, 
    embcode+"ch_%d" %(channel+1), 
    xyresolution=xyres, 
    zresolution=zres,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    tracking_args = tracking_args, 
    error_correction_args=error_correction_args,    
    plot_args = plot_args,
)


### RUN SEGMENTATION AND TRACKING ###
CT.run()

# from embdevtools.celltrack.core.tools.cell_tools import remove_small_cells, remove_small_planes_at_boders

# remove_small_cells(CT.jitcells, 250, CT._del_cell, CT.update_labels)
# remove_small_planes_at_boders(CT.jitcells, 200, CT._del_cell, CT.update_labels, CT._stacks)



# ### PLOTTING ###
import numpy as np
_IMGS_A12 = np.asarray([[255*(IMG/IMG.max()) for IMG in IMGS_A12[0]]]).astype('uint8')
_IMGS_F3 = np.asarray([[255*(IMG/IMG.max()) for IMG in IMGS_F3[0]]]).astype('uint8')
_IMGS_Casp3 = np.asarray([[255*(IMG/IMG.max()) for IMG in IMGS_Casp3[0]]]).astype('uint8')

IMGS_plot = construct_RGB(R=_IMGS_A12, G=_IMGS_F3, B=_IMGS_Casp3)

CT.plot_tracking(plot_args, stacks_for_plotting=IMGS_plot)


CASP3 = []
A12 = []
F3 = []
na12 = 0
nf3 = 0
labs = []
for cell in CT.jitcells:
    casp3 = []
    a12 = []
    f3 = []
    for zid, z in enumerate(cell.zs[0]):
        mask = cell.masks[0][zid]
        casp3.append(np.mean(_IMGS_Casp3[0][z][mask[:,1], mask[:,0]]))
        a12.append(np.mean(_IMGS_A12[0][z][mask[:,1], mask[:,0]]))
        f3.append(np.mean(_IMGS_F3[0][z][mask[:,1], mask[:,0]]))

    # idx = np.argmax(casp3)
    zz = np.int64(cell.centers[0][0])
    idx = cell.zs[0].index(zz)
    if f3[idx] > a12[idx]:
        nf3 +=1
    else:
        na12 +=1
        labs.append(cell.label)
    CASP3.append(casp3[idx])
    A12.append(a12[idx])
    F3.append(f3[idx])

    # if np.mean(f3) > np.mean(a12):
    #     nf3 +=1
    # else:
    #     na12 +=1
    #     labs.append(cell.label)
    # CASP3.append(np.mean(casp3))
    # A12.append(np.mean(a12))
    # F3.append(np.mean(f3))

import matplotlib.pyplot as plt
plt.scatter(A12, F3)
plt.show()

plt.hist(F3, bins=50)
plt.hist(A12, bins=50)
plt.show()

segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    'blur': None, 
}
          
### CREATE CELLTRACKING CLASS ###
CT_A12 = CellTracking(
    IMGS_A12, 
    path_save, 
    embcode+"ch_%d" %(channel+1), 
    xyresolution=xyres, 
    zresolution=zres,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    tracking_args = tracking_args, 
    error_correction_args=error_correction_args,    
    plot_args = plot_args,
)


### RUN SEGMENTATION AND TRACKING ###
CT_A12.run()
CT_A12.plot_tracking(plot_args, stacks_for_plotting=IMGS_plot)

### CREATE CELLTRACKING CLASS ###
CT_F3 = CellTracking(
    IMGS_F3, 
    path_save, 
    embcode+"ch_%d" %(channel+1), 
    xyresolution=xyres, 
    zresolution=zres,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    tracking_args = tracking_args, 
    error_correction_args=error_correction_args,    
    plot_args = plot_args,
)


### RUN SEGMENTATION AND TRACKING ###
CT_F3.run()
CT_F3.plot_tracking(plot_args, stacks_for_plotting=IMGS_plot)

print("A12", na12 / len(CT_A12.jitcells))
print("F3", nf3 / len(CT_F3.jitcells))