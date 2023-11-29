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
CT_Casp3 = CellTracking(
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
CT_Casp3.run()
CT_Casp3.plot_tracking(plot_args, stacks_for_plotting=IMGS_Casp3)

# ### PLOTTING ###
import numpy as np
_IMGS_A12 = np.asarray([[255*(IMG/IMG.max()) for IMG in IMGS_A12[0]]]).astype('uint8')
_IMGS_F3 = np.asarray([[255*(IMG/IMG.max()) for IMG in IMGS_F3[0]]]).astype('uint8')
_IMGS_Casp3 = np.asarray([[255*(IMG/IMG.max()) for IMG in IMGS_Casp3[0]]]).astype('uint8')

IMGS_plot = construct_RGB(R=_IMGS_A12, G=_IMGS_F3, B=_IMGS_Casp3)


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

from embdevtools.celltrack.core.tools.cell_tools import remove_small_cells

diam_th = 5.6 / xyres
area_th = np.pi * (diam_th/2)**2 
remove_small_cells(CT_A12.jitcells, area_th, CT_A12._del_cell, CT_A12.update_labels)
remove_small_cells(CT_F3.jitcells, area_th, CT_F3._del_cell, CT_F3.update_labels)


from libpysal import weights, examples
from libpysal.cg import voronoi_frames

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# In order for networkx to plot the nodes of our graph correctly, we
# need to construct the array of coordinates for each point in our dataset.
# To get this as a numpy array, we extract the x and y coordinates from the
# geometry column.

cells = [cell for sublist in [CT_A12.jitcells, CT_F3.jitcells, CT_Casp3.jitcells] for cell in sublist]
fates = [s for s, sublist in enumerate([CT_A12.jitcells, CT_F3.jitcells, CT_Casp3.jitcells]) for cell in sublist]
centers = []
labs = []

for c, cell in enumerate(cells):
    centers.append(cell.centers[0]*[zres, xyres, xyres])
    if c >= len(CT_A12.jitcells):
        labs.append(cell.label + CT_A12.max_label + 1)
    else:
        labs.append(cell.label)


centers = np.array(centers)
labs = np.array(labs)
from scipy.spatial import Delaunay

tri = Delaunay(centers)

def find_neighbors(pindex, triang):
    neighbors = list()
    for simplex in triang.simplices:
        if pindex in simplex:
            neighbors.extend([simplex[i] for i in range(len(simplex)) if simplex[i] != pindex])
            '''
            this is a one liner for if a simplex contains the point we`re interested in,
            extend the neighbors list by appending all the *other* point indices in the simplex
            '''
    #now we just have to strip out all the dulicate indices and return the neighbors list:
    return list(set(neighbors))

neighs = []
for p, point in enumerate(centers):
    neighs_p = find_neighbors(p, tri)
    neighs.append(neighs_p)

true_neighs = []
dist_th = 30 #microns
dist_th_near = 5
for p, neigh_p in enumerate(neighs):
    true_neigh_p = []
    for neigh in neigh_p:
        dist = np.linalg.norm(centers[p]-centers[neigh])
        if dist < dist_th:
            if dist > dist_th_near:
                true_neigh_p.append(neigh)
    true_neighs.append(true_neigh_p)

neighs_labs = []
neighs_fates = []
for p, neigh_p in enumerate(true_neighs):
    lbs = []
    fts = []
    for neigh in neigh_p:
        lbs.append(labs[neigh])
        fts.append(fates[neigh])
    neighs_labs.append(lbs)
    neighs_fates.append(fts)

