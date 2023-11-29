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
channel_A12 = 0
IMGS_A12, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel_A12)
### LOAD HYPERSTACKS ###
channel_F3 = 1
IMGS_F3, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel_F3)
### LOAD HYPERSTACKS ###
channel_Casp3 = 2
IMGS_Casp3, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel_Casp3)
### LOAD HYPERSTACKS ###
channel_DAPI = 4
IMGS_DAPI, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel_DAPI)


### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

### DEFINE ARGUMENTS ###
          
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

segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    'blur': None, 
}
          
### CREATE CELLTRACKING CLASS ###
CT_A12 = CellTracking(
    IMGS_A12, 
    path_save, 
    embcode+"ch_%d" %(channel_A12+1), 
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
CT_A12.plot_tracking(plot_args)

### CREATE CELLTRACKING CLASS ###
CT_F3 = CellTracking(
    IMGS_F3, 
    path_save, 
    embcode+"ch_%d" %(channel_F3+1), 
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
CT_F3.plot_tracking(plot_args)

masks = []
for cell in CT_A12.jitcells:
    z = int(cell.centers[0][0])
    zid = cell.zs[0].index(z)
    masks.append(cell.masks[0][zid])

areasA12 = [len(mask) for mask in masks]
diametersA12 = [2*np.sqrt(area/np.pi) for area in areasA12]
diametersA12 = np.array(diametersA12)*xyres

masks = []
for cell in CT_F3.jitcells:
    z = int(cell.centers[0][0])
    zid = cell.zs[0].index(z)
    masks.append(cell.masks[0][zid])
areasF3 = [len(mask) for mask in masks]
diametersF3 = [2*np.sqrt(area/np.pi) for area in areasF3]
diametersF3 = np.array(diametersF3)*xyres

from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt

# data = np.concatenate((diametersA12, diametersF3))
data =  diametersF3
x = np.arange(0, step=0.1, stop=np.max(data))
bw = 1.0
modelo_kde = KernelDensity(kernel='linear', bandwidth=bw)
modelo_kde.fit(X=data.reshape(-1, 1))
densidad_pred = np.exp(modelo_kde.score_samples(x.reshape((-1,1))))
local_minima = argrelextrema(densidad_pred, np.less)[0]
x_th = np.ones(len(x))*x[local_minima[0]]
y_th = np.linspace(0, np.max(densidad_pred), num=len(x))
plt.plot(x_th, y_th, c='k', ls='--')
plt.hist(data, bins=50, density=True, label="hist")
plt.plot(x, densidad_pred, lw=5, label="kde")
plt.legend()
plt.xlabel("diam (µm)")
plt.yticks([])
plt.title("debris threshold = {:0.2f}µm".format(x[local_minima[0]]))
# plt.savefig("/home/pablo/test.png")
plt.show()

from embdevtools.celltrack.core.tools.cell_tools import remove_small_cells, remove_small_planes_at_boders

diam_th = 5.6 / xyres
area_th = np.pi * (diam_th/2)**2 
remove_small_cells(CT_A12.jitcells, area_th, CT_A12._del_cell, CT_A12.update_labels)
remove_small_cells(CT_F3.jitcells, area_th, CT_F3._del_cell, CT_F3.update_labels)

# remove_small_planes_at_boders(CT.jitcells, 200, CT._del_cell, CT.update_labels, CT._stacks)

CT_A12.plot_tracking(plot_args)
CT_F3.plot_tracking(plot_args)

masks = []
for cell in CT_A12.jitcells:
    z = int(cell.centers[0][0])
    zid = cell.zs[0].index(z)
    masks.append(cell.masks[0][zid])

areasA12 = [len(mask) for mask in masks]
diametersA12 = [2*np.sqrt(area/np.pi) for area in areasA12]
diametersA12 = np.array(diametersA12)*xyres

masks = []
for cell in CT_F3.jitcells:
    z = int(cell.centers[0][0])
    zid = cell.zs[0].index(z)
    masks.append(cell.masks[0][zid])
areasF3 = [len(mask) for mask in masks]
diametersF3 = [2*np.sqrt(area/np.pi) for area in areasF3]
diametersF3 = np.array(diametersF3)*xyres

from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt

data = np.concatenate((diametersA12, diametersF3))
x = np.arange(0, step=0.1, stop=np.max(data))
bw = 1.0
modelo_kde = KernelDensity(kernel='linear', bandwidth=bw)
modelo_kde.fit(X=data.reshape(-1, 1))
densidad_pred = np.exp(modelo_kde.score_samples(x.reshape((-1,1))))
local_minima = argrelextrema(densidad_pred, np.less)[0]
x_th = np.ones(len(x))*x[local_minima[0]]
y_th = np.linspace(0, np.max(densidad_pred), num=len(x))
plt.plot(x_th, y_th, c='k', ls='--')
plt.hist(data, bins=50, density=True, label="hist")
plt.plot(x, densidad_pred, lw=5, label="kde")
plt.legend()
plt.xlabel("diam (µm)")
plt.yticks([])
plt.title("debris threshold = {:0.2f}µm".format(x[local_minima[0]]))
# plt.savefig("/home/pablo/test.png")
plt.show()