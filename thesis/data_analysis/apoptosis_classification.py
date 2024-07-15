### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, get_file_names, correct_path, save_4Dstack, arboretum_napari
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
})
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath'
mpl.rc('font', size=14) 
mpl.rc('axes', labelsize=14) 
mpl.rc('xtick', labelsize=14) 
mpl.rc('ytick', labelsize=14) 
mpl.rc('legend', fontsize=14) 

from embdevtools.celltrack.core.tools.tools import increase_point_resolution
from embdevtools import construct_RGB

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/thesis/figures/figures/apoptosis_classification/"

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/48hr/WT/'
path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/48hr/WT/'

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

# for f, file in enumerate(files):
file = files[1]
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
CT_F3.load()

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
CT_A12.load()

ch_Casp3 = channel_names.index("Casp3")
chans = [ch_Casp3]
for _ch in range(len(channel_names)):
    if _ch not in chans:
        chans.append(_ch)

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    # 'plot_stack_dims': (256, 256), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[ch_Casp3],
    'min_outline_length':75,
}

batch_args = {
    'name_format':"ch"+str(ch_Casp3)+"_{}_early",
    'extension':".tif",
}
CT_Casp3_early = CellTracking(
    path_data,
    path_save,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    error_correction_args=error_correction_args,
    plot_args=plot_args,
    batch_args=batch_args,
    channels=chans
)
CT_Casp3_early.load()

batch_args = {
    'name_format':"ch"+str(ch_Casp3)+"_{}_mid",
    'extension':".tif",
}
CT_Casp3_mid = CellTracking(
    path_data,
    path_save,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    error_correction_args=error_correction_args,
    plot_args=plot_args,
    batch_args=batch_args,
    channels=chans
)
CT_Casp3_mid.load()

batch_args = {
    'name_format':"ch"+str(ch_Casp3)+"_{}_late",
    'extension':".tif",
}
CT_Casp3_late = CellTracking(
    path_data,
    path_save,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    error_correction_args=error_correction_args,
    plot_args=plot_args,
    batch_args=batch_args,
    channels=chans
)
CT_Casp3_late.load()

from embdevtools import extract_fluoro
results_F3 = extract_fluoro(CT_F3)
ch = channel_names.index("F3")
f3 = results_F3["channel_{}".format(ch)]

results_A12 = extract_fluoro(CT_A12)
ch = channel_names.index("A12")
a12 = results_A12["channel_{}".format(ch)]


centers_early = []
outlines_early = []
masks_early = []
Casp3_early = []
F3_early = []
A12_early = []

for cell in CT_Casp3_early.jitcells:
    center = cell.centers[0]
    centers_early.append(center)
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    masks_early.append(mask)
    outline = cell.outlines[0][zid]
    outlines_early.append(outline)
    Casp3_early.append(np.mean(CT_Casp3_early.hyperstack[0,z,ch_Casp3,:,:][mask[:,1], mask[:,0]]))
    F3_early.append(np.mean(CT_Casp3_early.hyperstack[0,z,ch_F3,:,:][mask[:,1], mask[:,0]]))
    A12_early.append(np.mean(CT_Casp3_early.hyperstack[0,z,ch_A12,:,:][mask[:,1], mask[:,0]]) - (np.mean(a12) - np.mean(f3)))

centers_mid = []
outlines_mid = []
masks_mid = []
Casp3_mid = []
F3_mid = []
A12_mid = []
for cell in CT_Casp3_mid.jitcells:
    center = cell.centers[0]
    centers_mid.append(center)
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    masks_mid.append(mask)
    outline = cell.outlines[0][zid]
    outlines_mid.append(outline)
    Casp3_mid.append(np.mean(CT_Casp3_mid.hyperstack[0,z,ch_Casp3,:,:][mask[:,1], mask[:,0]]))
    F3_mid.append(np.mean(CT_Casp3_mid.hyperstack[0,z,ch_F3,:,:][mask[:,1], mask[:,0]]))
    A12_mid.append(np.mean(CT_Casp3_mid.hyperstack[0,z,ch_A12,:,:][mask[:,1], mask[:,0]]) - (np.mean(a12) - np.mean(f3)))
    
centers_late = []
outlines_late = []
masks_late = []
Casp3_late = []
F3_late = []
A12_late = []
for cell in CT_Casp3_late.jitcells:
    center = cell.centers[0]
    centers_late.append(center)
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    masks_late.append(mask)
    outline = cell.outlines[0][zid]
    outlines_late.append(outline)
    Casp3_late.append(np.mean(CT_Casp3_late.hyperstack[0,z,ch_Casp3,:,:][mask[:,1], mask[:,0]]))
    F3_late.append(np.mean(CT_Casp3_late.hyperstack[0,z,ch_F3,:,:][mask[:,1], mask[:,0]]))
    A12_late.append(np.mean(CT_Casp3_late.hyperstack[0,z,ch_A12,:,:][mask[:,1], mask[:,0]]) - (np.mean(a12) - np.mean(f3)))
    


def channelnorm(im, channel, vmin, vmax):
    c = (im[:,:,channel]-vmin) / (vmax-vmin)
    c[c<0.] = 0.0
    c[c>1.] = 1.0
    im[:,:,channel] = c
    return im


x = ['H2B-mCherry', 'H2B-emiRFP']
r = 30


stack_early = construct_RGB(R=CT_Casp3_early.hyperstack[0,:,1,:,:], G=CT_Casp3_early.hyperstack[0,:,0,:,:], B=CT_Casp3_early.hyperstack[0,:,ch_Casp3,:,:])
stack_early = stack_early/255.0
stack_early[stack_early > 1.0] = 1.0

for z in range(stack_early.shape[0]):
    stack_early[z,:,:] = channelnorm(stack_early[z], 0, vmin=0.0, vmax=1.0)
    stack_early[z,:,:] = channelnorm(stack_early[z], 1, vmin=0.0, vmax=1.0)
    stack_early[z,:,:] = channelnorm(stack_early[z], 2, vmin=0.0, vmax=0.4)

stack_mid = construct_RGB(R=CT_Casp3_mid.hyperstack[0,:,1,:,:], G=CT_Casp3_mid.hyperstack[0,:,0,:,:], B=CT_Casp3_mid.hyperstack[0,:,ch_Casp3,:,:])
stack_mid = stack_mid/255.0
stack_mid[stack_mid > 1.0] = 1.0

for z in range(stack_mid.shape[0]):
    stack_mid[z,:,:] = channelnorm(stack_mid[z], 0, vmin=0.0, vmax=1.0)
    stack_mid[z,:,:] = channelnorm(stack_mid[z], 1, vmin=0.0, vmax=1.0)
    stack_mid[z,:,:] = channelnorm(stack_mid[z], 2, vmin=0.0, vmax=0.4)
    
stack_late = construct_RGB(R=CT_Casp3_late.hyperstack[0,:,1,:,:], G=CT_Casp3_late.hyperstack[0,:,0,:,:], B=CT_Casp3_late.hyperstack[0,:,ch_Casp3,:,:])
stack_late = stack_late/255.0
stack_late[stack_late > 1.0] = 1.0

for z in range(stack_late.shape[0]):
    stack_late[z,:,:] = channelnorm(stack_late[z], 0, vmin=0.0, vmax=1.0)
    stack_late[z,:,:] = channelnorm(stack_late[z], 1, vmin=0.0, vmax=1.0)
    stack_late[z,:,:] = channelnorm(stack_late[z], 2, vmin=0.0, vmax=0.4)


import random 
import matplotlib as mpl
plt.rcParams.update({
    "text.usetex": True,
})
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath'
mpl.rc('font', size=14) 
mpl.rc('axes', labelsize=14) 
mpl.rc('xtick', labelsize=14) 
mpl.rc('ytick', labelsize=14) 
mpl.rc('legend', fontsize=14) 

fig ,ax = plt.subplots(1,3, figsize=(12,4))
c1=4
# c1 = random.randint(0, len(centers_early)-1)

center = np.rint(centers_early[c1]).astype("int32")
outline = outlines_early[c1]
mask = masks_early[c1]
z = center[0]

ax[0].set_title("Early")
ax[0].imshow(stack_early[z])
ax[0].plot(outline[:,0], outline[:,1], c="C5",lw=3)
x = [outline[0,0], outline[-1,0]]
y = [outline[0,1], outline[-1,1]]
ax[0].plot(x, y, c="C5",lw=3)
ax[0].set_xlim(center[1]-r,center[1]+r)
ax[0].set_ylim(center[2]-r,center[2]+r)
ax[0].grid(False)
ax[0].set_xticks([])
ax[0].set_yticks([])

c2=7
# c2 = random.randint(0, len(centers_mid)-1)
center = np.rint(centers_mid[c2]).astype("int32")
outline = outlines_mid[c2]
mask = masks_mid[c2]
z = center[0]

ax[1].set_title("Mid")
ax[1].imshow(stack_mid[z])
ax[1].plot(outline[:,0], outline[:,1], c="C5",lw=3)
x = [outline[0,0], outline[-1,0]]
y = [outline[0,1], outline[-1,1]]
ax[1].plot(x, y, c="C5",lw=3)
ax[1].set_xlim(center[1]-r,center[1]+r)
ax[1].set_ylim(center[2]-r,center[2]+r)
ax[1].grid(False)
ax[1].set_xticks([])
ax[1].set_yticks([])

c3=212
# c3 = random.randint(0, len(centers_late)-1)
center = np.rint(centers_late[c3]).astype("int32")
outline = outlines_late[c3]
mask = masks_late[c3]
z = center[0]

ax[2].set_title("Late")
ax[2].imshow(stack_late[z])
ax[2].plot(outline[:,0], outline[:,1], c="C5",lw=3)
x = [outline[0,0], outline[-1,0]]
y = [outline[0,1], outline[-1,1]]
ax[2].plot(x, y, c="C5",lw=3)
ax[2].set_xlim(center[1]-r,center[1]+r)
ax[2].set_ylim(center[2]-r,center[2]+r)
ax[2].grid(False)
ax[2].set_xticks([])
ax[2].set_yticks([])
plt.tight_layout()
plt.savefig(path_figures+"apostages.svg")
plt.savefig(path_figures+"apostages.pdf")
plt.show()
