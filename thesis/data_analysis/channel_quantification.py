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

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/thesis/figures/figures/quantification/"

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

batch_args = {
    'name_format':"{}",
    'name_format_save':"all_{}",
    'extension':".tif",
}
CT_all1 = CellTracking(
    path_data,
    path_save,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    error_correction_args=error_correction_args,
    plot_args=plot_args,
    batch_args=batch_args,
    channels=chans
)
CT_all1.run()


CT_all1.load()
labs = [cell.label for cell in CT_all1.jitcells]
for lab in labs:
    CT_all1._del_cell(lab)

CT_all1.update_labels()

for cell in CT_F3.jitcells:
    CT_all1.append_cell_from_cell(cell)

for cell in CT_A12.jitcells:
    CT_all1.append_cell_from_cell(cell)

CT_all1.update_labels()

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    # 'plot_stack_dims': (256, 256), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[ch_A12, ch_F3, ch_A12],
    'min_outline_length':75,
}

CT_all1.plot_tracking(plot_args=plot_args)


def channelnorm(im, channel, vmin, vmax):
    c = (im[:,:,channel]-vmin) / (vmax-vmin)
    print(c.shape)
    c[c<0.] = 0.0
    c[c>1.] = 1.0
    im[:,:,channel] = c
    return im


F3 = []
A12 = []
for cell in CT_F3.jitcells:
    center = cell.centers[0]
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    outline = cell.outlines[0][zid]
    F3.append(np.mean(CT_all1.hyperstack[0,z,0,:,:][mask[:,1], mask[:,0]]))

for cell in CT_A12.jitcells:
    center = cell.centers[0]
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    outline = cell.outlines[0][zid]
    A12.append(np.mean(CT_all1.hyperstack[0,z,1,:,:][mask[:,1], mask[:,0]]))

F3_all = []
A12_all = []
colors=[]
centers=[]
masks = []
outlines = []
for cell in CT_F3.jitcells:
    center = cell.centers[0]
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    masks.append(mask)
    outline = cell.outlines[0][zid]
    outlines.append(outline)
    centers.append(center)

    F3_all.append(np.mean(CT_all1.hyperstack[0,z,0,:,:][mask[:,1], mask[:,0]]))
    A12_all.append(np.mean(CT_all1.hyperstack[0,z,1,:,:][mask[:,1], mask[:,0]]))
    colors.append([0.0,0.8,0.0, 0.3])
    
for cell in CT_A12.jitcells:
    center = cell.centers[0]
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    masks.append(mask)
    outline = cell.outlines[0][zid]
    outlines.append(outline)
    centers.append(center)

    F3_all.append(np.mean(CT_all1.hyperstack[0,z,0,:,:][mask[:,1], mask[:,0]]))
    A12_all.append(np.mean(CT_all1.hyperstack[0,z,1,:,:][mask[:,1], mask[:,0]]))
    colors.append([0.8,0.0,0.8, 0.3])

x = ['H2B-mCherry', 'H2B-emiRFP']
r = 20


stack = construct_RGB(R=CT_all1.hyperstack[0,:,1,:,:], G=CT_all1.hyperstack[0,:,0,:,:], B=CT_all1.hyperstack[0,:,1,:,:])
stack = stack/255.0

for z in range(stack.shape[0]):
    stack[z,:,:] = channelnorm(stack[z], 0, vmin=0.0, vmax=0.9)
    stack[z,:,:] = channelnorm(stack[z], 1, vmin=0.0, vmax=0.9)
    stack[z,:,:] = channelnorm(stack[z], 2, vmin=0.0, vmax=0.9)

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

fig ,ax = plt.subplots(2,4, figsize=(12,6))
c1=138
# c1 = random.randint(0, len(centers)-1)
c2=1098
# c2 = random.randint(0, len(centers)-1)
center = np.rint(centers[c1]).astype("int32")
outline = outlines[c1]
mask = masks[c1]
z = center[0]
np.mean(CT_all1.hyperstack[0,z,0,:,:][mask[:,0], mask[:,1]])

ax[0,0].set_title("H2B-mCherry")
ax[0,0].imshow(stack[z])
ax[0,0].plot(outline[:,0], outline[:,1], c="C5",lw=3)
x = [outline[0,0], outline[-1,0]]
y = [outline[0,1], outline[-1,1]]
ax[0,0].plot(x, y, c="C5",lw=3)
ax[0,0].set_xlim(center[1]-r,center[1]+r)
ax[0,0].set_ylim(center[2]-r,center[2]+r)
ax[0,0].grid(False)
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
ax[0,0].set_ylabel("Center plane", fontsize=16)

center = np.rint(centers[c2]).astype("int32")
outline = outlines[c2]
mask = masks[c2]
z = center[0]
np.mean(CT_all1.hyperstack[0,z,1,:,:][mask[:,1], mask[:,0]])

ax[0,1].set_title("H2B-emiRFP")
ax[0,1].imshow(stack[z])
ax[0,1].plot(outline[:,0], outline[:,1], c="C5",lw=3)
x = [outline[0,0], outline[-1,0]]
y = [outline[0,1], outline[-1,1]]
ax[0,1].plot(x, y, c="C5",lw=3)
ax[0,1].set_xlim(center[1]-r,center[1]+r)
ax[0,1].set_ylim(center[2]-r,center[2]+r)
ax[0,1].grid(False)
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])

ax[0,2].hist(F3_all, bins=30, color=[0.0, 0.8, 0.0], alpha=0.4)
ax[0,2].hist(A12_all, bins=30, color=[0.8, 0.0, 0.8], alpha=0.4)
ax[0,2].set_yticks([])
ax[0,2].spines[['right', 'top', "left"]].set_visible(False)

ax[0,3].set_ylabel("mean H2B-emiRFP [a.u.]")
ax[0,3].scatter(F3_all, A12_all,c=colors, edgecolors='none')
ax[0,3].set_xlim(-5, 160)
ax[0,3].set_ylim(-5, 160)
ax[0,3].set_xticks([0,50,100,150])
ax[0,3].set_yticks([0,50,100,150])
ax[0,3].spines[['right', 'top']].set_visible(False)

F3 = []
A12 = []
for cell in CT_F3.jitcells:
    center = cell.centers_all[0][0]
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    outline = cell.outlines[0][zid]
    F3.append(np.mean(CT_all1.hyperstack[0,z,0,:,:][mask[:,1], mask[:,0]]))

for cell in CT_A12.jitcells:
    center = cell.centers_all[0][0]
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    outline = cell.outlines[0][zid]
    A12.append(np.mean(CT_all1.hyperstack[0,z,1,:,:][mask[:,1], mask[:,0]]))

F3_all = []
A12_all = []
colors=[]
centers=[]
masks = []
outlines = []
for cell in CT_F3.jitcells:
    center = cell.centers_all[0][0]
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    masks.append(mask)
    outline = cell.outlines[0][zid]
    outlines.append(outline)
    centers.append(center)

    F3_all.append(np.mean(CT_all1.hyperstack[0,z,0,:,:][mask[:,1], mask[:,0]]))
    A12_all.append(np.mean(CT_all1.hyperstack[0,z,1,:,:][mask[:,1], mask[:,0]]))
    colors.append([0.0,0.8,0.0, 0.3])
    
for cell in CT_A12.jitcells:
    center = cell.centers_all[0][0]
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    masks.append(mask)
    outline = cell.outlines[0][zid]
    outlines.append(outline)
    centers.append(center)

    F3_all.append(np.mean(CT_all1.hyperstack[0,z,0,:,:][mask[:,1], mask[:,0]]))
    A12_all.append(np.mean(CT_all1.hyperstack[0,z,1,:,:][mask[:,1], mask[:,0]]))
    colors.append([0.8,0.0,0.8, 0.3])


# import random 
# c1= random.randint(0, len(centers)-1)
# c1=281
# c2= random.randint(0, len(centers)-1)
# c2=980
center = np.rint(centers[c1]).astype("int32")
outline = outlines[c1]
mask = masks[c1]
z = center[0]

ax[1,0].imshow(stack[z])
ax[1,0].plot(outline[:,0], outline[:,1], c="C5",lw=3)
x = [outline[0,0], outline[-1,0]]
y = [outline[0,1], outline[-1,1]]
ax[1,0].plot(x, y, c="C5",lw=3)
ax[1,0].set_xlim(center[1]-r,center[1]+r)
ax[1,0].set_ylim(center[2]-r,center[2]+r)
ax[1,0].grid(False)
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])
ax[1,0].set_ylabel("Border plane", fontsize=16)

center = np.rint(centers[c2]).astype("int32")
outline = outlines[c2]
mask = masks[c2]
z = center[0]

ax[1,1].imshow(stack[z])
ax[1,1].plot(outline[:,0], outline[:,1], c="C5",lw=3)
x = [outline[0,0], outline[-1,0]]
y = [outline[0,1], outline[-1,1]]
ax[1,1].plot(x, y, c="C5",lw=3)
ax[1,1].set_xlim(center[1]-r,center[1]+r)
ax[1,1].set_ylim(center[2]-r,center[2]+r)
ax[1,1].grid(False)
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])

ax[1,2].hist(F3, bins=30, color=[0.0, 0.8, 0.0], alpha=0.4)
ax[1,2].set_xlabel("mean H2B-mCherry [a.u.]")

ax[1,2].hist(A12, bins=30, color=[0.8, 0.0, 0.8], alpha=0.4)
ax[1,2].set_xlabel("mean H2B-emiRFP [a.u.]")
ax[1,2].set_yticks([])
ax[1,2].sharex(ax[0,2])
ax[1,2].spines[['right', 'top', "left"]].set_visible(False)

ax[1,3].set_xlabel("mean H2B-mCherry [a.u.]")
ax[1,3].set_ylabel("mean H2B-emiRFP [a.u.]")
ax[1,3].scatter(F3_all, A12_all,c=colors, edgecolors='none')
ax[1,3].set_xlim(-5, 160)
ax[1,3].set_ylim(-5, 160)
ax[1,3].set_xticks([0,50,100,150])
ax[1,3].set_yticks([0,50,100,150])
ax[1,3].spines[['right', 'top']].set_visible(False)

plt.tight_layout()
plt.savefig(path_figures+"pre_quant.svg")
plt.show()