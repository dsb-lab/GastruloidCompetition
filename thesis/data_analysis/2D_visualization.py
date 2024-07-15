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

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/thesis/figures/figures/visualization/"

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



### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/96hr/WT/'
path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/96hr/WT/'

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

# CT_Casp3.plot_tracking(plot_args=plot_args)

batch_args = {
    'name_format':"{}",
    'name_format_save':"all_{}",
    'extension':".tif",
}
CT_all2 = CellTracking(
    path_data,
    path_save,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    error_correction_args=error_correction_args,
    plot_args=plot_args,
    batch_args=batch_args,
    channels=chans
)
CT_all2.run()

CT_all2.load()
labs = [cell.label for cell in CT_all2.jitcells]
for lab in labs:
    CT_all2._del_cell(lab)

CT_all2.update_labels()

for cell in CT_F3.jitcells:
    CT_all2.append_cell_from_cell(cell)

for cell in CT_A12.jitcells:
    CT_all2.append_cell_from_cell(cell)

CT_all2.update_labels()

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    # 'plot_stack_dims': (256, 256), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[ch_A12, ch_F3, ch_A12],
    'min_outline_length':75,
}

CT_all2.plot_tracking(plot_args=plot_args)


def channelnorm(im, channel, vmin, vmax):
    c = (im[:,:,channel]-vmin) / (vmax-vmin)
    print(c.shape)
    c[c<0.] = 0.0
    c[c>1.] = 1.0
    im[:,:,channel] = c
    return im

    
z = 24
stack1 = construct_RGB(R=CT_all1.hyperstack[0,:,1,:,:], G=CT_all1.hyperstack[0,:,0,:,:], B=CT_all1.hyperstack[0,:,1,:,:])
stack2 = construct_RGB(R=CT_all2.hyperstack[0,:,1,:,:], G=CT_all2.hyperstack[0,:,0,:,:], B=CT_all2.hyperstack[0,:,1,:,:])

stack1 = stack1/255.0
stack2 = stack2/255.0

cmap = plt.get_cmap("tab10")

fig, ax = plt.subplots(1,3, figsize=(12,3.8))

stack1[z,:,:] = channelnorm(stack1[z], 0, vmin=0.0, vmax=0.8)
stack1[z,:,:] = channelnorm(stack1[z], 1, vmin=0.0, vmax=0.8)
stack1[z,:,:] = channelnorm(stack1[z], 2, vmin=0.0, vmax=0.8)

ax[0].imshow(stack1[z])
count = 0
for cell in CT_all1.jitcells:
    if z in cell.zs[0]:
        zid = cell.zs[0].index(z)
        outline = cell.outlines[0][zid]
        ax[0].plot(outline[:,0], outline[:,1], c=cmap(count%10), lw=0.5)
        endx = [outline[-1,0], outline[0,0]]
        endy = [outline[-1,1], outline[0,1]]
        ax[0].plot(endx, endy, c=cmap(count%10), lw=0.5)
        count+=1
ax[0].set_xlim(50, 512-50)
ax[0].set_ylim(50, 512-50)
ax[0].axis('off')
ax[0].set_title("48 hr F3-WT - A12-WT")

xyres = CT_all2.metadata["XYresolution"]
scale_mic = 50
scale = scale_mic/xyres # microns
w, h = scale, 5
xy=[512-w-20-50, 512-40-100]
ax[0].add_patch(mpl.patches.Rectangle(xy, w, h, color="white", zorder=20))
ax[0].text(xy[0]+23, 512-8-100, r"{:d} $\mu$m".format(scale_mic), color="white", fontsize=10)


z=43
stack2[z,:,:] = channelnorm(stack2[z], 0, vmin=0.0, vmax=0.2)
stack2[z,:,:] = channelnorm(stack2[z], 1, vmin=0.0, vmax=0.4)
stack2[z,:,:] = channelnorm(stack2[z], 2, vmin=0.0, vmax=0.2)

ax[1].imshow(stack2[z])
count = 0
for cell in CT_all2.jitcells:
    if z in cell.zs[0]:
        zid = cell.zs[0].index(z)
        outline = cell.outlines[0][zid]
        ax[1].plot(outline[:,0], outline[:,1], c=cmap(count%10), lw=0.5)
        endx = [outline[-1,0], outline[0,0]]
        endy = [outline[-1,1], outline[0,1]]
        ax[1].plot(endx, endy, c=cmap(count%10), lw=0.5)
        count+=1
# ax[1].set_xlim(50, 512-50)
# ax[1].set_ylim(50, 512-50)
ax[1].axis('off')
ax[1].set_title("96 hr F3-WT - A12-WT")

xyres = CT_all2.metadata["XYresolution"]
scale_mic = 50
scale = scale_mic/xyres # microns
w, h = scale, 5
xy=[512-w-20, 512-40]
ax[1].add_patch(mpl.patches.Rectangle(xy, w, h, color="white", zorder=20))
ax[1].text(xy[0]+23, 512-8, r"{:d} $\mu$m".format(scale_mic), color="white", fontsize=10)

im = np.ones_like(stack1, dtype="float32")
ax[2].imshow(im[0])
ax[2].set_title("Segmentation 48 hr F3-WT")
ax[2].axis('off')
plt.tight_layout()
plt.savefig(path_figures+"48_96.svg")
plt.show()
