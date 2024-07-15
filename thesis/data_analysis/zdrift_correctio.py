### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, get_file_names, correct_path, save_4Dstack, arboretum_napari
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
color_list = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
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

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/thesis/figures/figures/zdrift/"

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
file = files[0]
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

from embdevtools import extract_fluoro, get_intenity_profile
results_F3 = extract_fluoro(CT_F3)
ch = channel_names.index("F3")
f3 = results_F3["channel_{}".format(ch)]

correction_function_F3, intensity_profile_F3, z_positions_F3 = get_intenity_profile(CT_F3, ch_F3, cell_number_threshold=7.0, fit_everything=False)
import matplotlib.pyplot as plt

stack = CT_F3.hyperstack[0,:,ch_F3].astype("float32")
for z in range(stack.shape[0]):
    stack[z] = stack[z] / correction_function_F3[z]
stack *= np.mean(intensity_profile_F3)
CT_F3.hyperstack[0,:,ch_F3] = stack.astype("uint8")

results_F3_corrected = extract_fluoro(CT_F3)
import matplotlib.pyplot as plt
ch = channel_names.index("F3")
f3_corrected = results_F3_corrected["channel_{}".format(ch)]

from embdevtools import extract_fluoro, get_intenity_profile
results_A12 = extract_fluoro(CT_A12)
ch = channel_names.index("A12")
a12 = results_A12["channel_{}".format(ch)]

correction_function_A12, intensity_profile_A12, z_positions_A12 = get_intenity_profile(CT_A12, ch_A12, cell_number_threshold=6.0, fit_everything=False)
stack = CT_A12.hyperstack[0,:,ch_A12].astype("float32")
for z in range(stack.shape[0]):
    stack[z] = stack[z] / correction_function_A12[z]
stack *= np.mean(intensity_profile_A12)
CT_A12.hyperstack[0,:,ch_A12] = stack.astype("uint8")

results_A12_corrected = extract_fluoro(CT_A12)
import matplotlib.pyplot as plt
ch = channel_names.index("A12")
a12_corrected = results_A12_corrected["channel_{}".format(ch)]

F3_pre = []
A12_pre = []
for cell in CT_F3.jitcells:
    center = cell.centers_all[0][0]
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    outline = cell.outlines[0][zid]
    F3_pre.append(np.mean(CT_A12.hyperstack[0,z,0,:,:][mask[:,1], mask[:,0]]))

for cell in CT_A12.jitcells:
    center = cell.centers_all[0][0]
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    outline = cell.outlines[0][zid]
    A12_pre.append(np.mean(CT_F3.hyperstack[0,z,1,:,:][mask[:,1], mask[:,0]]))

F3_post = []
A12_post = []
for cell in CT_F3.jitcells:
    center = cell.centers_all[0][0]
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    outline = cell.outlines[0][zid]
    F3_post.append(np.mean(CT_F3.hyperstack[0,z,0,:,:][mask[:,1], mask[:,0]]))

for cell in CT_A12.jitcells:
    center = cell.centers_all[0][0]
    z = int(center[0])
    zid = cell.zs[0].index(z)

    mask = cell.masks[0][zid]
    outline = cell.outlines[0][zid]
    A12_post.append(np.mean(CT_A12.hyperstack[0,z,1,:,:][mask[:,1], mask[:,0]]))

z=50
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,4, figsize=(14,4), width_ratios=[1.2,1.2,1.5,1.5])
mean1 = np.mean(correction_function_F3)
mean2 = np.mean(correction_function_A12)
ax[0].plot(range(CT_F3.slices), correction_function_F3, ls="-", lw=3, label="H2B-mCherry", color=[0.0, 0.8, 0.0])
ax[0].plot(range(CT_F3.slices), correction_function_A12, ls="-", lw=3, label="H2B-emiRFP", color=[0.8, 0.0, 0.8])
ax[0].set_ylabel(r"$P$")
ax[0].set_xlabel(r"$z$")
ax[1].set_xlabel(r"$z$")

ax[1].plot(range(CT_F3.slices), correction_function_F3 , ls="-", lw=3, label="H2B-mCherry", color=[0.0, 0.8, 0.0])
ax[1].plot(range(CT_F3.slices), correction_function_A12-(mean2-mean1) , ls="-", lw=3, label="H2B-emiRFP", color=[0.8, 0.0, 0.8])
ax[1].sharey(ax[0])
plt.setp(ax[1].get_yticklabels(), visible=False)

ax[2].set_title("H2B-mCherry")
ax[2].hist(F3_pre, alpha=0.5)
ax[2].hist(F3_post, alpha=0.8)
ax[2].spines[['right', 'top', "left"]].set_visible(False)
ax[2].set_yticks([])

ax[3].set_title("H2B-emiRFP")
ax[3].hist(A12_pre, alpha=0.5, label="uncorrected")
ax[3].hist(A12_post, alpha=0.8, label="corrected")
ax[3].spines[['right', 'top', "left"]].set_visible(False)
ax[3].legend(loc="upper right")
ax[3].set_yticks([])
plt.tight_layout()
plt.savefig(path_figures+"zdrift.svg")
plt.show()
