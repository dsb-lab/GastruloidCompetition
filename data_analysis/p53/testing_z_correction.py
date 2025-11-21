### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, check_or_create_dir, get_file_names, get_intenity_profile

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams.update({
    "text.usetex": True,
})
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath'
mpl.rc('font', size=18) 
mpl.rc('axes', labelsize=18) 
mpl.rc('xtick', labelsize=18) 
mpl.rc('ytick', labelsize=18) 
mpl.rc('legend', fontsize=18) 

CONDS = ["WT", "KO"]
repeats = ["n2", "n3", "n4"]


COND = CONDS[0]
REP = repeats[0]

path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/input/{}/{}/".format(COND,REP)
path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/{}/{}/".format(COND,REP)

check_or_create_dir(path_save_dir)
files = get_file_names(path_data_dir)

channel_names = ["A12", "p53", "F3", "DAPI"]
    
ch_F3 = channel_names.index("F3")
ch_A12 = channel_names.index("A12")
ch_p53 = channel_names.index("p53")
ch_DAPI = channel_names.index("DAPI")

file = files[0]

path_data = path_data_dir+file
file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
path_save = path_save_dir+embcode
    
check_or_create_dir(path_save)

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    'blur': [2,1], 
    'min_outline_length':100,
}

concatenation3D_args = {
    'do_3Dconcatenation': False
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'points',
}

ch = channel_names.index("F3")

batch_args = {
    'name_format':"ch"+str(ch)+"_{}",
    'extension':".tif",
} 
plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    'plot_stack_dims': (256, 256), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[ch],
    'min_outline_length':75,
}

chans = [ch]
for _ch in range(len(channel_names)):
    if _ch not in chans:
        chans.append(_ch)
        
CT_F3 = cellSegTrack(
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

zn = CT_F3.hyperstack.shape[1]
p53_F3 = [[] for z in range(zn)]
p53_A12 = [[] for z in range(zn)]

ch = channel_names.index("A12")
batch_args = {
    'name_format':"ch"+str(ch)+"_{}",
    'extension':".tif",
}

chans = [ch]
for _ch in range(len(channel_names)):
    if _ch not in chans:
        chans.append(_ch)

CT_A12 = cellSegTrack(
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

chs_prof = []
chs_corr = []
for ch in range(CT_A12.hyperstack.shape[2]):
    correction_function, intensity_profile, z_positions = get_intenity_profile(CT_A12, ch)
    chs_prof.append(intensity_profile)
    chs_corr.append(correction_function)


import numpy as np

fig, ax = plt.subplots(2,2, sharex=True)

x = range(len(chs_prof[0]))
ch = 0
ax[0,0].plot(x, chs_corr[ch])
ax[0,0].set_ylabel(channel_names[ch])
ax[0,0].legend()

ch = 1
ax[0,1].plot(x, chs_corr[ch])
ax[0,1].set_ylabel(channel_names[ch])
ax[0,1].legend()

ch = 2
ax[1,0].plot(x, chs_corr[ch])
ax[1,0].set_ylabel(channel_names[ch])
ax[1,0].set_xlabel("% of total depth")
ax[1,0].legend()

ch = 3
ax[1,1].plot(x, chs_corr[ch])
ax[1,1].set_ylabel(channel_names[ch])
ax[1,1].set_xlabel("% of total depth")
ax[1,1].legend()

plt.tight_layout()
plt.show()


import numpy as np

fig, ax = plt.subplots(2,2, sharex=True)

x = range(len(chs_prof[0]))
ch = 0
ax[0,0].plot(x, chs_prof[ch])
ax[0,0].set_ylabel(channel_names[ch])
ax[0,0].legend()

ch = 1
ax[0,1].plot(x, chs_prof[ch])
ax[0,1].set_ylabel(channel_names[ch])
ax[0,1].legend()

ch = 2
ax[1,0].plot(x, chs_prof[ch])
ax[1,0].set_ylabel(channel_names[ch])
ax[1,0].set_xlabel("% of total depth")
ax[1,0].legend()

ch = 3
ax[1,1].plot(x, chs_prof[ch])
ax[1,1].set_ylabel(channel_names[ch])
ax[1,1].set_xlabel("% of total depth")
ax[1,1].legend()

plt.tight_layout()
plt.show()
