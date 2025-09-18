### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, check_or_create_dir, get_file_names

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

F3 = [[] for z in range(10)]
A12 = [[] for z in range(10)]
colors = [[] for z in range(10)]
zs = []

CONDS = ["secondaryonly"]
files_to_exclude = [
    # "F3(150)+OsTIR9-40(25)_48h_emiRFP-2ndaryA488-mCh-DAPI_(40xSil)_Stack1.tif",
    "F3(150)+OsTIR9-40(25)_48h_emiRFP-2ndaryA488-mCh-DAPI_(40xSil)_Stack1_new filter.tif"
]

calibF3 = np.load("/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/secondaryonly/F3(150)+OsTIR9-40(25)_48h_emiRFP-2ndaryA488-mCh-DAPI_(40xSil)_Stack1/calibration_F3_to_p53.npz")
p53_F3_s_global = float(calibF3["s"])
p53_F3_0z = calibF3["b0z"]

calibA12 = np.load("/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/secondaryonly/F3(150)+OsTIR9-40(25)_48h_emiRFP-2ndaryA488-mCh-DAPI_(40xSil)_Stack1/calibration_A12_to_p53.npz")
p53_A12_s_global = float(calibA12["s"])
p53_A12_0z = calibA12["b0z"]


for C, COND in enumerate(CONDS):
    path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
    path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
        
    check_or_create_dir(path_save_dir)

    files = get_file_names(path_data_dir)

    channel_names = ["A12", "p53", "F3", "DAPI"]

    for f, file in enumerate(files):
        
        if file in files_to_exclude: continue
        
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

        ch = channel_names.index("A12")
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
        zs.append(CT_A12.hyperstack.shape[1])
  
        ch_p53 = channel_names.index("p53")

        for cell in CT_F3.jitcells:
            center = cell.centers[0]
            z = int(center[0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]
            F3[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]]))
            colors[z].append([0.0,0.8,0.0, 0.3])
            
        for cell in CT_A12.jitcells:
            center = cell.centers[0]
            z = int(center[0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]
            A12[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]]))
            colors[z].append([0.8,0.0,0.8, 0.3])


F3_means = np.array([np.mean(f3) for f3 in F3])
F3_stds = np.array([np.std(f3) for f3 in F3])

A12_means = np.array([np.mean(a12) for a12 in A12])
A12_stds = np.array([np.std(a12) for a12 in A12])

fig, ax = plt.subplots()

ax.plot(range(10), F3_means, color=[0.0,0.8,0.0, 1.0], label="F3")
ax.fill_between(range(10), F3_means - F3_stds, F3_means + F3_stds, color=[0.0,0.8,0.0], alpha=0.2)

ax.plot(range(10), A12_means, color=[0.8,0.0,0.8, 1.0], label="A12")
ax.fill_between(range(10), A12_means - A12_stds, A12_means + A12_stds, color=[0.8,0.0,0.8], alpha=0.2)

ax.set_xlabel("z")
ax.set_ylabel("p53 [a.u.]")
ax.title("p53 background analysis")
ax.legend()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# --- assumes you already have F3 and A12 as lists/arrays of 1D arrays per z ---
# e.g. F3[z] -> values for F3 at slice z, same for A12[z]
F3_means = np.array([np.mean(f3) for f3 in F3])
F3_stds  = np.array([np.std(f3)  for f3 in F3])

A12_means = np.array([np.mean(a12) for a12 in A12])
A12_stds  = np.array([np.std(a12)  for a12 in A12])

assert len(F3_means) == len(A12_means), "F3 and A12 must have the same number of z-slices"
n_z = len(F3_means)

# Colors (match your lines)
col_F3  = (0.0, 0.8, 0.0, 0.7)
col_A12 = (0.8, 0.0, 0.8, 0.7)

# X positions
idx = np.arange(n_z)
bar_width = 0.38
x_F3  = idx - bar_width/2
x_A12 = idx + bar_width/2

# For reproducible jitter
rng = np.random.default_rng(42)

fig, ax = plt.subplots(figsize=(9, 4.8))

# Bars with STD error bars
bars_F3 = ax.bar(
    x_F3, F3_means, width=bar_width, yerr=F3_stds, capsize=3,
    color=col_F3, alpha=0.85, label="F3", zorder=2
)
bars_A12 = ax.bar(
    x_A12, A12_means, width=bar_width, yerr=A12_stds, capsize=3,
    color=col_A12, alpha=0.85, label="A12", zorder=2
)

# Jittered scatter of raw points for each z, centered on each bar
jitter_span = bar_width * 0.6  # how wide to spread points around the bar center
dot_size = 18

for z in range(n_z):
    # F3 points at this z
    f3_vals = np.asarray(F3[z])
    if f3_vals.size:
        xj = x_F3[z] + rng.uniform(-jitter_span/2, jitter_span/2, size=f3_vals.size)
        ax.scatter(xj, f3_vals, s=dot_size, marker='o', color=(col_F3[0], col_F3[1], col_F3[2], 0.5),
                   edgecolors="grey", zorder=3)

    # A12 points at this z
    a12_vals = np.asarray(A12[z])
    if a12_vals.size:
        xj = x_A12[z] + rng.uniform(-jitter_span/2, jitter_span/2, size=a12_vals.size)
        ax.scatter(xj, a12_vals, s=dot_size, marker='o', color=(col_A12[0], col_A12[1], col_A12[2], 0.5),
                   edgecolors="grey", zorder=3)

# Cosmetics
ax.set_xticks(idx)
ax.set_xticklabels([str(z) for z in range(n_z)])
ax.set_xlabel("z")
ax.set_ylabel("p53 [a.u.]")
ax.set_title("p53 background analysis")
ax.legend(frameon=False)
ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)
plt.tight_layout()
plt.show()

print("F3")
F3_means
print("A12")
A12_means

def correct_cell_pixels(CT_ref, mask, z, ch_B, ch_C, s, b0z):
    """Return per-pixel corrected C for one cell at plane z."""
    yy = mask[:, 1].astype(np.intp)
    xx = mask[:, 0].astype(np.intp)
    C_vals = CT_ref.hyperstack[0, z, ch_C, :, :][yy, xx].astype(np.float32)
    B_vals = CT_ref.hyperstack[0, z, ch_B, :, :][yy, xx].astype(np.float32)
    return C_vals - float(b0z[z]) - float(s) * B_vals

F3 = [[] for z in range(10)]
A12 = [[] for z in range(10)]
colors = [[] for z in range(10)]
zs = []

for C, COND in enumerate(CONDS):
    path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
    path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
        
    check_or_create_dir(path_save_dir)

    files = get_file_names(path_data_dir)

    channel_names = ["A12", "p53", "F3", "DAPI"]

    for f, file in enumerate(files):
        
        if file in files_to_exclude: continue
        
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

        ch = channel_names.index("A12")
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
        zs.append(CT_A12.hyperstack.shape[1])
  
        ch_p53 = channel_names.index("p53")
        ch_F3 = channel_names.index("F3")
        ch_A12 = channel_names.index("A12")

        for cell in CT_F3.jitcells:
            z = int(cell.centers[0][0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]
            Ccorr_vals = correct_cell_pixels(CT_F3, mask, z, ch_F3, ch_p53, p53_F3_s_global, p53_F3_0z)
            F3[z].append(float(np.mean(Ccorr_vals)))
            colors[z].append([0.0, 0.8, 0.0, 0.3])
                    
        for cell in CT_A12.jitcells:
            z = int(cell.centers[0][0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]
            Ccorr_vals = correct_cell_pixels(CT_A12, mask, z, ch_A12, ch_p53, p53_A12_s_global, p53_A12_0z)
            A12[z].append(float(np.mean(Ccorr_vals)))
            colors[z].append([0.8, 0.0, 0.8, 0.3])


import numpy as np
import matplotlib.pyplot as plt

# --- assumes you already have F3 and A12 as lists/arrays of 1D arrays per z ---
# e.g. F3[z] -> values for F3 at slice z, same for A12[z]
F3_means = np.array([np.mean(f3) for f3 in F3])
F3_stds  = np.array([np.std(f3)  for f3 in F3])

A12_means = np.array([np.mean(a12) for a12 in A12])
A12_stds  = np.array([np.std(a12)  for a12 in A12])

assert len(F3_means) == len(A12_means), "F3 and A12 must have the same number of z-slices"
n_z = len(F3_means)

# Colors (match your lines)
col_F3  = (0.0, 0.8, 0.0, 0.7)
col_A12 = (0.8, 0.0, 0.8, 0.7)

# X positions
idx = np.arange(n_z)
bar_width = 0.38
x_F3  = idx - bar_width/2
x_A12 = idx + bar_width/2

# For reproducible jitter
rng = np.random.default_rng(42)

fig, ax = plt.subplots(figsize=(9, 4.8))

# Bars with STD error bars
bars_F3 = ax.bar(
    x_F3, F3_means, width=bar_width, yerr=F3_stds, capsize=3,
    color=col_F3, alpha=0.85, label="F3", zorder=2
)
bars_A12 = ax.bar(
    x_A12, A12_means, width=bar_width, yerr=A12_stds, capsize=3,
    color=col_A12, alpha=0.85, label="A12", zorder=2
)

# Jittered scatter of raw points for each z, centered on each bar
jitter_span = bar_width * 0.6  # how wide to spread points around the bar center
dot_size = 18

for z in range(n_z):
    # F3 points at this z
    f3_vals = np.asarray(F3[z])
    if f3_vals.size:
        xj = x_F3[z] + rng.uniform(-jitter_span/2, jitter_span/2, size=f3_vals.size)
        ax.scatter(xj, f3_vals, s=dot_size, marker='o', color=(col_F3[0], col_F3[1], col_F3[2], 0.5),
                   edgecolors="grey", zorder=3)

    # A12 points at this z
    a12_vals = np.asarray(A12[z])
    if a12_vals.size:
        xj = x_A12[z] + rng.uniform(-jitter_span/2, jitter_span/2, size=a12_vals.size)
        ax.scatter(xj, a12_vals, s=dot_size, marker='o', color=(col_A12[0], col_A12[1], col_A12[2], 0.5),
                   edgecolors="grey", zorder=3)

# Cosmetics
ax.set_xticks(idx)
ax.set_xticklabels([str(z) for z in range(n_z)])
ax.set_xlabel("z")
ax.set_ylabel("p53 [a.u.]")
ax.set_title("p53 background analysis")
ax.legend(frameon=False)
ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)
plt.tight_layout()
plt.show()
