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

F3 = []
A12 = []

CONDS = ["auxin48", "auxin72", "noauxin72"]
F3_backgrounds = np.array([2496.39216396, 2717.58109258, 2523.45981179, 2206.13498618, 2097.02696383, 1921.61948137, 1828.3545534, 1700.845861, 1557.78809637, 1506.0134047])
A12_backgrounds = np.array([1615.10968548, 1539.48146973, 1527.88985255, 1458.18400646, 1404.53034921, 1352.31062397, 1306.44659299, 1262.34521779, 1203.13280638, 1176.99222449])
extreme_val_thresholds = [14826.371461477187, 17705.39551461342, 13941.872302924103, 13348.110504016084, 12133.609408707758, 10830.900530986986, 10363.001947595787, 9339.119899193325, 8004.611691222527, 7387.303160446984]

ExtremesF3 = {}
ExtremesA12 = {}
for COND in CONDS:
    ExtremesF3[COND] = []
    ExtremesA12[COND] = []
    
    path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
    path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
        
    check_or_create_dir(path_save_dir)

    files = get_file_names(path_data_dir)

    channel_names = ["A12", "p53", "F3", "DAPI"]
                
    ch_F3 = channel_names.index("F3")
    ch_A12 = channel_names.index("A12")
    ch_p53 = channel_names.index("p53")
    ch_DAPI = channel_names.index("DAPI")
    
    for f, file in enumerate(files):
        
        ExtremesF3[COND].append(0)
        ExtremesA12[COND].append(0)

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

        for cell in CT_F3.jitcells:
            center = cell.centers[0]
            z = int(center[0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]
            
            p53_val = np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]] - F3_backgrounds[z])
            if p53_val > extreme_val_thresholds[z]:
                ExtremesF3[COND][-1]+=1
        
        ExtremesF3[COND][-1]/=len(CT_F3.jitcells)
        
        for cell in CT_A12.jitcells:
            center = cell.centers[0]
            z = int(center[0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]
            
            p53_val = np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]] - A12_backgrounds[z])
            if p53_val > extreme_val_thresholds[z]:
                ExtremesA12[COND][-1]+=1
                
        ExtremesA12[COND][-1]/=len(CT_A12.jitcells)

all_extremeF3_auxin48 = ExtremesF3["auxin48"]
all_extremeF3_auxin72 = ExtremesF3["auxin72"]
all_extremeF3_noauxin72 = ExtremesF3["noauxin72"]

all_extremeA12_auxin48 = ExtremesA12["auxin48"]
all_extremeA12_auxin72 = ExtremesA12["auxin72"]
all_extremeA12_noauxin72 = ExtremesA12["noauxin72"]

# Bar positions
labels = ["F3 - auxin 48",
          "F3 - auxin 72", 
          "F3 - no auxin 72", 
          "A12 - auxin 48",
          "A12 - auxin 72", 
          "A12 - no auxin 72"]

means = [np.mean(all_extremeF3_auxin48), 
         np.mean(all_extremeF3_auxin72), 
         np.mean(all_extremeF3_noauxin72), 
         np.mean(all_extremeA12_auxin48), 
         np.mean(all_extremeA12_auxin72), 
         np.mean(all_extremeA12_noauxin72)]

stds = [np.std(all_extremeF3_auxin48), 
         np.std(all_extremeF3_auxin72), 
         np.std(all_extremeF3_noauxin72), 
         np.std(all_extremeA12_auxin48), 
         np.std(all_extremeA12_auxin72), 
         np.std(all_extremeA12_noauxin72)]

colors = [(0.0, 0.8, 0.0), 
          "cyan", 
          (0.0, 0.8, 0.0), 
          (0.8, 0.0, 0.8),
          (0.5, 0.0, 0.5),
          (0.8, 0.0, 0.8)
          ]

x = np.arange(len(labels))

# Plot
fig, ax = plt.subplots()
bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors)

# Labels and formatting
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Ratio of p53-high cells")
plt.show()

bins = 10
fig, ax = plt.subplots()
ax.hist(all_extremeF3_auxin48, bins=bins, color=colors[0], alpha=0.5)
ax.hist(all_extremeF3_auxin72, bins=bins, color=colors[1], alpha=0.5)
ax.hist(all_extremeF3_noauxin72, bins=bins, color=colors[2], alpha=0.5)
ax.hist(all_extremeA12_auxin48, bins=bins, color=colors[3], alpha=0.5)
ax.hist(all_extremeA12_auxin72, bins=bins, color=colors[4], alpha=0.5)
ax.hist(all_extremeA12_noauxin72, bins=bins, color=colors[5], alpha=0.5)
plt.show()

from scipy.stats import ttest_ind

# Define the comparisons you want to annotate (pairs of bar indices)
comparisons = [(1,2), (2,3)]  # e.g. A12 KO vs A12 WT, F3 WT vs F3 KO

# Bar positions
labels = ["A12 - KO", "A12 - WT", "F3 with WT", "F3 with KO"]
data = [all_extremeA12_KO, all_extremeA12_WT, all_extremeF3_WT, all_extremeF3_KO]

means = [np.mean(d) for d in data]
stds  = [np.std(d) for d in data]
colors = [(0.6, 0.0, 0.6), (0.8, 0.0, 0.8), (0.0, 0.8, 0.0), "cyan"]
x = np.arange(len(labels))

# Run tests and add bars
ymax = max(max(d) for d in data) * 1.01  # starting height for bars
h = 0.02 * ymax   # height of bars
step = 0

fig, ax = plt.subplots(figsize=(8,5))

# Plot bars
bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7)

# Overlay scatter points
for i, vals in enumerate(data):
    # random jitter inside the bar width
    jitter = np.random.uniform(-0.2, 0.2, size=len(vals))  
    ax.scatter(np.full(len(vals), x[i]) + jitter, vals,
               color="black", s=20, alpha=0.7, zorder=3)

# Labels and formatting
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20)
ax.set_ylabel("Ratio of p53-high cells")
ax.set_title("Bar plot with individual datapoints")


for i, j in comparisons:
    vals1, vals2 = data[i], data[j]

    # Example: Welch’s t-test
    _, p = ttest_ind(vals1, vals2, equal_var=False)

    # Bar coordinates
    x1, x2 = x[i], x[j]
    y = ymax + step*h
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')

    # Decide significance stars
    if p < 0.001:
        stars = '***'
    elif p < 0.01:
        stars = '**'
    elif p < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'  # not significant

    ax.text((x1+x2)/2, y+h, stars, ha='center', va='bottom', fontsize=12)

    step += 4  # space out multiple bars

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53/barplots.svg")
plt.show()


import pandas as pd

# Keep labels and data in the same order you plot
labels = ["A12 - KO", "A12 - WT", "F3 with WT", "F3 with KO"]
data = [all_extremeA12_KO, all_extremeA12_WT, all_extremeF3_WT, all_extremeF3_KO]

# Build a dict of Series to handle different lengths (NaN padding)
cols = {lab: pd.Series(vals) for lab, vals in zip(labels, data)}

df = pd.DataFrame(cols)
# Save to CSV
path_save = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53/"
df.to_csv(path_save+"barplot_underlying_data.csv", index=False)

