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

calibF3 = np.load("/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/secondaryonly/F3(150)+OsTIR9-40(25)_48h_emiRFP-2ndaryA488-mCh-DAPI_(40xSil)_Stack1/calibration_F3_to_p53.npz")
p53_F3_s_global = float(calibF3["s"])
p53_F3_0z = calibF3["b0z"]

calibA12 = np.load("/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/secondaryonly/F3(150)+OsTIR9-40(25)_48h_emiRFP-2ndaryA488-mCh-DAPI_(40xSil)_Stack1/calibration_A12_to_p53.npz")
p53_A12_s_global = float(calibA12["s"])
p53_A12_0z = calibA12["b0z"]

def build_union_masks(CT_list):
    """
    Build per-z 2D boolean masks marking in-cell pixels from the union of all cells
    across provided CT objects (e.g., CT_F3 and CT_A12).
    Returns: list of length Z with arrays (Y, X) dtype=bool.
    """
    CT0 = CT_list[0]
    Z = CT0.hyperstack.shape[1]
    Y = CT0.hyperstack.shape[-2]
    X = CT0.hyperstack.shape[-1]
    Mz_list = [np.zeros((Y, X), dtype=bool) for _ in range(Z)]
    for CT in CT_list:
        for cell in CT.jitcells:
            z = int(cell.centers[0][0])
            if z < 0 or z >= Z:
                continue
            # find mask for this z
            try:
                zid = cell.zs[0].index(z)
            except ValueError:
                continue
            mask = cell.masks[0][zid]
            yy = mask[:, 1].astype(np.intp)
            xx = mask[:, 0].astype(np.intp)
            Mz_list[z][yy, xx] = True
    return Mz_list

def estimate_b0z_for_file(CT, Mz_list, ch_B, ch_C, s_global, q=0.2):
    # q=0.5 (median) if few high-C cells; q=0.1–0.2 if many might be high
    import numpy as np
    Z = CT.hyperstack.shape[1]
    b0z = np.full(Z, np.nan, dtype=np.float64)
    for z in range(Z):
        Mz = Mz_list[z]
        if not np.any(Mz): continue
        Bz = CT.hyperstack[0, z, ch_B, :, :].astype(np.float64)
        Cz = CT.hyperstack[0, z, ch_C, :, :].astype(np.float64)
        resid = (Cz - s_global * Bz)[Mz].ravel()
        if resid.size < 50: continue
        b0z[z] = float(np.quantile(resid, q))
    # fill empties from available planes
    if np.any(np.isnan(b0z)):
        b0z[np.isnan(b0z)] = np.nanmedian(b0z)
    return b0z


def correct_cell_pixels(CT_ref, mask, z, ch_B, ch_C, s, b0z):
    """Return per-pixel corrected C for one cell at plane z."""
    yy = mask[:, 1].astype(np.intp)
    xx = mask[:, 0].astype(np.intp)
    C_vals = CT_ref.hyperstack[0, z, ch_C, :, :][yy, xx].astype(np.float32)
    B_vals = CT_ref.hyperstack[0, z, ch_B, :, :][yy, xx].astype(np.float32)
    return C_vals - float(b0z[z]) - float(s) * B_vals

extreme_val_thresholds = [13068.678466796875, 16818.155395507812, 12987.3173828125, 13056.200744628906, 11479.900085449219, 10226.168518066406, 9835.00520324707, 9196.476104736328, 7593.840606689453, 6879.673645019531]
extreme_val_thresholds = [8179.218505859375, 10396.903930664062, 8149.3251953125, 8101.474426269531, 7154.188659667969, 6463.565734863281, 6199.764053344727, 5759.637359619141, 4778.408111572266, 4321.629455566406]
# extreme_val_thresholds = [10623.948486328125, 13607.529663085938, 10568.3212890625, 10578.837585449219, 9317.044372558594, 8344.867126464844, 8017.384628295898, 7478.056732177734, 6186.124359130859, 5600.651550292969]
extreme_val_thresholds = [12504.209899902344, 15349.344207763672, 11434.93391418457, 11277.261016845703, 10070.376457214355, 8746.919189453125, 8621.400253295898, 7946.640838623047, 6647.256797790527, 5937.099349975586]
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

        ch_F3 = channel_names.index("F3")
        ch_A12 = channel_names.index("A12")
        ch_p53 = channel_names.index("p53")
        ch_DAPI = channel_names.index("DAPI")
        
        Mz_list = build_union_masks([CT_F3])
        p53_F3_0z = estimate_b0z_for_file(CT_F3, Mz_list, ch_F3, ch_p53, p53_F3_s_global)
        
        for cell in CT_F3.jitcells:
            center = cell.centers[0]
            z = int(center[0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]
            
            Ccorr_vals = correct_cell_pixels(CT_F3, mask, z, ch_F3, ch_p53, p53_F3_s_global, p53_F3_0z)
            p53_val = float(np.mean(Ccorr_vals))
            if p53_val > extreme_val_thresholds[z]:
                ExtremesF3[COND][-1]+=1
        
        ExtremesF3[COND][-1]/=len(CT_F3.jitcells)
        
        Mz_list = build_union_masks([CT_A12])
        p53_A12_0z = estimate_b0z_for_file(CT_A12, Mz_list, ch_A12, ch_p53, p53_A12_s_global)
        for cell in CT_A12.jitcells:
            center = cell.centers[0]
            z = int(center[0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]
            
            Ccorr_vals = correct_cell_pixels(CT_A12, mask, z, ch_A12, ch_p53, p53_A12_s_global, p53_A12_0z)
            p53_val = float(np.mean(Ccorr_vals))            
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
ax.set_ylabel("Proportion of p53-high cells")
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
comparisons = [(0,1), (1,2)] 

# Bar positions
# Bar positions
labels = ["F3 - auxin 48",
          "F3 - auxin 72", 
          "F3 - no auxin 72", 
          "A12 - auxin 48",
          "A12 - auxin 72", 
          "A12 - no auxin 72"]

data = [all_extremeF3_auxin48, 
         all_extremeF3_auxin72, 
         all_extremeF3_noauxin72, 
         all_extremeA12_auxin48, 
         all_extremeA12_auxin72, 
         all_extremeA12_noauxin72]


means = [np.mean(d) for d in data]
stds  = [np.std(d) for d in data]
colors = [(0.0, 0.8, 0.0), 
          "cyan", 
          (0.0, 0.8, 0.0), 
          (0.8, 0.0, 0.8),
          (0.5, 0.0, 0.5),
          (0.8, 0.0, 0.8)
          ]
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
ax.set_ylabel("Proportion of p53-high cells")
ax.set_title("Bar plot with individual datapoints")


for i, j in comparisons:
    vals1, vals2 = data[i], data[j]

    # Example: Welch’s t-test
    _, p = ttest_ind(vals1, vals2, equal_var=False)
    print(p)
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
# plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53/barplots.svg")
plt.show()


# import pandas as pd

# # Keep labels and data in the same order you plot
# labels = ["A12 - KO", "A12 - WT", "F3 with WT", "F3 with KO"]
# data = [all_extremeA12_KO, all_extremeA12_WT, all_extremeF3_WT, all_extremeF3_KO]

# # Build a dict of Series to handle different lengths (NaN padding)
# cols = {lab: pd.Series(vals) for lab, vals in zip(labels, data)}

# df = pd.DataFrame(cols)
# # Save to CSV
# path_save = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53/"
# df.to_csv(path_save+"barplot_underlying_data.csv", index=False)

