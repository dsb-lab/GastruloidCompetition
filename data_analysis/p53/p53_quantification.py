### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, check_or_create_dir, get_file_names

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

import numpy as np
import matplotlib.pyplot as plt

files_to_exclude = [
    "n2_F3(150)+WT(150)_72h_emiRFP-p53-mCh-DAPI_(40xSil)_Stack1.tif",
    "n2_F3(150)+WT(150)_72h_emiRFP-p53-mCh-DAPI_(40xSil)_Stack2.tif"
]

CONDS = ["WT", "KO"]
repeats = ["n2", "n3", "n4"]

extreme_val_thresholds = [8887.94087721909, 7176.828172813844, 6146.195054116493, 5829.832880998152, 5206.4655345066585, 4966.53252355899, 4250.089479850633, 3973.264464806628, 3614.7661866441063, 3332.6717493337082]
# extreme_val_thresholds = [7418.227507647827, 6050.493891032715, 5209.560718240686, 4935.292612696378, 4426.689971542723, 4218.030557837337, 3639.625770837681, 3406.214860957358, 3110.682483789724, 2875.9321309865254]
# extreme_val_thresholds = [5948.5141380765635, 4924.159609251585, 4272.926382364876, 4040.7523443946034, 3646.9144085787852, 3469.5285921156833, 3029.1620618247284, 2839.1652571080867, 2606.598780935342, 2419.1925126393435]

ExtremesF3 = {}
ExtremesA12 = {}
for COND in CONDS:
    ExtremesF3[COND] = {}
    ExtremesA12[COND] = {}

    for REP in repeats:
        ExtremesF3[COND][REP] = []
        ExtremesA12[COND][REP] = []

        path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/input/{}/{}/".format(COND,REP)
        path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/{}/{}/".format(COND,REP)

        check_or_create_dir(path_save_dir)

        files = get_file_names(path_data_dir)

        channel_names = ["A12", "p53", "F3", "DAPI"]
                    
        ch_F3 = channel_names.index("F3")
        ch_A12 = channel_names.index("A12")
        ch_p53 = channel_names.index("p53")
        ch_DAPI = channel_names.index("DAPI")
        
        for f, file in enumerate(files):
            
            if file in files_to_exclude: continue
            ExtremesF3[COND][REP].append(0)
            ExtremesA12[COND][REP].append(0)

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
                
                p53_val = np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]])
                if p53_val > extreme_val_thresholds[z]:
                    ExtremesF3[COND][REP][-1]+=1
            
            ExtremesF3[COND][REP][-1]/=len(CT_F3.jitcells)
            
            for cell in CT_A12.jitcells:
                center = cell.centers[0]
                z = int(center[0])
                zid = cell.zs[0].index(z)
                mask = cell.masks[0][zid]
                
                p53_val = np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]])
                if p53_val > extreme_val_thresholds[z]:
                    ExtremesA12[COND][REP][-1]+=1
                    
            ExtremesA12[COND][REP][-1]/=len(CT_A12.jitcells)

ExtremesF3_WT_means = [np.mean(ExtremesF3["WT"][REP]) for REP in repeats]
ExtremesF3_WT_stds = [np.std(ExtremesF3["WT"][REP]) for REP in repeats]

ExtremesF3_KO_means = [np.mean(ExtremesF3["KO"][REP]) for REP in repeats]
ExtremesF3_KO_stds = [np.std(ExtremesF3["KO"][REP]) for REP in repeats]

# Set up bar positions
x = np.arange(len(repeats))   # one position per REP
width = 0.35                  # width of each bar

fig, ax = plt.subplots()

# Bars
bars_WT = ax.bar(x - width/2, ExtremesF3_WT_means, width,
                 yerr=ExtremesF3_WT_stds, label="F3 with WT",
                 color=(0.0, 0.8, 0.0), capsize=5)

bars_KO = ax.bar(x + width/2, ExtremesF3_KO_means, width,
                 yerr=ExtremesF3_KO_stds, label="F3 with KO",
                 color="cyan", capsize=5)

# Labels in the middle of each pair
ax.set_xticks(x)                  # one tick per pair
ax.set_xticklabels(repeats)       # label with REP names

# Axes labels & legend
ax.set_ylabel("Ratio of p53-high cells")
ax.set_title("WT vs KO (F3)")
ax.legend()

plt.show()

all_extremeF3_WT = [val for REP in repeats for val in ExtremesF3["WT"][REP]]
all_extremeF3_KO = [val for REP in repeats for val in ExtremesF3["KO"][REP]]

all_extremeA12_WT = [val for REP in repeats for val in ExtremesA12["WT"][REP]]
all_extremeA12_KO = [val for REP in repeats for val in ExtremesA12["KO"][REP]]


from scipy.stats import ttest_ind
from scipy.stats import shapiro

stat, p_shapiro = shapiro(all_extremeF3_WT)
print("WT Shapiro-Wilk p =", p_shapiro)

stat, p_shapiro = shapiro(all_extremeF3_KO)
print("KO Shapiro-Wilk p =", p_shapiro)

t_stat, p_t = ttest_ind(all_extremeF3_WT, all_extremeF3_KO, equal_var=False)
print(f"T-test: t={t_stat:.3f}, p={p_t:.3e}")

from scipy.stats import mannwhitneyu

u_stat, p_mw = mannwhitneyu(all_extremeF3_WT, all_extremeF3_KO, alternative="two-sided")
print(f"Mann–Whitney U: U={u_stat}, p={p_mw:.3e}")

from scipy.stats import ks_2samp

ks_stat, p_ks = ks_2samp(all_extremeF3_WT, all_extremeF3_KO)
print(f"KS test: D={ks_stat:.3f}, p={p_ks:.3e}")



# Bar positions
labels = ["A12 - KO","A12 - WT", "F3 with WT", "F3 with KO"]
means = [np.mean(all_extremeA12_KO), np.mean(all_extremeA12_WT), np.mean(all_extremeF3_WT), np.mean(all_extremeF3_KO)]
stds = [np.std(all_extremeA12_KO), np.std(all_extremeA12_WT), np.std(all_extremeF3_WT), np.std(all_extremeF3_KO)]
colors = [(0.6, 0.0, 0.6), (0.8, 0.0, 0.8), (0.0, 0.8, 0.0), "cyan"]

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
ax.hist(all_extremeA12_KO, bins=bins, color=colors[0], alpha=0.5)
ax.hist(all_extremeA12_WT, bins=bins, color=colors[1], alpha=0.5)
ax.hist(all_extremeF3_WT, bins=bins, color=colors[2], alpha=0.5)
ax.hist(all_extremeF3_KO, bins=bins, color=colors[3], alpha=0.5)
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

