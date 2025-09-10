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


F3_all = [[] for z in range(10)]
F3_F3 = [[] for z in range(10)]
F3_A12 = [[] for z in range(10)]
F3_DAPI = [[] for z in range(10)]
F3_p53 = [[] for z in range(10)]

F3_p53_WT = [[] for z in range(10)]
F3_p53_KO = [[] for z in range(10)]

A12_p53_WT = [[] for z in range(10)]
A12_p53_KO = [[] for z in range(10)]

A12_all = [[] for z in range(10)]
A12_F3 = [[] for z in range(10)]
A12_A12 = [[] for z in range(10)]
A12_DAPI = [[] for z in range(10)]
A12_p53 = [[] for z in range(10)]

DAPI_all = [[] for z in range(10)]

colors = [[] for z in range(10)]

files_to_exclude = [
    "n2_F3(150)+WT(150)_72h_emiRFP-p53-mCh-DAPI_(40xSil)_Stack1.tif",
    "n2_F3(150)+WT(150)_72h_emiRFP-p53-mCh-DAPI_(40xSil)_Stack2.tif"
]

CONDS = ["WT", "KO"]

repeats = ["n2", "n3", "n4"]

zs = []

all_files = []

for COND in CONDS:
    for REP in repeats:
        
        path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/input/{}/{}/".format(COND,REP)
        path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/{}/{}/".format(COND,REP)

        check_or_create_dir(path_save_dir)

        files = get_file_names(path_data_dir)

        channel_names = ["A12", "p53", "F3", "DAPI"]

        for f, file in enumerate(files):
            
            if file in files_to_exclude: continue
            
            all_files.append(file)
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

            zs.append(CT_A12.hyperstack.shape[1])
            
            ch_F3 = channel_names.index("F3")
            ch_A12 = channel_names.index("A12")
            ch_p53 = channel_names.index("p53")
            ch_DAPI = channel_names.index("DAPI")

            for cell in CT_F3.jitcells:
                center = cell.centers[0]
                z = int(center[0])
                zid = cell.zs[0].index(z)
                mask = cell.masks[0][zid]

                F3_all[z].append(np.mean(CT_A12.hyperstack[0,z,ch_F3,:,:][mask[:,1], mask[:,0]]))
                A12_all[z].append(np.mean(CT_A12.hyperstack[0,z,ch_A12,:,:][mask[:,1], mask[:,0]]))
                DAPI_all[z].append(np.mean(CT_A12.hyperstack[0,z,ch_DAPI,:,:][mask[:,1], mask[:,0]]))

                F3_F3[z].append(np.mean(CT_A12.hyperstack[0,z,ch_F3,:,:][mask[:,1], mask[:,0]]))
                F3_A12[z].append(np.mean(CT_A12.hyperstack[0,z,ch_A12,:,:][mask[:,1], mask[:,0]]))
                F3_DAPI[z].append(np.mean(CT_A12.hyperstack[0,z,ch_DAPI,:,:][mask[:,1], mask[:,0]]))
                F3_p53[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]]))
                
                if COND=="WT":
                    F3_p53_WT[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]]))
                else:
                    F3_p53_KO[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]]))

                colors[z].append([0.0,0.8,0.0, 0.3])
                
            for cell in CT_A12.jitcells:
                center = cell.centers[0]
                z = int(center[0])
                zid = cell.zs[0].index(z)
                mask = cell.masks[0][zid]

                F3_all[z].append(np.mean(CT_A12.hyperstack[0,z,ch_F3,:,:][mask[:,1], mask[:,0]]))
                A12_all[z].append(np.mean(CT_A12.hyperstack[0,z,ch_A12,:,:][mask[:,1], mask[:,0]]))
                DAPI_all[z].append(np.mean(CT_A12.hyperstack[0,z,ch_DAPI,:,:][mask[:,1], mask[:,0]]))
                
                A12_F3[z].append(np.mean(CT_A12.hyperstack[0,z,ch_F3,:,:][mask[:,1], mask[:,0]]))
                A12_A12[z].append(np.mean(CT_A12.hyperstack[0,z,ch_A12,:,:][mask[:,1], mask[:,0]]))
                A12_DAPI[z].append(np.mean(CT_A12.hyperstack[0,z,ch_DAPI,:,:][mask[:,1], mask[:,0]]))
                A12_p53[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]]))
                
                if COND=="WT":
                    A12_p53_WT[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]]))
                else:
                    A12_p53_KO[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]]))


                colors[z].append([0.8,0.0,0.8, 0.3])


F3_means = np.array([np.mean(f3) for f3 in F3_F3])
F3_stds = np.array([np.std(f3) for f3 in F3_F3])

A12_means = np.array([np.mean(a12) for a12 in A12_A12])
A12_stds = np.array([np.std(a12) for a12 in A12_A12])

DAPI_means = np.array([np.mean(dapi) for dapi in DAPI_all])
DAPI_stds = np.array([np.std(dapi) for dapi in DAPI_all])

F3_p53_means = np.array([np.mean(p53) for p53 in F3_p53])
F3_p53_stds = np.array([np.std(p53) for p53 in F3_p53])

A12_p53_means = np.array([np.mean(p53) for p53 in A12_p53])
A12_p53_stds = np.array([np.std(p53) for p53 in A12_p53])

F3_p53_WT_means = np.array([np.mean(p53) for p53 in F3_p53_WT])
F3_p53_WT_stds = np.array([np.std(p53) for p53 in F3_p53_WT])
F3_p53_KO_means = np.array([np.mean(p53) for p53 in F3_p53_KO])
F3_p53_KO_stds = np.array([np.std(p53) for p53 in F3_p53_KO])

fig, ax = plt.subplots()

ax.plot(range(10), F3_means, color=[0.0,0.8,0.0, 1.0], label="H2B-mCherry on F3")
ax.fill_between(range(10), F3_means - F3_stds, F3_means + F3_stds, color=[0.0,0.8,0.0], alpha=0.2)

ax.plot(range(10), A12_means, color=[0.8,0.0,0.8, 1.0], label="H2B-emiRFP on A12")
ax.fill_between(range(10), A12_means - A12_stds, A12_means + A12_stds, color=[0.8,0.0,0.8], alpha=0.2)

ax.plot(range(10), DAPI_means, color="cyan", label="DAPI on all cells")
ax.fill_between(range(10), DAPI_means - DAPI_stds, DAPI_means + DAPI_stds, color="cyan", alpha=0.2)

ax.set_xlabel("z")
ax.set_ylabel("fluoro [a.u.]")
ax.title("Mean with std ribbon")
ax.legend()
plt.show()

fig, ax = plt.subplots()
 
ax.plot(range(10), F3_p53_means, color=[0.0,0.8,0.0, 1.0], label="p53 on F3")
ax.fill_between(range(10), F3_p53_means - F3_p53_stds, F3_p53_means + F3_p53_stds, color=[0.0,0.8,0.0], alpha=0.2)

ax.plot(range(10), A12_p53_means, color=[0.8,0.0,0.8, 1.0], label="p53 on a12")
ax.fill_between(range(10), A12_p53_means - A12_p53_stds, A12_p53_means + A12_p53_stds, color=[0.8,0.0,0.8], alpha=0.2)

axt = ax.twinx()
axt.plot(range(10), DAPI_means, color="blue", label="DAPI on all cells")
axt.fill_between(range(10), DAPI_means - DAPI_stds, DAPI_means + DAPI_stds, color="blue", alpha=0.2)
ax.legend(loc=1)
axt.legend(loc=2)

plt.show()

fig, ax = plt.subplots()
 
ax.plot(range(10), F3_p53_means, color=[0.0,0.8,0.0, 1.0], label="p53 on F3")
# ax.fill_between(range(10), F3_p53_means - F3_p53_stds, F3_p53_means + F3_p53_stds, color=[0.0,0.8,0.0], alpha=0.2)

ax.plot(range(10), A12_p53_means, color=[0.8,0.0,0.8, 1.0], label="p53 on a12")
# ax.fill_between(range(10), A12_p53_means - A12_p53_stds, A12_p53_means + A12_p53_stds, color=[0.8,0.0,0.8], alpha=0.2)

ax.set_ylabel("mCherry and emiFRP [a.u.]")

axt = ax.twinx()
axt.plot(range(10), DAPI_means, color="blue", label="DAPI on all cells")
# axt.fill_between(range(10), DAPI_means - DAPI_stds, DAPI_means + DAPI_stds, color="blue", alpha=0.2)
axt.set_ylabel("DAPI [a.u.]")

ax.set_xlabel("z")

plt.legend()
plt.show()

fig, ax = plt.subplots()
 
ax.plot(range(10), F3_p53_WT_means, color=[0.0,0.8,0.0, 1.0], label="p53 on F3 with WT")
ax.fill_between(range(10), F3_p53_WT_means - F3_p53_WT_stds, F3_p53_WT_means + F3_p53_WT_stds, color=[0.0,0.8,0.0], alpha=0.2)

ax.plot(range(10), F3_p53_KO_means, color="cyan", label="p53 on F3 with KO")
ax.fill_between(range(10), F3_p53_KO_means - F3_p53_KO_stds, F3_p53_KO_means + F3_p53_KO_stds, color="cyan", alpha=0.2)
ax.set_xlabel("z")

plt.legend()
plt.show()


import numpy as np

def count_extremes(data, method="iqr", threshold=1.5, data_set_for_threshold=None):
    """
    Count extreme points in a 1D array.
    method = "iqr", "std", or "percentile"
    threshold:
       - iqr: multiplier (default 1.5)
       - std: number of stds (default 2)
       - percentile: cutoff (default 5 → outside 5th–95th)
    """
    
    data = np.asarray(data)
    if data_set_for_threshold is None:
        data_set_for_threshold=data
    else:
        data_set_for_threshold = np.asarray(data_set_for_threshold)
        
    if method == "iqr":
        q1, q3 = np.percentile(data_set_for_threshold, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
    elif method == "std":
        mu, sigma = np.mean(data_set_for_threshold), np.std(data_set_for_threshold)
        lower, upper = mu - threshold * sigma, mu + threshold * sigma
    elif method == "percentile":
        lower, upper = np.percentile(data_set_for_threshold, [threshold, 100-threshold])
    else:
        raise ValueError("Unknown method")
    
    mask = data > upper
    return np.sum(mask), len(data), mask

WT_extremes = []
KO_extremes = []

all_vals = [[*F3_p53_WT[z], *F3_p53_KO[z]] for z in range(10)]

iqr_outlier_threshold = 3.5

for z, arr in enumerate(F3_p53_WT):
    n_ext, n_total, mask = count_extremes(arr, method="iqr", threshold=iqr_outlier_threshold, data_set_for_threshold=all_vals[z])
    WT_extremes.append((z, n_ext, n_total))

for z, arr in enumerate(F3_p53_KO):
    n_ext, n_total, mask = count_extremes(arr, method="iqr", threshold=iqr_outlier_threshold, data_set_for_threshold=all_vals[z])
    KO_extremes.append((z, n_ext, n_total))

print("WT extremes:", WT_extremes)
print("KO extremes:", KO_extremes)

extreme_threshold = []
# Overlay individual points (WT)
z_vals = np.arange(len(F3_p53_WT))
for z in z_vals:

    q1, q3 = np.percentile(np.array(all_vals[z]), [25, 75])
    iqr = q3 - q1
    upper = q3 + iqr_outlier_threshold * iqr
    extreme_threshold.append(upper)


fig, ax = plt.subplots(figsize=(12,6))

offset = 0.42  # spacing for left/right
z_vals_plot = z_vals*1.3
# ---- A12 KO (dark-magenta, left) ----
vp1 = ax.violinplot(
    A12_p53_KO, positions=z_vals_plot - offset, widths=0.25,
    showmeans=False, showextrema=False, showmedians=False
)
for body in vp1['bodies']:
    body.set_facecolor((0.6,0.0,0.6,0.5))  # dark-magenta
    body.set_edgecolor('black')
    body.set_linewidth(0.5)
    
# ---- A12 WT (magenta, left-center) ----
vp1 = ax.violinplot(
    A12_p53_WT, positions=z_vals_plot - 0.33*offset, widths=0.25,
    showmeans=False, showextrema=False, showmedians=False
)
for body in vp1['bodies']:
    body.set_facecolor((0.8,0.0,0.8,0.5))  # magenta
    body.set_edgecolor('black')
    body.set_linewidth(0.5)

# ---- F3 WT (green, center) ----
vp2 = ax.violinplot(
    F3_p53_WT, positions=z_vals_plot + 0.33*offset, widths=0.25,
    showmeans=False, showextrema=False, showmedians=False
)
for body in vp2['bodies']:
    body.set_facecolor((0.0,0.8,0.0,0.5))  # green
    body.set_edgecolor('black')
    body.set_linewidth(0.5)

# ---- F3 KO (cyan, right) ----
vp3 = ax.violinplot(
    F3_p53_KO, positions=z_vals_plot + offset, widths=0.25,
    showmeans=False, showextrema=False, showmedians=False
)
for body in vp3['bodies']:
    body.set_facecolor((0,1,1,0.5))  # cyan
    body.set_edgecolor('black')
    body.set_linewidth(0.5)

# ---- Overlay individual points ----
for z, arr in zip(z_vals, A12_p53_KO):
    z_plot = z_vals_plot[z]
    ax.scatter(np.full_like(arr, z_plot - offset), arr,
               color=(0.6,0.0,0.6), alpha=0.7, s=5, zorder=3)
    q1, q3 = np.percentile(np.array(all_vals[z]), [25, 75])
    iqr = q3 - q1
    upper = q3 + iqr_outlier_threshold * iqr

    ext_mask = arr > upper
    
    ax.scatter(np.full_like(arr, z_plot - offset)[ext_mask], np.array(arr)[ext_mask],
           color="red", marker="x", s=30, zorder=4)
    
for z, arr in zip(z_vals, A12_p53_WT):
    z_plot = z_vals_plot[z]
    ax.scatter(np.full_like(arr, z_plot - 0.33*offset), arr,
               color=(0.8,0.0,0.8), alpha=0.7, s=5, zorder=3)
    q1, q3 = np.percentile(np.array(all_vals[z]), [25, 75])
    iqr = q3 - q1
    upper = q3 + iqr_outlier_threshold * iqr

    ext_mask = arr > upper
    ax.scatter(np.full_like(arr, z_plot - 0.33*offset)[ext_mask], np.array(arr)[ext_mask],
           color="red", marker="x", s=30, zorder=4)
    
for z, arr in zip(z_vals, F3_p53_WT):
    z_plot = z_vals_plot[z]
    ax.scatter(np.full_like(arr, z_plot + 0.33*offset), arr,
               color=(0.0,0.8,0.0), alpha=0.7, s=5, zorder=3)
    q1, q3 = np.percentile(np.array(all_vals[z]), [25, 75])
    iqr = q3 - q1
    upper = q3 + iqr_outlier_threshold * iqr

    ext_mask = arr > upper
    ax.scatter(np.full_like(arr, z_plot + 0.33*offset)[ext_mask], np.array(arr)[ext_mask],
           color="red", marker="x", s=30, zorder=4)
    
for z, arr in zip(z_vals, F3_p53_KO):
    z_plot = z_vals_plot[z]
    ax.scatter(np.full_like(arr, z_plot + offset), arr,
               color='cyan', alpha=0.7, s=5, zorder=3)
    q1, q3 = np.percentile(np.array(all_vals[z]), [25, 75])
    iqr = q3 - q1
    upper = q3 + iqr_outlier_threshold * iqr

    ext_mask = arr > upper
    ax.scatter(np.full_like(arr, z_plot + offset)[ext_mask], np.array(arr)[ext_mask],
           color="red", marker="x", s=30, zorder=4)
    
# ---- Axis + legend ----
ax.set_xlabel("z")
ax.set_ylabel("p53")
ax.set_xticks(z_vals_plot)
ax.set_title("p53 distributions across z")
ax.set_xticklabels(z_vals)
legend_elems = [
    plt.Line2D([0],[0], color=(0.6,0.0,0.6), lw=4, label="A12-KO"),
    plt.Line2D([0],[0], color=(0.8,0.0,0.8), lw=4, label="A12-WT"),
    plt.Line2D([0],[0], color=(0.0,0.8,0.0), lw=4, label="F3 with WT"),
    plt.Line2D([0],[0], color='cyan', lw=4, label="F3 with KO"),
]
ax.legend(handles=legend_elems, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53/p53distviolin.svg")
plt.show()



fig, ax = plt.subplots(figsize=(10,5))

offset = 0.15  # how far to separate WT and KO violins

# Plot WT violins
vp1 = ax.violinplot(
    F3_p53_WT, positions=z_vals - offset, widths=0.25,
    showmeans=False, showextrema=False, showmedians=False
)
for body in vp1['bodies']:
    body.set_facecolor((0.0,0.8,0.0,0.5))
    body.set_edgecolor('black')
    body.set_linewidth(0.5)

# Plot KO violins
vp2 = ax.violinplot(
    F3_p53_KO, positions=z_vals + offset, widths=0.25,
    showmeans=False, showextrema=False, showmedians=False
)
for body in vp2['bodies']:
    body.set_facecolor((0,1,1,0.5))  # cyan
    body.set_edgecolor('black')
    body.set_linewidth(0.5)

# Overlay individual points (WT)
for z, arr in zip(z_vals, F3_p53_WT):
    ax.scatter(np.full_like(arr, z - offset), arr,
               color=(0.0,0.8,0.0), alpha=0.7, s=10, zorder=3)
    q1, q3 = np.percentile(np.array(all_vals[z]), [25, 75])
    iqr = q3 - q1
    upper = q3 + iqr_outlier_threshold * iqr

    ext_mask = arr > upper
    ax.scatter(np.full_like(arr, z - offset)[ext_mask], np.array(arr)[ext_mask],
           color="red", marker="x", s=30, zorder=4)


# Overlay individual points (KO)
for z, arr in zip(z_vals, F3_p53_KO):
    ax.scatter(np.full_like(arr, z + offset), arr,
               color='cyan', alpha=0.7, s=10, zorder=3)
    q1, q3 = np.percentile(np.array(all_vals[z]), [25, 75])
    iqr = q3 - q1
    lower = q1 - iqr_outlier_threshold * iqr
    upper = q3 + iqr_outlier_threshold * iqr

    ext_mask = arr > upper
    ax.scatter(np.full_like(arr, z + offset)[ext_mask], np.array(arr)[ext_mask],
           color="red", marker="x", s=30, zorder=4)

ax.set_xlabel("z")
ax.set_ylabel("p53 on F3")
ax.set_xticks(z_vals)
ax.set_title("WT vs KO distributions per z")

# Custom legend
legend_elems = [
    plt.Line2D([0],[0], color=(0.0,0.8,0.0), lw=4, label="WT"),
    plt.Line2D([0],[0], color='cyan', lw=4, label="KO")
]
ax.legend(handles=legend_elems, frameon=False)

plt.tight_layout()
plt.show()


WT_extremes_ratio = np.array([(WT_extremes[i][1]/WT_extremes[i][2])*100 for i in range(10)])
KO_extremes_ratio = np.array([(KO_extremes[i][1]/KO_extremes[i][2])*100 for i in range(10)])

fig, ax = plt.subplots()

ax.plot(range(10), WT_extremes_ratio, color=[0.0,0.8,0.0, 1.0], label="F3 with WT")
ax.plot(range(10), KO_extremes_ratio, color="cyan", label="F3 with KO")

ax.set_xlabel("z")
ax.set_ylabel("% of p53-high cells")
ax.title("Mean with std ribbon")
ax.legend()
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(26,6))

ax[0].plot(range(10), F3_p53_WT_means, color=[0.0,0.8,0.0, 1.0], label="p53 on F3 with WT")
ax[0].fill_between(range(10), F3_p53_WT_means - F3_p53_WT_stds, F3_p53_WT_means + F3_p53_WT_stds, color=[0.0,0.8,0.0], alpha=0.2)

ax[0].plot(range(10), F3_p53_KO_means, color="cyan", label="p53 on F3 with KO")
ax[0].fill_between(range(10), F3_p53_KO_means - F3_p53_KO_stds, F3_p53_KO_means + F3_p53_KO_stds, color="cyan", alpha=0.2)
ax[0].set_xlabel("z")
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_ylabel("p53 [a.u.]")
ax[0].set_title("p53 over z in WT vs KO")
ax[0].legend(frameon=False)
offset = 0.15  # how far to separate WT and KO violins

# Plot WT violins
vp1 = ax[1].violinplot(
    F3_p53_WT, positions=z_vals - offset, widths=0.25,
    showmeans=False, showextrema=False, showmedians=False
)
for body in vp1['bodies']:
    body.set_facecolor((0.0,0.8,0.0,0.5))
    body.set_edgecolor('black')
    body.set_linewidth(0.5)

# Plot KO violins
vp2 = ax[1].violinplot(
    F3_p53_KO, positions=z_vals + offset, widths=0.25,
    showmeans=False, showextrema=False, showmedians=False
)
for body in vp2['bodies']:
    body.set_facecolor((0,1,1,0.5))  # cyan
    body.set_edgecolor('black')
    body.set_linewidth(0.5)

# Overlay individual points (WT)
for z, arr in zip(z_vals, F3_p53_WT):
    ax[1].scatter(np.full_like(arr, z - offset), arr,
               color=(0.0,0.8,0.0), alpha=0.7, s=10, zorder=3)
    q1, q3 = np.percentile(np.array(all_vals[z]), [25, 75])
    iqr = q3 - q1
    upper = q3 + iqr_outlier_threshold * iqr

    ext_mask = arr > upper
    ax[1].scatter(np.full_like(arr, z - offset)[ext_mask], np.array(arr)[ext_mask],
           color="red", marker="x", s=30, zorder=4)


# Overlay individual points (KO)
for z, arr in zip(z_vals, F3_p53_KO):
    ax[1].scatter(np.full_like(arr, z + offset), arr,
               color='cyan', alpha=0.7, s=10, zorder=3)
    q1, q3 = np.percentile(np.array(all_vals[z]), [25, 75])
    iqr = q3 - q1
    lower = q1 - iqr_outlier_threshold * iqr
    upper = q3 + iqr_outlier_threshold * iqr

    ext_mask = arr > upper
    ax[1].scatter(np.full_like(arr, z + offset)[ext_mask], np.array(arr)[ext_mask],
           color="red", marker="x", s=30, zorder=4, label="p53-high cells")

ax[1].set_xlabel("z")
ax[1].set_ylabel("p53 [a.u.]")
ax[1].set_xticks(z_vals)
ax[1].set_title("p53 distribution over z")

# Custom legend
legend_elems = [
    plt.Line2D([0], [0], color=(0.0,0.8,0.0), lw=4, label="F3 with WT"),
    plt.Line2D([0], [0], color='cyan', lw=4, label="F3 with KO"),
    plt.Line2D([0], [0], color='red', marker='x', linestyle="None", 
               markersize=10, markeredgewidth=2, label="p53-high cells")
]

ax[1].legend(handles=legend_elems, loc="best", frameon=False)

ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

ax[2].plot(range(10), WT_extremes_ratio, color=[0.0,0.8,0.0, 1.0], label="F3 with WT")
ax[2].plot(range(10), KO_extremes_ratio, color="cyan", label="F3 with KO")

ax[2].set_xlabel("z")
ax[2].set_ylabel(r"\% of p53-high cells")
ax[2].set_title("p53 high values over z")
ax[2].legend(frameon=False)
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53/p53overz.svg")
plt.show()

import pandas as pd

# Build dictionary: one column per violin
data_dict = {}

for z, arr in enumerate(A12_p53_KO):
    arr = A12_p53_KO[z]
    data_dict[f"A12-KO_z{z}"] = pd.Series(arr)
    arr = A12_p53_WT[z]
    data_dict[f"A12-WT_z{z}"] = pd.Series(arr)
    arr = F3_p53_WT[z]
    data_dict[f"F3-WT_z{z}"] = pd.Series(arr)
    arr = F3_p53_KO[z]
    data_dict[f"F3-KO_z{z}"] = pd.Series(arr)

# Convert to DataFrame (NaN fills uneven lengths)
df = pd.DataFrame(data_dict)

# Save to CSV
path_save = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53/"
df.to_csv(path_save+"violin_data.csv", index=False)