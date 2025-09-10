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

A12 = []

CONDS = ["auxin48", "auxin72", "noauxin72"]

for C, COND in enumerate(CONDS):
    path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
    path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
        
    check_or_create_dir(path_save_dir)

    files = get_file_names(path_data_dir)

    channel_names = ["A12", "p53", "F3", "DAPI"]

    A12.append([])

    for f, file in enumerate(files):
                
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
  
        ch_A12 = channel_names.index("A12")

        a12 = []
        for cell in CT_A12.jitcells:
            center = cell.centers[0]
            z = int(center[0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]
            a12.append(np.mean(CT_A12.hyperstack[0,z,ch_A12,:,:][mask[:,1], mask[:,0]]))

        A12[-1].append(a12)


A12_all = []
for C, COND in enumerate(CONDS):
    a12 = []
    for g in range(len(A12[C])):
        a12 = [*a12, *A12[C][g]]
    A12_all.append(a12)
    
A12_means = np.array([np.mean(a12) for a12 in A12_all])
A12_stds = np.array([np.std(a12) for a12 in A12_all])

import numpy as np
import matplotlib.pyplot as plt

def gastruloid_means(per_condition_lists):
    """Convert list-of-lists-of-values into per-gastruloid means per condition."""
    per_cond_g_means = []
    for cond_list in per_condition_lists:
        # cond_list: list of gastruloids; each gastruloid can be a scalar or array-like
        g_means = [np.mean(np.ravel(g)) for g in cond_list]
        per_cond_g_means.append(g_means)
    return per_cond_g_means

def summarize(per_cond_g_means):
    """Return mean and std across gastruloids for each condition."""
    means = [float(np.mean(gm)) if len(gm)>0 else np.nan for gm in per_cond_g_means]
    # Use sample std (ddof=1) when >=2 gastruloids, else 0
    stds  = [float(np.std(gm, ddof=1)) if len(gm) >= 2 else 0.0 for gm in per_cond_g_means]
    return np.array(means), np.array(stds)



def plot_gastruloid_bars(CONDS, A12, ylabel="Mean fluorescence (per gastruloid)"):
    # Colors you’ve been using
    color_A12  = (0.8, 0.0, 0.8, 0.7)  # magenta

    # Compute per-gastruloid means, then summary stats per condition
    A12_gmeans  = gastruloid_means(A12)
    A12_mean, A12_std   = summarize(A12_gmeans)

    # X positions: one group per condition; two bars per group
    x = np.arange(len(CONDS))
    width = 0.38
    off = width/2

    fig, ax = plt.subplots(figsize=(8, 5))
    # Bars
    b_A12  = ax.bar(x - off, A12_mean, width, yerr=A12_std, capsize=5, label="A12",  color=color_A12,  edgecolor="black", linewidth=0.7)

    # Overlay the individual gastruloid means as jittered dots
    rng = np.random.default_rng(42)
    jitter_scale = width * 0.30
    for i in range(len(CONDS)):
        # A12 dots
        xs = (x[i] - off) + rng.uniform(-jitter_scale, jitter_scale, size=len(A12_gmeans[i]))
        ax.scatter(xs, A12_gmeans[i], s=35, marker="o", facecolors="none", edgecolors=color_A12, linewidths=1.2)

    # Cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(CONDS, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title("Per-condition fluorescence (across gastruloids)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.1)

    plt.tight_layout()
    return fig, ax


def plot_gastruloid_bars_all(CONDS, A12_all):
    # Colors you’ve been using
    color_A12  = (0.8, 0.0, 0.8, 0.7)  # magenta

    A12_mean = np.array([np.mean(a12) for a12 in A12_all])
    A12_std = np.array([np.std(a12) for a12 in A12_all])


    # X positions: one group per condition; two bars per group
    x = np.arange(len(CONDS))
    width = 0.38
    off = width/2

    fig, ax = plt.subplots(figsize=(8, 5))
    # Bars
    b_A12  = ax.bar(x - off, A12_mean, width, yerr=A12_std, capsize=5, label="A12",  color=color_A12,  edgecolor="black", linewidth=0.7)

    # Overlay the individual gastruloid means as jittered dots
    rng = np.random.default_rng(42)
    jitter_scale = width * 0.30
    for i in range(len(CONDS)):
        # A12 dots
        xs = (x[i] - off) + rng.uniform(-jitter_scale, jitter_scale, size=len(A12_all[i]))
        ax.scatter(xs, A12_all[i], s=35, marker="o", facecolors="none", edgecolors=color_A12, linewidths=1.2)

    # Cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(CONDS, rotation=0)
    ax.set_ylabel("mean fluorescence (per cell)")
    ax.set_title("Per-condition fluorescence (merged samples)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.1)

    plt.tight_layout()
    return fig, ax

fig, ax = plot_gastruloid_bars(CONDS, A12, ylabel="Mean fluorescence (a.u.)")
plt.show()

fig, ax = plot_gastruloid_bars_all(CONDS, A12_all)
plt.show()
