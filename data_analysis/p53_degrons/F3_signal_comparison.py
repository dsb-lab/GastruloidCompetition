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

CONDS = ["auxin48", "auxin72", "noauxin72"]

for C, COND in enumerate(CONDS):
    path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
    path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
        
    check_or_create_dir(path_save_dir)

    files = get_file_names(path_data_dir)

    channel_names = ["A12", "p53", "F3", "DAPI"]

    F3.append([])

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
  
        ch_F3 = channel_names.index("F3")

        f3 = []
        for cell in CT_F3.jitcells:
            center = cell.centers[0]
            z = int(center[0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]
            f3.append(np.mean(CT_F3.hyperstack[0,z,ch_F3,:,:][mask[:,1], mask[:,0]]))

        F3[-1].append(f3)


F3_all = []
for C, COND in enumerate(CONDS):
    f3 = []
    for g in range(len(F3[C])):
        f3 = [*f3, *F3[C][g]]
    F3_all.append(f3)
    
F3_means = np.array([np.mean(f3) for f3 in F3_all])
F3_stds = np.array([np.std(f3) for f3 in F3_all])

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



def plot_gastruloid_bars(CONDS, F3, ylabel="Mean fluorescence (per gastruloid)"):
    # Colors you’ve been using
    color_F3  = (0.0, 0.8, 0.0, 0.7)  # green

    # Compute per-gastruloid means, then summary stats per condition
    F3_gmeans  = gastruloid_means(F3)
    F3_mean, F3_std   = summarize(F3_gmeans)

    # X positions: one group per condition; two bars per group
    x = np.arange(len(CONDS))
    width = 0.38
    off = width/2

    fig, ax = plt.subplots(figsize=(8, 5))
    # Bars
    b_F3  = ax.bar(x - off, F3_mean, width, yerr=F3_std, capsize=5, label="F3",  color=color_F3,  edgecolor="black", linewidth=0.7)

    # Overlay the individual gastruloid means as jittered dots
    rng = np.random.default_rng(42)
    jitter_scale = width * 0.30
    for i in range(len(CONDS)):
        # F3 dots
        xs = (x[i] - off) + rng.uniform(-jitter_scale, jitter_scale, size=len(F3_gmeans[i]))
        ax.scatter(xs, F3_gmeans[i], s=35, marker="o", facecolors="none", edgecolors=color_F3, linewidths=1.2)

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


def plot_gastruloid_bars_all(CONDS, F3_all):
    # Colors you’ve been using
    color_F3  = (0.0, 0.8, 0.0, 0.7)  # green

    F3_mean = np.array([np.mean(f3) for f3 in F3_all])
    F3_std = np.array([np.std(f3) for f3 in F3_all])


    # X positions: one group per condition; two bars per group
    x = np.arange(len(CONDS))
    width = 0.38
    off = width/2

    fig, ax = plt.subplots(figsize=(8, 5))
    # Bars
    b_F3  = ax.bar(x - off, F3_mean, width, yerr=F3_std, capsize=5, label="F3",  color=color_F3,  edgecolor="black", linewidth=0.7)

    # Overlay the individual gastruloid means as jittered dots
    rng = np.random.default_rng(42)
    jitter_scale = width * 0.30
    for i in range(len(CONDS)):
        # F3 dots
        xs = (x[i] - off) + rng.uniform(-jitter_scale, jitter_scale, size=len(F3_all[i]))
        ax.scatter(xs, F3_all[i], s=35, marker="o", facecolors="none", edgecolors=color_F3, linewidths=1.2)

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

fig, ax = plot_gastruloid_bars(CONDS, F3, ylabel="Mean fluorescence (a.u.)")
plt.show()

fig, ax = plot_gastruloid_bars_all(CONDS, F3_all)
plt.show()
