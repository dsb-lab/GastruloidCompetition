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

CONDS = ["auxin_48-72_48", "auxin_48-72_72" , "auxin_48-72_96", "auxin_72-96_72", "auxin_72-96_96", "noauxin_72", "noauxin_96"]

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

for C, COND in enumerate(CONDS):
    path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
    path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
        
    check_or_create_dir(path_save_dir)

    files = get_file_names(path_data_dir)

    channel_names = ["A12", "p53", "F3", "DAPI"]

    F3.append([])
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
  
        ch_p53 = channel_names.index("p53")
        ch_F3 = channel_names.index("F3")
        ch_A12 = channel_names.index("A12")

        Mz_list = build_union_masks([CT_F3])
        p53_F3_0z = estimate_b0z_for_file(CT_F3, Mz_list, ch_F3, ch_p53, p53_F3_s_global)
        
        f3 = []
        for cell in CT_F3.jitcells:
            z = int(cell.centers[0][0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]
            Ccorr_vals = correct_cell_pixels(CT_F3, mask, z, ch_F3, ch_p53, p53_F3_s_global, p53_F3_0z)
            f3.append(float(np.mean(Ccorr_vals)))
            # f3.append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]]))
            
        Mz_list = build_union_masks([CT_A12])
        p53_A12_0z = estimate_b0z_for_file(CT_A12, Mz_list, ch_A12, ch_p53, p53_A12_s_global)
        
        a12 = []
        for cell in CT_A12.jitcells:
            z = int(cell.centers[0][0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]
            Ccorr_vals = correct_cell_pixels(CT_A12, mask, z, ch_A12, ch_p53, p53_A12_s_global, p53_A12_0z)
            a12.append(float(np.mean(Ccorr_vals)))
            # a12.append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]]))
        F3[-1].append(f3)
        A12[-1].append(a12)    


F3_all = []
A12_all = []
for C, COND in enumerate(CONDS):
    f3 = []
    a12 = []
    for g in range(len(F3[C])):
        f3 = [*f3, *F3[C][g]]
        a12 = [*a12, *A12[C][g]]
    F3_all.append(f3)
    A12_all.append(a12)
    

F3_means = np.array([np.mean(f3) for f3 in F3_all])
F3_stds = np.array([np.std(f3) for f3 in F3_all])

A12_means = np.array([np.mean(a12) for a12 in A12_all])
A12_stds = np.array([np.std(a12) for a12 in A12_all])


import numpy as np
import matplotlib.pyplot as plt

# Example usage:
# CONDS = ["auxin48", "auxin72", "noauxin72"]
# F3 = [ [arr_g1, arr_g2, arr_g3],  [arr_g1, arr_g2, arr_g3],  [arr_g1, arr_g2, arr_g3] ]
# A12 = [ [arr_g1, arr_g2, arr_g3], [arr_g1, arr_g2, arr_g3], [arr_g1, arr_g2, arr_g3] ]
# where each arr_gX is a 1D array/list of fluorescence values for that gastruloid
# (If you already have a single number per gastruloid, it works the same.)

def gastruloid_means(per_condition_lists):
    """Convert list-of-lists-of-values into per-gastruloid means per condition."""
    per_cond_g_means = []
    for cond_list in per_condition_lists:
        # cond_list: list of gastruloids; each gastruloid can be a scalar or array-like
        g_means = [np.median(np.ravel(g)) for g in cond_list]
        per_cond_g_means.append(g_means)
    return per_cond_g_means

def summarize(per_cond_g_means):
    """Return mean and std across gastruloids for each condition."""
    means = [float(np.median(gm)) if len(gm)>0 else np.nan for gm in per_cond_g_means]
    # Use sample std (ddof=1) when >=2 gastruloids, else 0
    stds  = [float(np.std(gm, ddof=1)) if len(gm) >= 2 else 0.0 for gm in per_cond_g_means]
    return np.array(means), np.array(stds)



def plot_gastruloid_bars(CONDS, F3, A12, ylabel="Mean fluorescence (per gastruloid)"):
    # Colors you’ve been using
    color_A12 = (0.8, 0.0, 0.8, 0.7)  # purple-ish
    color_F3  = (0.0, 0.8, 0.0, 0.7)  # green

    # Compute per-gastruloid means, then summary stats per condition
    F3_gmeans  = gastruloid_means(F3)
    A12_gmeans = gastruloid_means(A12)
    F3_mean, F3_std   = summarize(F3_gmeans)
    A12_mean, A12_std = summarize(A12_gmeans)

    # X positions: one group per condition; two bars per group
    x = np.arange(len(CONDS))
    width = 0.38
    off = width/2

    fig, ax = plt.subplots(figsize=(10, 6))
    # Bars
    b_F3  = ax.bar(x - off, F3_mean, width, yerr=F3_std, capsize=5, label="F3",  color=color_F3,  edgecolor="black", linewidth=0.7)
    b_A12 = ax.bar(x + off, A12_mean, width, yerr=A12_std, capsize=5, label="A12", color=color_A12, edgecolor="black", linewidth=0.7)

    # Overlay the individual gastruloid means as jittered dots
    rng = np.random.default_rng(42)
    jitter_scale = width * 0.30
    for i in range(len(CONDS)):
        # F3 dots
        xs = (x[i] - off) + rng.uniform(-jitter_scale, jitter_scale, size=len(F3_gmeans[i]))
        ax.scatter(xs, F3_gmeans[i], s=35, marker="o", facecolors="none", edgecolors=color_F3, linewidths=1.2)
        # A12 dots
        xs = (x[i] + off) + rng.uniform(-jitter_scale, jitter_scale, size=len(A12_gmeans[i]))
        ax.scatter(xs, A12_gmeans[i], s=35, marker="o", facecolors="none", edgecolors=color_A12, linewidths=1.2)

    # Cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(CONDS, rotation=-45, ha="left")
    ax.set_ylabel(ylabel)
    ax.set_title("Per-condition fluorescence (across gastruloids)")
    ax.legend(frameon=False, loc=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.1)

    plt.tight_layout()
    return fig, ax


def plot_gastruloid_bars_all(CONDS, F3_all, A12_all):
    # Colors you’ve been using
    color_A12 = (0.8, 0.0, 0.8, 0.7)  # purple-ish
    color_F3  = (0.0, 0.8, 0.0, 0.7)  # green

    F3_mean = np.array([np.mean(f3) for f3 in F3_all])
    F3_std = np.array([np.std(f3) for f3 in F3_all])

    A12_mean = np.array([np.mean(a12) for a12 in A12_all])
    A12_std = np.array([np.std(a12) for a12 in A12_all])

    # X positions: one group per condition; two bars per group
    x = np.arange(len(CONDS))
    width = 0.38
    off = width/2

    fig, ax = plt.subplots(figsize=(10, 6))
    # Bars
    b_F3  = ax.bar(x - off, F3_mean, width, yerr=F3_std, capsize=5, label="F3",  color=color_F3,  edgecolor="black", linewidth=0.7)
    b_A12 = ax.bar(x + off, A12_mean, width, yerr=A12_std, capsize=5, label="A12", color=color_A12, edgecolor="black", linewidth=0.7)

    # Overlay the individual gastruloid means as jittered dots
    rng = np.random.default_rng(42)
    jitter_scale = width * 0.30
    for i in range(len(CONDS)):
        # F3 dots
        xs = (x[i] - off) + rng.uniform(-jitter_scale, jitter_scale, size=len(F3_all[i]))
        ax.scatter(xs, F3_all[i], s=35, marker="o", facecolors="none", edgecolors=color_F3, linewidths=1.2)
        # A12 dots
        xs = (x[i] + off) + rng.uniform(-jitter_scale, jitter_scale, size=len(A12_all[i]))
        ax.scatter(xs, A12_all[i], s=35, marker="o", facecolors="none", edgecolors=color_A12, linewidths=1.2)

    # Cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(CONDS, rotation=-45, ha="left")
    ax.set_ylabel("mean fluorescence (per cell)")
    ax.set_title("Per-condition fluorescence (merged samples)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.1)
    plt.tight_layout()
    return fig, ax



def plot_gastruloid_bars_A12(CONDS, A12, ylabel="Mean fluorescence (per gastruloid)"):
    # Colors you’ve been using
    color_A12 = (0.8, 0.0, 0.8, 0.7)  # purple-ish

    # Compute per-gastruloid means, then summary stats per condition
    A12_gmeans = gastruloid_means(A12)
    A12_mean, A12_std = summarize(A12_gmeans)

    # X positions: one group per condition; two bars per group
    x = np.arange(len(CONDS))
    width = 0.38*2
    off = 0

    fig, ax = plt.subplots(figsize=(10, 6))
    # Bars
    b_A12 = ax.bar(x + off, A12_mean, width, yerr=A12_std, capsize=5, label="A12", color=color_A12, edgecolor="black", linewidth=0.7)

    # Overlay the individual gastruloid means as jittered dots
    rng = np.random.default_rng(42)
    jitter_scale = width * 0.30
    for i in range(len(CONDS)):
        # A12 dots
        xs = (x[i] + off) + rng.uniform(-jitter_scale, jitter_scale, size=len(A12_gmeans[i]))
        ax.scatter(xs, A12_gmeans[i], s=35, marker="o", facecolors="none", edgecolors=color_A12, linewidths=1.2)

    # Cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(CONDS, rotation=-45, ha="left")
    ax.set_ylabel(ylabel)
    ax.set_title("Per-condition fluorescence (across gastruloids)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.1)

    plt.tight_layout()
    return fig, ax



fig, ax = plot_gastruloid_bars(CONDS, F3, A12, ylabel="Mean fluorescence (a.u.)")
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53quant_all.pdf")
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53quant_all.svg")
plt.show()

fig, ax = plot_gastruloid_bars_all(CONDS, F3_all, A12_all)
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53quant_alldatapoints.pdf")
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53quant_alldatapoints.svg")
plt.show()


# Colors you’ve been using
color_A12 = (0.8, 0.0, 0.8, 0.7)  # purple-ish

# Compute per-gastruloid means, then summary stats per condition
A12_gmeans = gastruloid_means(A12)
F3_gmeans = gastruloid_means(F3)

A12_mean, A12_std = summarize(A12_gmeans)
F3_mean, F3_std = summarize(F3_gmeans)


# Bar positions
labels = ["A12 - no auxin 72",
          "A12 - auxin(72-96) 72"
          "A12 - auxin(48-72) 72", 
          ]
comparisons = [(0,1), (0,2), (1,2)] 

labels = ["A12 - no auxin 48",
          "A12 - no auxin 72",
          "A12 - auxin(72-96) 72"
          "A12 - auxin(48-72) 72", 
          "A12 - auxin(48-72) 96",
          "A12 - no auxin 96",
          "A12 - auxin(72-96) 96", 
          ]
comparisons = [(0,1), (1,2), (1,3), (2,3), (3,4), (4,5), (4,6), (5,6)] 

selected_CONDS = [5, 3, 1]
selected_CONDS = [0, 5, 3, 1, 2, 6,4]

conds = [CONDS[i] for i in selected_CONDS]
A12_mean_sel = [A12_mean[i] for i in selected_CONDS]
A12_std_sel = [A12_std[i] for i in selected_CONDS]
A12_gmeans_sel = [A12_gmeans[i] for i in selected_CONDS]

F3_mean_sel = [F3_mean[i] for i in selected_CONDS]
F3_std_sel = [F3_std[i] for i in selected_CONDS]
F3_gmeans_sel = [F3_gmeans[i] for i in selected_CONDS]


# X positions: one group per condition; two bars per group
x = np.arange(len(conds))
width = 0.38*1.5
off = 0

fig, ax = plt.subplots(figsize=(10, 8))
# Bars
b_A12 = ax.bar(x + off, A12_mean_sel, width, yerr=A12_std_sel, capsize=5, label="A12", color=color_A12, edgecolor="black", linewidth=0.7)

# Overlay the individual gastruloid means as jittered dots
rng = np.random.default_rng(42)
jitter_scale = width * 0.30
for i in range(len(CONDS)):
    # A12 dots
    xs = (x[i] + off) + rng.uniform(-jitter_scale, jitter_scale, size=len(A12_std_sel[i]))
    ax.scatter(xs, A12_std_sel[i], s=35, marker="o", facecolors="none", edgecolors=color_A12, linewidths=1.2)

# Cosmetics
ax.set_xticks(x)
ax.set_xticklabels(conds, rotation=-45, ha="left")
ax.set_ylabel(" p53 [a. u.]")

# ax.set_ylabel(r"distance to merge point ($\mu$m)")
ax.set_title("Per-condition fluorescence (across gastruloids)")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.margins(y=0.1)

from scipy.stats import ttest_ind

data = [A12_all[i] for i in selected_CONDS]
data = A12_gmeans_sel

# Run tests and add bars
ymax = max(max(d) for d in data) * 1.01  # starting height for bars
h = 0.02 * ymax   # height of bars
step = 0
steps=0

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

    ax.text((x1+x2)/2, y+h, stars, ha='center', va='bottom', fontsize=17)

    step += 7  # space out multiple bars
    steps+=1
    if steps==4: step=0 

plt.tight_layout()
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53quant.pdf")
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53quant.svg")
plt.show()


# Colors you’ve been using
color_A12 = (0.8, 0.0, 0.8, 0.7)  # purple-ish

# corr_A12_gmeans = []
# for i in range(len(A12_gmeans)):
#     gm = []
#     for j in range(len(A12_gmeans[i])):
#        gm.append(np.abs(A12_gmeans[i][j] - F3_gmeans[i][j]) )
#     corr_A12_gmeans.append(gm)

corr_A12_gmeans = []
for i in range(len(A12_gmeans)):
    gm = []
    for j in range(len(A12_gmeans[i])):
       gm.append(A12_gmeans[i][j] / F3_gmeans[i][j]) 
    corr_A12_gmeans.append(gm)


A12_mean, A12_std = summarize(corr_A12_gmeans)

A12_mean_sel = [A12_mean[i] for i in selected_CONDS]
A12_std_sel = [A12_std[i] for i in selected_CONDS]
A12_gmeans_sel = [corr_A12_gmeans[i] for i in selected_CONDS]

# X positions: one group per condition; two bars per group
x = np.arange(len(conds))
width = 0.38*1.5
off = 0

fig, ax = plt.subplots(figsize=(10, 8))
# Bars
b_A12 = ax.bar(x + off, A12_mean_sel, width, yerr=A12_std_sel, capsize=5, label="A12", color=color_A12, edgecolor="black", linewidth=0.7)

# Overlay the individual gastruloid means as jittered dots
rng = np.random.default_rng(42)
jitter_scale = width * 0.30
for i in range(len(CONDS)):
    # A12 dots
    xs = (x[i] + off) + rng.uniform(-jitter_scale, jitter_scale, size=len(A12_std_sel[i]))
    ax.scatter(xs, A12_std_sel[i], s=35, marker="o", facecolors="none", edgecolors=color_A12, linewidths=1.2)

# Cosmetics
ax.set_xticks(x)
ax.set_xticklabels(conds, rotation=-45, ha="left")
# ax.set_ylabel(r"$\delta$ p53")
# ax.set_ylabel(r"$|${\unsansmath$\Delta$} p53$|$ [a.u.]")
ax.set_ylabel(r"p53 ratio")

# ax.set_ylabel(r"{\unsansmath$\Delta$} p53 [a.u.]")
# ax.set_ylabel(" p53 [a. u.]")

# ax.set_ylabel(r"distance to merge point ($\mu$m)")
ax.set_title("Per-condition fluorescence (across gastruloids)")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.margins(y=0.1)

from scipy.stats import ttest_ind

data = [A12_all[i] for i in selected_CONDS]
data = A12_gmeans_sel

# Run tests and add bars
ymax = max(max(d) for d in data) * 1.01  # starting height for bars
h = 0.02 * ymax   # height of bars
step = 0
steps = 0
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

    ax.text((x1+x2)/2, y+h, stars, ha='center', va='bottom', fontsize=25)

    step += 7  # space out multiple bars
    steps+=1
    if steps==4: step=0 
plt.tight_layout()
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53ratio.pdf")
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53ratio.svg")
# plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53absolutediff.pdf")
# plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53absolutediff.svg")
plt.show()


# Colors you’ve been using
color_A12 = (0.8, 0.0, 0.8, 0.7)  # purple-ish

# corr_A12_gmeans = []
# for i in range(len(A12_gmeans)):
#     gm = []
#     for j in range(len(A12_gmeans[i])):
#        gm.append(np.abs(A12_gmeans[i][j] - F3_gmeans[i][j]) )
#     corr_A12_gmeans.append(gm)

corr_A12_gmeans = []
for i in range(len(A12_gmeans)):
    gm = []
    for j in range(len(A12_gmeans[i])):
       gm.append(np.log(A12_gmeans[i][j]+1) / np.log(F3_gmeans[i][j]+1)) 
    corr_A12_gmeans.append(gm)

A12_mean, A12_std = summarize(corr_A12_gmeans)

A12_mean_sel = [A12_mean[i] for i in selected_CONDS]
A12_std_sel = [A12_std[i] for i in selected_CONDS]
A12_gmeans_sel = [corr_A12_gmeans[i] for i in selected_CONDS]

# X positions: one group per condition; two bars per group
x = np.arange(len(conds))
width = 0.38*1.5
off = 0

fig, ax = plt.subplots(figsize=(10, 8))
# Bars
b_A12 = ax.bar(x + off, A12_mean_sel, width, yerr=A12_std_sel, capsize=5, label="A12", color=color_A12, edgecolor="black", linewidth=0.7)

# Overlay the individual gastruloid means as jittered dots
rng = np.random.default_rng(42)
jitter_scale = width * 0.30
for i in range(len(CONDS)):
    # A12 dots
    xs = (x[i] + off) + rng.uniform(-jitter_scale, jitter_scale, size=len(A12_std_sel[i]))
    ax.scatter(xs, A12_std_sel[i], s=35, marker="o", facecolors="none", edgecolors=color_A12, linewidths=1.2)

# Cosmetics
ax.set_xticks(x)
ax.set_xticklabels(conds, rotation=-45, ha="left")
# ax.set_ylabel(r"$\delta$ p53")
# ax.set_ylabel(r"$|${\unsansmath$\Delta$} p53$|$ [a.u.]")
ax.set_ylabel(r"p53 log-ratio")

# ax.set_ylabel(r"{\unsansmath$\Delta$} p53 [a.u.]")
# ax.set_ylabel(" p53 [a. u.]")

# ax.set_ylabel(r"distance to merge point ($\mu$m)")
ax.set_title("Per-condition fluorescence (across gastruloids)")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.margins(y=0.1)

from scipy.stats import ttest_ind

data = [A12_all[i] for i in selected_CONDS]
data = A12_gmeans_sel

# Run tests and add bars
ymax = max(max(d) for d in data) * 1.01  # starting height for bars
h = 0.02 * ymax   # height of bars
step = 0
steps = 0
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

    ax.text((x1+x2)/2, y+h, stars, ha='center', va='bottom', fontsize=25)

    step += 2  # space out multiple bars
    steps+=1
    if steps==4: step=0 
ax.set_ylim(0.75,1.25)
plt.tight_layout()
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53logratio.pdf")
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53logratio.svg")
plt.show()

# Colors you’ve been using
color_A12 = (0.8, 0.0, 0.8, 0.7)  # purple-ish

corr_A12_gmeans = []
for i in range(len(A12_gmeans)):
    gm = []
    for j in range(len(A12_gmeans[i])):
       gm.append(A12_gmeans[i][j] / F3_gmeans[i][j]) 
    corr_A12_gmeans.append(gm)


A12_mean, A12_std = summarize(corr_A12_gmeans)

A12_mean_sel = [A12_mean[i] for i in selected_CONDS]
A12_std_sel = [A12_std[i] for i in selected_CONDS]
A12_gmeans_sel = [corr_A12_gmeans[i] for i in selected_CONDS]

# X positions: one group per condition; two bars per group
x = np.arange(len(conds))
width = 0.38*1.5
off = 0

fig, ax = plt.subplots(figsize=(10, 8))
# Bars
b_A12 = ax.bar(x + off, A12_mean_sel, width, yerr=A12_std_sel, capsize=5, label="A12", color=color_A12, edgecolor="black", linewidth=0.7)

# Overlay the individual gastruloid means as jittered dots
rng = np.random.default_rng(42)
jitter_scale = width * 0.30
for i in range(len(CONDS)):
    # A12 dots
    xs = (x[i] + off) + rng.uniform(-jitter_scale, jitter_scale, size=len(A12_std_sel[i]))
    ax.scatter(xs, A12_std_sel[i], s=35, marker="o", facecolors="none", edgecolors=color_A12, linewidths=1.2)

# Cosmetics
ax.set_xticks(x)
ax.set_xticklabels(conds, rotation=-45, ha="left")
# ax.set_ylabel(r"$\delta$ p53")
# ax.set_ylabel(r"$|${\unsansmath$\Delta$} p53$|$ [a.u.]")
ax.set_ylabel(r"p53 ratio")

# ax.set_ylabel(r"{\unsansmath$\Delta$} p53 [a.u.]")
# ax.set_ylabel(" p53 [a. u.]")

# ax.set_ylabel(r"distance to merge point ($\mu$m)")
ax.set_title("Per-condition fluorescence (across gastruloids)")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.margins(y=0.1)

from scipy.stats import ttest_ind

corr_A12_gmeans = []
for i in range(len(A12_gmeans)):
    gm = []
    for j in range(len(A12_gmeans[i])):
       gm.append(np.log(A12_gmeans[i][j]+1) / np.log(F3_gmeans[i][j]+1)) 
    corr_A12_gmeans.append(gm)

A12_mean, A12_std = summarize(corr_A12_gmeans)

A12_mean_sel = [A12_mean[i] for i in selected_CONDS]
A12_std_sel = [A12_std[i] for i in selected_CONDS]

data = A12_gmeans_sel
A12_gmeans_sel = [corr_A12_gmeans[i] for i in selected_CONDS]

# Run tests and add bars
ymax = max(max(d) for d in data) * 1.01  # starting height for bars
h = 0.02 * ymax   # height of bars
step = 0
steps = 0
data = A12_gmeans_sel

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

    ax.text((x1+x2)/2, y+h, stars, ha='center', va='bottom', fontsize=25)

    step += 5.7  # space out multiple bars
    steps+=1
    if steps==4: step=0 
plt.tight_layout()
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53ratio_stats_on_logratio.pdf")
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/p53_quantification/p53ratio_stats_on_logratio.svg")
plt.show()