### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, get_file_names, correct_path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
from scipy.ndimage import gaussian_filter1d

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

import matplotlib.pyplot as plt

model = StarDist2D.from_pretrained('2D_versatile_fluo')

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/center_to_edge_binned/"

DISTS_F3_WT = []
DISTS_A12_WT = []
DISTS_apo_WT = []

DISTS_F3_KO = []
DISTS_A12_KO = []
DISTS_apo_KO = []

EXPERIMENTS = ["2023_11_17_Casp3", "2024_03_Casp3"]
# EXPERIMENTS = ["2023_11_17_Casp3"]

TIMES = ["48hr", "72hr", "96hr"]

all_files = []
all_data = []

for TIME in TIMES:
    dists_F3_WT = []
    dists_A12_WT = []
    dists_Casp3_WT = []

    dists_F3_KO = []
    dists_A12_KO = []
    dists_Casp3_KO = []
    
    for EXP in EXPERIMENTS:
        ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/stacks/{}/KO/'.format(EXP, TIME)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/ctobjects/{}/KO/'.format(EXP, TIME)
        path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/{}/early_apoptosis/{}/KO/'.format(EXP, TIME)

        ### GET FULL FILE NAME AND FILE CODE ###
        files = get_file_names(path_data_dir)
            
        for f, file in enumerate(files):
            path_data = path_data_dir+file
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)

            file_path = path_save_results+embcode
            dists_contour_Casp3_current = np.load(file_path+"_dists_contour_Casp3.npy")
            dists_contour_A12_current = np.load(file_path+"_dists_contour_A12.npy")
            dists_contour_F3_current = np.load(file_path+"_dists_contour_F3.npy")
            
            dists_centroid_Casp3_current = np.load(file_path+"_dists_centroid_Casp3.npy")
            dists_centroid_A12_current = np.load(file_path+"_dists_centroid_A12.npy")
            dists_centroid_F3_current = np.load(file_path+"_dists_centroid_F3.npy")
            
            dists = dists_centroid_F3_current / (dists_centroid_F3_current + dists_contour_F3_current)
            all_files.append(file + " F3")
            all_data.append(dists)
            dists_F3_KO.append(dists)
            
            dists = dists_centroid_A12_current / (dists_centroid_A12_current + dists_contour_A12_current)
            all_files.append(file + " A12")
            all_data.append(dists)
            dists_A12_KO.append(dists)

            dists = dists_centroid_Casp3_current / (dists_centroid_Casp3_current + dists_contour_Casp3_current)
            dists_Casp3_KO.append(dists)
                    
        DISTS_F3_KO.append(dists_F3_KO)
        DISTS_A12_KO.append(dists_A12_KO)
        DISTS_apo_KO.append(dists_Casp3_KO)

        ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/stacks/{}/WT/'.format(EXP, TIME)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/ctobjects/{}/WT/'.format(EXP, TIME)
        path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/{}/early_apoptosis/{}/WT/'.format(EXP, TIME)


        ### GET FULL FILE NAME AND FILE CODE ###
        files = get_file_names(path_data_dir)
        
        for f, file in enumerate(files):
            path_data = path_data_dir+file
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)

            file_path = path_save_results+embcode
            
            dists_contour_Casp3_current = np.load(file_path+"_dists_contour_Casp3.npy")
            dists_contour_A12_current = np.load(file_path+"_dists_contour_A12.npy")
            dists_contour_F3_current = np.load(file_path+"_dists_contour_F3.npy")
            
            dists_centroid_Casp3_current = np.load(file_path+"_dists_centroid_Casp3.npy")
            dists_centroid_A12_current = np.load(file_path+"_dists_centroid_A12.npy")
            dists_centroid_F3_current = np.load(file_path+"_dists_centroid_F3.npy")
            
            dists = dists_centroid_F3_current / (dists_centroid_F3_current + dists_contour_F3_current)
            all_files.append(file + " F3")
            all_data.append(dists)
            dists_F3_WT.append(dists)
            
            dists = dists_centroid_A12_current / (dists_centroid_A12_current + dists_contour_A12_current)
            all_files.append(file + " A12")
            all_data.append(dists)
            dists_A12_WT.append(dists)

            dists = dists_centroid_Casp3_current / (dists_centroid_Casp3_current + dists_contour_Casp3_current)
            dists_Casp3_WT.append(dists)

    DISTS_F3_WT.append(dists_F3_WT)
    DISTS_A12_WT.append(dists_A12_WT)
    DISTS_apo_WT.append(dists_Casp3_WT)


import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from itertools import permutations
import random

def permutation_test_wasserstein(group1, group2, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    obs = wasserstein_distance(group1, group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    perm_stats = []
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm1, perm2 = combined[:n1], combined[n1:]
        perm_stats.append(wasserstein_distance(perm1, perm2))
    pval = np.mean([s >= obs for s in perm_stats])
    return obs, pval



F3 = DISTS_F3_WT[0]
A12 = DISTS_A12_WT[0]
results = []
for g in range(len(F3)):
    f3 = F3[g]
    a12 = A12[g]
    
    d_median = np.median(f3) - np.median(a12)
    w_dist, pval = permutation_test_wasserstein(f3, a12)
    
    results.append({
        "gastruloid": g,
        "median_diff": d_median,
        "wasserstein": w_dist,
        "pval": pval
    })

results_df = pd.DataFrame(results)
print(results_df)

import numpy as np
import matplotlib.pyplot as plt

# ---------- ECDF pooled across gastruloids ----------
def ecdf_pooled(f3_list, a12_list, title="Pooled ECDF (F3 vs A12)"):
    # Concatenate all gastruloids for each group
    f3_all  = np.concatenate([arr for arr in f3_list  if len(arr) > 0]) if len(f3_list)  else np.array([])
    a12_all = np.concatenate([arr for arr in a12_list if len(arr) > 0]) if len(a12_list) else np.array([])

    fig, ax = plt.subplots(figsize=(6,5))
    for label, vals in [("F3", f3_all), ("A12", a12_all)]:
        if vals.size == 0:
            continue
        x = np.sort(vals)
        y = np.arange(1, x.size+1) / x.size
        ax.step(x, y, where="post", label=label)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("R (radial position)"); ax.set_ylabel("ECDF")
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

# ---------- Per-gastruloid side-by-side violins ----------
def violins_per_gastruloid_lists(f3_list, a12_list, title="Radial distributions per gastruloid"):
    # Positions for each gastruloid
    n = max(len(f3_list), len(a12_list))
    x = np.arange(n)
    width = 0.35

    # Ensure same length by padding with empty arrays
    pad = lambda L, n: L + [np.array([])]*(n - len(L))
    f3_list  = pad(f3_list, n)
    a12_list = pad(a12_list, n)

    fig, ax = plt.subplots(figsize=(max(8, 1.2*n), 5))

    # Make violins; matplotlib accepts list-of-arrays directly
    parts_f3  = ax.violinplot(f3_list,  positions=x - width/2, showmedians=True, widths=0.7*width)
    parts_a12 = ax.violinplot(a12_list, positions=x + width/2, showmedians=True, widths=0.7*width)

    # Light styling (no explicit colors to keep defaults)
    for pc in parts_f3['bodies'] + parts_a12['bodies']:
        pc.set_alpha(0.5)

    # Add medians to legend proxies
    h_f3  = plt.Line2D([0],[0], lw=6, alpha=0.5, label="F3")
    h_a12 = plt.Line2D([0],[0], lw=6, alpha=0.5, label="A12")
    ax.legend([h_f3, h_a12], ["F3", "A12"], frameon=False)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_xlim(x[0]-1, x[-1]+1 if n>1 else 1)
    ax.set_xlabel("Gastruloid index")
    ax.set_ylabel("R (radial position)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# ---------- Optional: jittered raw points on top of violins ----------
def add_jitter_scatter_from_lists(ax, data_lists, positions, jitter=0.07, s=8, alpha=0.5, seed=0):
    rng = np.random.default_rng(seed)
    for pos, arr in zip(positions, data_lists):
        if arr is None or len(arr) == 0:
            continue
        xs = pos + rng.uniform(-jitter, jitter, size=len(arr))
        ax.scatter(xs, arr, s=s, alpha=alpha)

# Convenience wrapper to draw violins + raw points
def violins_with_points(f3_list, a12_list, title="Radial distributions per gastruloid"):
    n = max(len(f3_list), len(a12_list))
    x = np.arange(n); width = 0.35
    pad = lambda L, n: L + [np.array([])]*(n - len(L))
    f3_list  = pad(f3_list, n)
    a12_list = pad(a12_list, n)

    fig, ax = plt.subplots(figsize=(max(8, 1.2*n), 5))
    parts_f3  = ax.violinplot(f3_list,  positions=x - width/2, showmedians=True, widths=0.7*width)
    parts_a12 = ax.violinplot(a12_list, positions=x + width/2, showmedians=True, widths=0.7*width)
    for pc in parts_f3['bodies'] + parts_a12['bodies']:
        pc.set_alpha(0.5)

    # Raw points
    add_jitter_scatter_from_lists(ax, f3_list,  positions=x - width/2)
    add_jitter_scatter_from_lists(ax, a12_list, positions=x + width/2)

    ax.legend([plt.Line2D([0],[0], lw=6, alpha=0.5),
               plt.Line2D([0],[0], lw=6, alpha=0.5)],
              ["F3", "A12"], frameon=False)
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in x])
    ax.set_xlim(x[0]-1, x[-1]+1 if n>1 else 1)
    ax.set_xlabel("Gastruloid index"); ax.set_ylabel("R (radial position)")
    ax.set_title(title)
    plt.tight_layout(); plt.show()

# ---------- (Optional) Pooled histograms (density) ----------
def pooled_hist(f3_list, a12_list, bins=20, title="Pooled density (hist)"):
    f3_all  = np.concatenate([arr for arr in f3_list  if len(arr) > 0]) if len(f3_list)  else np.array([])
    a12_all = np.concatenate([arr for arr in a12_list if len(arr) > 0]) if len(a12_list) else np.array([])
    fig, ax = plt.subplots(figsize=(6,5))
    if f3_all.size:
        ax.hist(f3_all, bins=bins, range=(0,1), density=True, alpha=0.5, label="F3")
    if a12_all.size:
        ax.hist(a12_all, bins=bins, range=(0,1), density=True, alpha=0.5, label="A12")
    ax.set_xlim(0,1)
    ax.set_xlabel("R (radial position)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

f3_list  = DISTS_F3_WT[0]   # list of arrays, one per gastruloid
a12_list = DISTS_A12_WT[0]  # list of arrays, one per gastruloid

# Big-picture comparison at this time point (pooled ECDF):
ecdf_pooled(f3_list, a12_list, title="Timepoint X – Experiment 0: Pooled ECDF")

# Inspect gastruloid-to-gastruloid variability:
violins_per_gastruloid_lists(f3_list, a12_list, title="Timepoint X – Exp 0: Per-gastruloid violins")

# Same, but with raw points overlaid:
violins_with_points(f3_list, a12_list, title="Timepoint X – Exp 0: Violins + points")

# Optional: pooled density via histograms
pooled_hist(f3_list, a12_list, bins=20, title="Timepoint X – Exp 0: Pooled hist")

import numpy as np
from scipy.stats import wasserstein_distance

def permutation_test_wasserstein(group1, group2, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = wasserstein_distance(group1, group2)
    pooled = np.concatenate([group1, group2])
    n1 = len(group1)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        p1, p2 = pooled[:n1], pooled[n1:]
        if wasserstein_distance(p1, p2) >= obs:
            cnt += 1
    pval = (cnt + 1) / (n_perm + 1)
    return obs, pval

def per_gastruloid_stats(f3_list, a12_list, n_perm=2000, seed=0):
    stats = []
    for g, (f3, a12) in enumerate(zip(f3_list, a12_list)):
        if len(f3)==0 or len(a12)==0:
            stats.append((g, np.nan, np.nan, np.nan))
            continue
        dmed = float(np.median(f3) - np.median(a12))
        w, p = permutation_test_wasserstein(f3, a12, n_perm=n_perm, seed=seed+g)
        stats.append((g, dmed, w, p))
    # returns lists aligned by gastruloid index
    gids, dmeds, wstats, pvals = map(list, zip(*stats))
    return gids, dmeds, wstats, pvals

gids, dmeds, wstats, pvals = per_gastruloid_stats(f3_list, a12_list, n_perm=5000, seed=42)

from scipy.stats import combine_pvalues

p_global_fisher = combine_pvalues(pvals, method='fisher')[1]
print("Global p (Fisher, across gastruloids):", p_global_fisher)


import numpy as np

def bootstrap_ci(data, n_boot=10000, ci=95, seed=0, func=np.median):
    """
    Generic bootstrap CI for a 1D array.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        stats.append(func(sample))
    lower = np.percentile(stats, (100-ci)/2)
    upper = np.percentile(stats, 100 - (100-ci)/2)
    return func(data), (lower, upper)

# --- Example using your Δmedian values ---

# Step 1. Compute per-gastruloid Δmedian (F3 median - A12 median)
f3_list  = DISTS_F3_WT[0]   # list of arrays, one per gastruloid
a12_list = DISTS_A12_WT[0]  # list of arrays, one per gastruloid

dmeds = []
for f3, a12 in zip(f3_list, a12_list):
    if len(f3)==0 or len(a12)==0:
        continue
    dmeds.append(np.median(f3) - np.median(a12))

dmeds = np.array(dmeds)

# Step 2. Median of Δmedians + bootstrap CI
median_dmed, ci_dmed = bootstrap_ci(dmeds, n_boot=10000, ci=95, seed=20)

print(f"Median Δmedian across gastruloids = {median_dmed:.4f}")
print(f"95% bootstrap CI = [{ci_dmed[0]:.4f}, {ci_dmed[1]:.4f}]")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---- helpers ----
def delta_medians_per_gastruloid(f3_list, a12_list):
    dmeds = []
    for f3, a12 in zip(f3_list, a12_list):
        if len(f3)==0 or len(a12)==0:
            continue
        dmeds.append(np.median(f3) - np.median(a12))
    return np.asarray(dmeds, dtype=float)

def bootstrap_ci(data, n_boot=10000, ci=95, seed=42, func=np.median):
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        return np.nan, (np.nan, np.nan)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boots[i] = func(sample)
    lower = np.percentile(boots, (100-ci)/2)
    upper = np.percentile(boots, 100 - (100-ci)/2)
    return func(data), (lower, upper)

from scipy.stats import norm

def bootstrap_ci_bca(x, statfunc=np.median, n_boot=10000, alpha=0.05, seed=42):
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    theta_hat = statfunc(x)

    # bootstrap
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = statfunc(rng.choice(x, size=n, replace=True))
    boots.sort()

    # bias correction
    prop_less = np.clip((boots < theta_hat).mean(), 1e-6, 1-1e-6)
    z0 = norm.ppf(prop_less)

    # acceleration via jackknife
    jack = np.array([statfunc(np.concatenate((x[:i], x[i+1:]))) for i in range(n)])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack)**3)
    den = 6.0 * (np.sum((jack_mean - jack)**2)**1.5 + 1e-12)
    a = num/den if den != 0 else 0.0

    # adjusted quantiles
    z_lo, z_hi = norm.ppf(alpha/2), norm.ppf(1 - alpha/2)
    a1 = norm.cdf(z0 + (z0 + z_lo)/(1 - a*(z0 + z_lo)))
    a2 = norm.cdf(z0 + (z0 + z_hi)/(1 - a*(z0 + z_hi)))
    lo = np.percentile(boots, 100*a1)
    hi = np.percentile(boots, 100*a2)
    return float(theta_hat), (float(lo), float(hi))


# ---- compute stats ----
all_dmeds   = []  # list of arrays (Δmedian per gastruloid) per timepoint
medians     = []
cis         = []

for f3_list, a12_list in zip(DISTS_F3_WT, DISTS_A12_WT):
    dmeds = delta_medians_per_gastruloid(f3_list, a12_list)
    all_dmeds.append(dmeds)
    med, ci = bootstrap_ci(dmeds, n_boot=1000, ci=95, seed=42)
    # med, ci = bootstrap_ci_bca(dmeds, n_boot=10000, alpha=0.05, seed=42)
    medians.append(med)
    cis.append(ci)

# ---- plot ----
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(TIMES))  # 0,1,2
rng = np.random.default_rng(0)
jitter = 0.07
box_halfwidth = 0.35  # width for CI box per timepoint
median_halfwidth = 0.25

for i, (dmeds, med, ci) in enumerate(zip(all_dmeds, medians, cis)):
    # scatter of per-gastruloid Δmedians with horizontal jitter
    xs = x[i] + rng.uniform(-jitter, jitter, size=dmeds.size)
    ax.scatter(xs, dmeds, s=22, alpha=0.7)

    # shaded CI band (rectangle)
    ci_low, ci_high = ci
    rect = Rectangle((x[i] - box_halfwidth, ci_low),
                     2*box_halfwidth, ci_high - ci_low,
                     alpha=0.18, linewidth=0)
    ax.add_patch(rect)

    # median Δmedian (thick horizontal line)
    ax.hlines(med, x[i] - median_halfwidth, x[i] + median_halfwidth, linewidth=3)

    # small text annotation
    ax.text(x[i], ci_high, f"med={med:.3f}\nCI[{ci_low:.3f},{ci_high:.3f}]",
            ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(TIMES)
ax.axhline(0, linestyle="--", linewidth=1)   # reference: no difference
# ax.set_ylabel(r"\DeltaΔ median = median(R_F3) - median(R_A12)")
# ax.set_title("Radial positioning difference across time points")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("repeat {}".format(EXP))
plt.tight_layout()
plt.show()

# (Optional) print a concise numeric summary:
for tp, med, (lo, hi), dmeds in zip(TIMES, medians, cis, all_dmeds):
    print(f"{tp}: median Δmedian = {med:.4f}  (95% CI [{lo:.4f}, {hi:.4f}]),  n_gastruloids={len(dmeds)}")



import numpy as np
from scipy.stats import binomtest, binom

def exact_median_ci(deltas, alpha=0.05):
    x = np.sort(np.asarray(deltas, float))
    n = len(x)
    r = int(binom.ppf(alpha/2, n, 0.5))     # lower tail count
    s = int(np.ceil(binom.isf(alpha/2, n, 0.5)))  # upper tail cutoff
    L = max(0, r)
    U = min(n-1, s-1)
    return x[L], x[U]  # distribution-free CI for the median

def sign_test(deltas):
    d = np.asarray(deltas, float)
    d = d[d != 0]                # drop exact zeros
    n = len(d)
    k = int(np.sum(d > 0))
    return binomtest(k, n, 0.5, alternative="two-sided").pvalue


f3_list  = DISTS_F3_WT[2]   # list of arrays, one per gastruloid
a12_list = DISTS_A12_WT[2]  # list of arrays, one per gastruloid

dmeds = []
for f3, a12 in zip(f3_list, a12_list):
    if len(f3)==0 or len(a12)==0:
        continue
    dmeds.append(np.median(f3) - np.median(a12))

dmeds = np.array(dmeds)

ci_exact = exact_median_ci(dmeds, alpha=0.05)
p_sign   = sign_test(dmeds)
print("Exact 95% CI for median Δ:", ci_exact)
print("Sign test p-value:", p_sign)
