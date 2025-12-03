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

EXPERIMENTS = ["2023_11_17_Casp3"]
# EXPERIMENTS = ["2024_03_Casp3"]

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

from scipy.stats import ks_2samp

bins = 50
density=True

fig,ax = plt.subplots()
ax.hist(DISTS_F3_WT[0][0], bins=bins, density=density, alpha=0.5, color=(0.0, 0.8, 0.0))
ax.hist(DISTS_A12_WT[0][0], bins=bins, density=density, alpha=0.5, color=(0.9, 0.0, 0.9))
ax.hist(DISTS_A12_KO[0][0], bins=bins, density=density, alpha=0.5, color=(0.6, 0.0, 0.6))
ax.hist(DISTS_F3_KO[0][0], bins=bins, density=density, alpha=0.5, color="cyan")
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Step 1: Pool all A and B points
# -------------------------
all_A = np.concatenate([exp for exp in DISTS_F3_KO[1]])
all_B = np.concatenate([exp for exp in DISTS_F3_WT[1]])
all_C = np.concatenate([exp for exp in DISTS_A12_WT[1]])
all_D = np.concatenate([exp for exp in DISTS_A12_KO[1]])
print(f"Total points pooled: A={len(all_A)}, B={len(all_B)}")

# -------------------------
# Step 2: Compute medians and bootstrap CI
# -------------------------
def bootstrap_median_ci(data, n_boot=1000, alpha=0.05):
    medians = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        medians.append(np.median(sample))
    lower = np.percentile(medians, 100*alpha/2)
    upper = np.percentile(medians, 100*(1-alpha/2))
    return np.median(data), lower, upper

med_A, ci_low_A, ci_up_A = bootstrap_median_ci(all_B)
med_B, ci_low_B, ci_up_B = bootstrap_median_ci(all_C)
print(f"\nPooled Medians:")
print(f"  A median = {med_A:.3f} [{ci_low_A:.3f}, {ci_up_A:.3f}]")
print(f"  B median = {med_B:.3f} [{ci_low_B:.3f}, {ci_up_B:.3f}]")

# -------------------------
# Step 3: KS test on pooled data
# -------------------------
stat, pval_ks = ks_2samp(all_C, all_B)
print(f"\nKS test (pooled) = statistic {stat:.3f}, p-value {pval_ks:.4e}")

# -------------------------
# Step 4: Permutation test for pooled median difference
# -------------------------
def permutation_test(A, B, n_permutations=10000):
    obs_diff = np.median(A) - np.median(B)
    pooled = np.concatenate([A, B])
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(pooled)
        new_A = pooled[:len(A)]
        new_B = pooled[len(A):]
        if abs(np.median(new_A) - np.median(new_B)) >= abs(obs_diff):
            count += 1
    return count / n_permutations

perm_pval = permutation_test(all_C, all_B)
print(f"Permutation test p-value (pooled) = {perm_pval:.4e}")

# -------------------------
# Step 5: ECDF plot
# -------------------------
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1)/len(x)
    return x, y

xA, yA = ecdf(all_A)
xB, yB = ecdf(all_B)
xC, yC = ecdf(all_C)
xD, yD = ecdf(all_D)

plt.figure(figsize=(6,4))
plt.step(xA, yA, where='post', label="F3 with KO", color="cyan")
plt.step(xB, yB, where='post', label="F3 with WT", color=(0.0, 0.8, 0.0))
plt.step(xC, yC, where='post', label="A12-WT", color=(0.8, 0.0, 0.8))
plt.step(xD, yD, where='post', label="A12-KO", color=(0.6, 0.0, 0.6))

plt.xlabel("Radial metric R")
plt.ylabel("ECDF")
plt.title("Pooled Populations ECDF")
plt.legend()
plt.tight_layout()
plt.show()
