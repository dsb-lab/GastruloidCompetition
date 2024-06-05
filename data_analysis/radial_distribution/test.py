### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, get_file_names, correct_path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

DISTS = []
DISTS_apo = []
DISTS_CONT = []
DISTS_CONT_apo = []
TIMES = ["48hr", "72hr", "96hr"]
for TIME in TIMES:
    dists_contour_Casp3 = []
    dists_contour_A12 = []
    dists_contour_F3 = []
    dists_centroid_Casp3 = []
    dists_centroid_A12 = []
    dists_centroid_F3 = []

    ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
    path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/WT/'.format(TIME)
    path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/WT/'.format(TIME)
    path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/late_apoptosis/{}/WT/'.format(TIME)

    ### GET FULL FILE NAME AND FILE CODE ###
    files = get_file_names(path_data_dir)

    channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
    if "96hr" in path_data_dir:
        channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

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
        
        dists_contour_Casp3 = [*dists_contour_Casp3, *dists_contour_Casp3_current]
        dists_contour_A12 = [*dists_contour_A12, *dists_contour_A12_current]
        dists_contour_F3 = [*dists_contour_F3, *dists_contour_F3_current]
        
        dists_centroid_Casp3 = [*dists_centroid_Casp3, *dists_centroid_Casp3_current]
        dists_centroid_A12 = [*dists_centroid_A12, *dists_centroid_A12_current]
        dists_centroid_F3 = [*dists_centroid_F3, *dists_centroid_F3_current]


    ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
    path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/KO/'.format(TIME)
    path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/KO/'.format(TIME)
    path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/late_apoptosis/{}/KO/'.format(TIME)


    ### GET FULL FILE NAME AND FILE CODE ###
    files = get_file_names(path_data_dir)

    channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
    if "96hr" in path_data_dir:
        channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

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
        
        dists_contour_Casp3 = [*dists_contour_Casp3, *dists_contour_Casp3_current]
        dists_contour_A12 = [*dists_contour_A12, *dists_contour_A12_current]
        dists_contour_F3 = [*dists_contour_F3, *dists_contour_F3_current]
        
        dists_centroid_Casp3 = [*dists_centroid_Casp3, *dists_centroid_Casp3_current]
        dists_centroid_A12 = [*dists_centroid_A12, *dists_centroid_A12_current]
        dists_centroid_F3 = [*dists_centroid_F3, *dists_centroid_F3_current]
        
    dists_contour = np.array([*dists_contour_A12, *dists_contour_F3])
    dists_centroid = np.array([*dists_centroid_A12, *dists_centroid_F3])

    dists = dists_centroid / (dists_centroid + dists_contour)
    dists_apo = np.array(dists_centroid_Casp3) / (np.array(dists_centroid_Casp3) + np.array(dists_contour_Casp3))

    DISTS.append(dists)
    DISTS_CONT.append(dists_contour)
    DISTS_apo.append(dists_apo)
    DISTS_CONT_apo.append(dists_contour_Casp3)

mean_dists = [np.mean(DISTS[t]) for t in range(len(TIMES))]
mean_dists_apo = [np.mean(DISTS_apo[t]) for t in range(len(TIMES))]
print()
print(mean_dists)
print(mean_dists_apo)

fig, ax = plt.subplots(2,3,figsize=(15,12))
for t, TIME in enumerate(TIMES):
    ax[0,t].set_title(TIME)
    dists = DISTS[t]
    n_hist, bins, patches = ax[0,t].hist(dists, alpha=0.5, bins=30, label=TIME, density=True, color="grey")
    dists = DISTS_apo[t]
    n_hist, bins, patches = ax[0,t].hist(dists, alpha=0.5, bins=30, label="{} apo".format(TIME), density=True, color="yellow")
    ax[0,t].set_xlabel("relative position on gastruloid")

    ax[1,t].set_title(TIME)
    dists = DISTS_CONT[t]
    n_hist, bins, patches = ax[1,t].hist(dists, alpha=0.5, bins=30, label=TIME, density=True, color="grey")
    dists = DISTS_CONT_apo[t]
    n_hist, bins, patches = ax[1,t].hist(dists, alpha=0.5, bins=30, label="{} apo".format(TIME), density=True, color="yellow")
    ax[1,t].set_xlabel("closest distance to edge")

    ax[0,t].legend()
    ax[1,t].legend()
plt.tight_layout()
plt.show()

import random
from scipy.stats import ks_2samp

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,3,figsize=(15,10))
for t, TIME in enumerate(TIMES):
    dists = DISTS[t]
    dists_apo = DISTS_apo[t]

    sample_size = len(dists_apo)
    max_samples = np.floor(len(dists) / sample_size).astype("int32")

    KS_dists = []
    for n in range(100):
        for s in range(max_samples):
            samp = random.sample(list(dists), sample_size)
            ks_dist = ks_2samp(dists, samp)
            KS_dists.append(ks_dist)
    statistics = np.array([ks.statistic for ks in KS_dists])

    casp3_sample = random.sample(list(dists_apo), sample_size)

    ks_casp3 = ks_2samp(dists, casp3_sample)
    statistic_casp3 = ks_casp3.statistic

    total_kss = len(statistics)
    frac_lower = np.sum(np.array(statistics < ks_casp3.statistic))/total_kss


    n_hist, bins, patches = ax[t].hist(statistics, color="grey", alpha=0.5, bins=30, label="KS muestras")
    ax[t].vlines(statistic_casp3, 0, np.max(n_hist), label="KS APO") 
    ax[t].set_xlabel("KS statistic")
    ax[t].set_title("{} , frac lower = {:0.4f}".format(TIME, frac_lower))
    ax[t].legend()

plt.tight_layout()
plt.show()


# ks_dist = ks_2samp(samp, dists_contour_Casp3)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].hist(dists_contour_F3, color="green", alpha=0.5, bins=50, density=True)
ax[0].hist(dists_contour_A12, color="magenta", alpha=0.5, bins=50, density=True)
ax[0].hist(dists_contour_Casp3, color="yellow", bins=50, density=True)
ax[0].set_xlabel("distance to closest embryo border")
ax[0].set_yticks([])
ax[1].hist(dists_centroid_F3, color="green", alpha=0.5, bins=50, density=True)
ax[1].hist(dists_centroid_A12, color="magenta", alpha=0.5, bins=50, density=True)
ax[1].hist(dists_centroid_Casp3, color="yellow", bins=50, density=True)
ax[1].set_xlim(0,100)
ax[1].set_xlabel("distance to embryo centroid")
ax[1].set_yticks([])
plt.show()

fig, ax = plt.subplots(2,figsize=(15,12))
for t, TIME in enumerate(TIMES):
    ax[0].set_title("All")
    dists = DISTS[t]
    n_hist, bins, patches = ax[0].hist(dists, alpha=0.5, bins=30, label=TIME, density=True)
    ax[0].set_xlabel("relative position on gastruloid")

    ax[1].set_title("Apo")
    dists = DISTS_apo[t]
    n_hist, bins, patches = ax[1].hist(dists, alpha=0.5, bins=30, label="{} apo".format(TIME), density=True)
    ax[1].set_xlabel("relative position on gastruloid")

    ax[0].legend()
    ax[1].legend()
plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(10,5))

# ax.hist(dists_contour_F3, color="green", alpha=0.5, bins=np.arange(0, 50, 5))
# ax.hist(dists_contour_A12, color="magenta", alpha=0.5, bins=np.arange(0, 50, 5))
# ax.hist(dists_contour_Casp3, color="yellow", bins=np.arange(0, 50, 5))
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel("distance to closest embryo border")
# ax.set_yticks([])

# plt.show()