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
    path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/early_apoptosis/{}/WT/'.format(TIME)

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
    path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/early_apoptosis/{}/KO/'.format(TIME)


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

t=1
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
statistics = [ks.statistic for ks in KS_dists]

casp3_sample = random.sample(list(dists_apo), sample_size)

ks_casp3 = ks_2samp(dists, casp3_sample)
statistic_casp3 = ks_casp3.statistic

total_kss = len(statistics)
frac_lower = np.sum(np.array(statistics < ks_casp3.statistic))/total_kss


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,5))
n_hist, bins, patches = ax.hist(statistics, color="grey", alpha=0.5, bins=30, label="KS muestras")
ax.vlines(statistic_casp3, 0, np.max(n_hist), label="KS APO") 
ax.set_xlabel("KS statistic")
ax.set_title("frac lower = {}".format(frac_lower))
ax.legend()

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