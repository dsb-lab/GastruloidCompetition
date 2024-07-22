### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, get_file_names, correct_path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/thesis/figures/spatial_distribution/"

DISTS_F3_WT = []
DISTS_A12_WT = []
DISTS_apo_WT = []

DISTS_F3_KO = []
DISTS_A12_KO = []
DISTS_apo_KO = []

TIMES = ["48hr", "72hr", "96hr"]
for TIME in TIMES:
    dists_F3 = []
    dists_A12 = []
    dists_Casp3 = []
    
    ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
    path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/KO/'.format(TIME)
    path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/KO/'.format(TIME)
    path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/mid_apoptosis/{}/KO/'.format(TIME)

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
        
        dists = dists_centroid_F3_current / (dists_centroid_F3_current + dists_contour_F3_current)
        dists_F3 = [*dists_F3, *dists]
        
        dists = dists_centroid_A12_current / (dists_centroid_A12_current + dists_contour_A12_current)
        dists_A12 = [*dists_A12, *dists]
        
        dists = dists_centroid_Casp3_current / (dists_centroid_Casp3_current + dists_contour_Casp3_current)
        dists_Casp3 = [*dists_Casp3, *dists]
        
                
    DISTS_F3_KO.append(dists_F3)
    DISTS_A12_KO.append(dists_A12)
    DISTS_apo_KO.append(dists_Casp3)

    ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
    path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/WT/'.format(TIME)
    path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/WT/'.format(TIME)
    path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/mid_apoptosis/{}/WT/'.format(TIME)


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
        
        dists = dists_centroid_F3_current / (dists_centroid_F3_current + dists_contour_F3_current)
        dists_F3 = [*dists_F3, *dists]
        
        dists = dists_centroid_A12_current / (dists_centroid_A12_current + dists_contour_A12_current)
        dists_A12 = [*dists_A12, *dists]
        
        dists = dists_centroid_Casp3_current / (dists_centroid_Casp3_current + dists_contour_Casp3_current)
        dists_Casp3 = [*dists_Casp3, *dists]
    

    DISTS_F3_WT.append(dists_F3)
    DISTS_A12_WT.append(dists_A12)
    DISTS_apo_WT.append(dists_Casp3)

# ks_dist = ks_2samp(samp, dists_contour_Casp3)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,3, figsize=(12,6),sharex=True)
for t, TIME in enumerate(TIMES):
    
    ax[0,t].hist(DISTS_F3_WT[t], color="green", alpha=0.5, bins=50, density=True)
    ax[0,t].hist(DISTS_A12_WT[t], color="magenta", alpha=0.5, bins=50, density=True)
    ax[0,t].set_yticks([])
    ax[0,t].set_xlim(-0.1,1.1)
    ax[0,t].set_title(TIME)
    ax[0,t].spines[['left', 'right', 'top']].set_visible(False)
    if t==0:
        ax[0,t].set_ylabel("WT")

    ax[1,t].hist(DISTS_F3_KO[t], color="green", alpha=0.5, bins=50, density=True, label="F3")
    ax[1,t].hist(DISTS_A12_KO[t], color="magenta", alpha=0.5, bins=50, density=True, label="A12")
    ax[1,t].set_yticks([])
    ax[1,t].set_xlim(-0.1,1.1)
    ax[1,t].spines[['left', 'right', 'top']].set_visible(False)
    ax[1,t].set_xlabel(r"relative position on gastruloid ($P$)")

    if t ==0:
        ax[1,t].set_ylabel("KO")

    if t==len(TIMES)-1:
        ax[1,t].legend(loc="upper left")

    # ax[1,t].hist(DISTS_apo[t], color="yellow", alpha=0.5, bins=50, density=True)
    # ax[1,t].set_xlabel(r"relative distance $P$")
    # ax[1,t].set_yticks([])
    # ax[1,t].set_xlim(-0.1,1.1)
    # ax[1,t].spines[['left', 'right', 'top']].set_visible(False)


    # ax[0].set_title("All")
    # dists = DISTS[t]
    # n_hist, bins, patches = ax[0].hist(dists, alpha=0.5, bins=30, label=TIME, density=True)
    # ax[0].set_xlabel("relative position on gastruloid")

    # ax[1].set_title("Apo")
    # dists = DISTS_apo[t]
    # n_hist, bins, patches = ax[1].hist(dists, alpha=0.5, bins=30, label="{} apo".format(TIME), density=True)
    # ax[1].set_xlabel("relative position on gastruloid")

    # ax[0].legend()
    # ax[1].legend()

plt.savefig(path_figures+"dists_condition.svg")
plt.savefig(path_figures+"dists_condition.pdf")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2, figsize=(10,3.5))
for t, TIME in enumerate(TIMES):
    
    dists = [*DISTS_F3_WT[t], *DISTS_A12_WT[t], *DISTS_F3_KO[t], *DISTS_A12_KO[t]]
    ax[0].hist(dists, alpha=0.5, bins=50, density=True, label=TIME)
    ax[0].set_yticks([])
    ax[0].set_xlim(-0.1,1.1)
    ax[0].set_xlabel(r"relative position on gastruloid ($P$)")
    ax[0].spines[['left', 'right', 'top']].set_visible(False)
    ax[0].set_title(r"$G(x)$")

    dists = [*DISTS_apo_WT[t], *DISTS_apo_KO[t]]
    ax[1].hist(dists, alpha=0.5, bins=50, density=True, label=TIME)
    ax[1].set_yticks([])
    ax[1].set_xlim(-0.1,1.1)
    ax[1].set_xlabel(r"relative position on gastruloid ($P$)")
    ax[1].spines[['left', 'right', 'top']].set_visible(False)
    ax[1].set_title(r"$F(x)$")
    if t==len(TIMES)-1:
        ax[1].legend(loc="upper left")

plt.tight_layout()
plt.savefig(path_figures+"dists_apo.svg")
plt.savefig(path_figures+"dists_apo.pdf")
plt.show()