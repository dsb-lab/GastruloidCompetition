### LOAD PACKAGE ###
from qlivecell import get_file_name, get_file_names
import numpy as np
import matplotlib.pyplot as plt
### LOAD STARDIST MODEL ###

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

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/thesis/figures/spatial_distribution/"

# path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/figures/figures_joshi_poster/"

DISTS_F3_WT = []
DISTS_A12_WT = []
DISTS_apo_WT = []

DISTS_F3_KO = []
DISTS_A12_KO = []
DISTS_apo_KO = []

TIMES = ["48hr", "72hr", "96hr"]

all_files = []
all_data = []

for TIME in TIMES:
    dists_F3 = []
    dists_A12 = []
    dists_Casp3 = []
    
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
        
        dists = dists_centroid_F3_current / (dists_centroid_F3_current + dists_contour_F3_current)
        all_files.append(file + " F3")
        all_data.append(dists)
        dists_F3 = [*dists_F3, *dists]
        
        dists = dists_centroid_A12_current / (dists_centroid_A12_current + dists_contour_A12_current)
        all_files.append(file + " A12")
        all_data.append(dists)
        dists_A12 = [*dists_A12, *dists]
        
        dists = dists_centroid_Casp3_current / (dists_centroid_Casp3_current + dists_contour_Casp3_current)
        dists_Casp3 = [*dists_Casp3, *dists]
        
                
    DISTS_F3_KO.append(dists_F3)
    DISTS_A12_KO.append(dists_A12)
    DISTS_apo_KO.append(dists_Casp3)

    ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
    path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/WT/'.format(TIME)
    path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/WT/'.format(TIME)
    path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/early_apoptosis/{}/WT/'.format(TIME)


    ### GET FULL FILE NAME AND FILE CODE ###
    files = get_file_names(path_data_dir)

    dists_F3 = []
    dists_A12 = []
    dists_Casp3 = []
    
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
        all_files.append(file + " F3")
        all_data.append(dists)
        dists_F3 = [*dists_F3, *dists]
        
        dists = dists_centroid_A12_current / (dists_centroid_A12_current + dists_contour_A12_current)
        all_files.append(file + " A12")
        all_data.append(dists)
        dists_A12 = [*dists_A12, *dists]
        
        dists = dists_centroid_Casp3_current / (dists_centroid_Casp3_current + dists_contour_Casp3_current)
        dists_Casp3 = [*dists_Casp3, *dists]
    

    DISTS_F3_WT.append(dists_F3)
    DISTS_A12_WT.append(dists_A12)
    DISTS_apo_WT.append(dists_Casp3)


import csv
output_file = path_figures+"data_dists_WT_KO.csv"

full_data = []
for f in range(len(all_files)):
    dat = [all_files[f], *all_data[f]]
    full_data.append(dat)

# Output CSV file path

# Write to CSV
with open(output_file, mode="w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the header
    csv_writer.writerows(full_data)


all_files = []
all_data = []

for TIME in TIMES:
    for stage in ["early", "mid", "late"]:
        dists_Casp3 = []
        
        ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/KO/'.format(TIME)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/KO/'.format(TIME)
        path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/{}_apoptosis/{}/KO/'.format(stage, TIME)

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
            dists_centroid_Casp3_current = np.load(file_path+"_dists_centroid_Casp3.npy")
            
            dists = dists_centroid_Casp3_current / (dists_centroid_Casp3_current + dists_contour_Casp3_current)
            all_files.append(file + " - Casp3 - " + stage)
            all_data.append(dists)

        ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/WT/'.format(TIME)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/WT/'.format(TIME)
        path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/{}_apoptosis/{}/WT/'.format(stage, TIME)

        ### GET FULL FILE NAME AND FILE CODE ###
        files = get_file_names(path_data_dir)

        for f, file in enumerate(files):
            path_data = path_data_dir+file
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)

            file_path = path_save_results+embcode
            dists_contour_Casp3_current = np.load(file_path+"_dists_contour_Casp3.npy")
            dists_centroid_Casp3_current = np.load(file_path+"_dists_centroid_Casp3.npy")

            dists = dists_centroid_Casp3_current / (dists_centroid_Casp3_current + dists_contour_Casp3_current)
            all_files.append(file + "- Casp3 - " + stage)
            all_data.append(dists)        


import csv
output_file = path_figures+"data_dists_WT_KO_apo.csv"

full_data = []
for f in range(len(all_files)):
    dat = [all_files[f], *all_data[f]]
    full_data.append(dat)

# Output CSV file path

# Write to CSV
with open(output_file, mode="w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the header
    csv_writer.writerows(full_data)