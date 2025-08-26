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

bin_ns = range(5, 50, 5)
sigmas = [3,4,5,7.5,10]
sct_size=10
line_w = 5
for density in [True, False]:
    for sigma in sigmas:
        for bin_n in bin_ns:
            if bin_n < 20:
                rem_pre=1
                rem_post=0
            else:
                rem_pre=3
                rem_post=3
            fig, ax = plt.subplots(2,3, figsize=(14,8),sharex=True, sharey=False)
            for t, TIME in enumerate(TIMES):
                
                _counts, bins = np.histogram(DISTS_F3_WT[t], bins=bin_n, density=density)
                dbins = np.mean(np.diff(bins))
                bins[1:] -= dbins
                bins[0] = 0
                counts = [_counts[i]/((4/3)*np.pi*(bins[i+1]**3-bins[i]**3)) for i in range(len(_counts))]
                
                if rem_post==0:
                    x = bins[1+rem_pre:]
                    y = counts[rem_pre:]
                else:
                    x = bins[1+rem_pre:-rem_post]
                    y = counts[rem_pre:-rem_post]

                ax[0,t].scatter(x, y, color="green", s=sct_size, alpha=0.5)
                y_sm = gaussian_filter1d(y, sigma=sigma)
                ax[0,t].plot(x, y_sm, color="green", lw=line_w)

                totals = np.sum(_counts)
                
                _counts, bins = np.histogram(DISTS_A12_WT[t], bins=bin_n, density=density)
                dbins = np.mean(np.diff(bins))
                bins[1:] -= dbins
                bins[0] = 0
                counts = [_counts[i]/((4/3)*np.pi*(bins[i+1]**3-bins[i]**3)) for i in range(len(_counts))]
                
                if rem_post==0:
                    x = bins[1+rem_pre:]
                    y = counts[rem_pre:]
                else:
                    x = bins[1+rem_pre:-rem_post]
                    y = counts[rem_pre:-rem_post]

                ax[0,t].scatter(x, y, color="magenta", s=sct_size, alpha=0.5)
                y_sm = gaussian_filter1d(y, sigma=sigma)
                ax[0,t].plot(x, y_sm, color="magenta", lw=line_w)

                totals+= np.sum(_counts)
                total_density = np.sum(totals)/((4/3)*np.pi*(bins[-1]**3))
                # ax[0,t].plot(bins, np.ones_like(bins)*total_density/2, color="grey", lw=line_w)

                # ax[0,t].set_yticks([])
                ax[0,t].set_xlim(-0.1,1.1)
                ax[0,t].set_title(TIME)
                # ax[0,t].spines[['left', 'right', 'top']].set_visible(False)
                if t==0:
                    ax[0,t].set_ylabel("WT")
                
                _counts, bins = np.histogram(DISTS_F3_KO[t], bins=bin_n, density=density)
                dbins = np.mean(np.diff(bins))
                bins[1:] -= dbins
                bins[0] = 0
                counts = [_counts[i]/((4/3)*np.pi*(bins[i+1]**3-bins[i]**3)) for i in range(len(_counts))]
                if rem_post==0:
                    x = bins[1+rem_pre:]
                    y = counts[rem_pre:]
                else:
                    x = bins[1+rem_pre:-rem_post]
                    y = counts[rem_pre:-rem_post]

                ax[1,t].scatter(x, y, color="green", s=sct_size, alpha=0.5)
                y_sm = gaussian_filter1d(y, sigma=sigma)
                ax[1,t].plot(x, y_sm, color="green", lw=line_w, label="F3")

                totals = np.sum(_counts)
                
                _counts, bins = np.histogram(DISTS_A12_KO[t], bins=bin_n, density=density)
                dbins = np.mean(np.diff(bins))
                bins[1:] -= dbins
                bins[0] = 0
                counts = [_counts[i]/((4/3)*np.pi*(bins[i+1]**3-bins[i]**3)) for i in range(len(_counts))]

                if rem_post==0:
                    x = bins[1+rem_pre:]
                    y = counts[rem_pre:]
                else:
                    x = bins[1+rem_pre:-rem_post]
                    y = counts[rem_pre:-rem_post]

                ax[1,t].scatter(x, y, color="magenta", s=sct_size, alpha=0.5)
                y_sm = gaussian_filter1d(y, sigma=sigma)
                ax[1,t].plot(x, y_sm, color="magenta", lw=line_w, label="A12")

                totals+= np.sum(_counts)
                total_density = np.sum(totals)/((4/3)*np.pi*(bins[-1]**3))
                # ax[1,t].plot(bins, np.ones_like(bins)*total_density/2, color="grey", lw=line_w, label="global density/2")
                
                # ax[1,t].set_yticks([])
                ax[1,t].set_xlim(-0.1,1.1)
                # ax[1,t].spines[['left', 'right', 'top']].set_visible(False)
                ax[1,t].set_xlabel(r"relative position on gastruloid")
                if t ==0:
                    ax[1,t].set_ylabel("KO")

                # if t==len(TIMES)-1:
                #     ax[1,t].legend(loc="upper left")
                
                # ax[0, 0].set_ylabel("cell per ")
                # ax[1,t].set_ylim
            plt.tight_layout()
            if density:
                plt.savefig(path_figures+"density/sigma{}_bins{}_density{}.png".format(sigma, bin_n, density))
            else:
                plt.savefig(path_figures+"nondensity/sigma{}_bins{}_density{}.png".format(sigma, bin_n, density))
