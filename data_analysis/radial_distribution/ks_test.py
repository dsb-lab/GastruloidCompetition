### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, get_file_names, correct_path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

import random
from scipy.stats import ks_2samp

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/thesis/figures/spatial_distribution/"

import matplotlib as mpl
plt.rcParams.update({
    "text.usetex": True,
})
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath'
mpl.rc('font', size=16) 
mpl.rc('axes', labelsize=16) 
mpl.rc('xtick', labelsize=16) 
mpl.rc('ytick', labelsize=16) 
mpl.rc('legend', fontsize=16) 

import matplotlib.pyplot as plt
fig, ax = plt.subplots(3,3,figsize=(15,12), sharex=True)

apo_stages = ["early", "mid", "late"]
apo_stages_names = ["early apo", "mid apo", "late apo"]
for ap, apo_stage in enumerate(apo_stages):
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
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/KO/'.format(TIME)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/KO/'.format(TIME)
        path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/{}_apoptosis/{}/KO/'.format(apo_stage, TIME)

        ### GET FULL FILE NAME AND FILE CODE ###
        files = get_file_names(path_data_dir)

        channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
        if "96hr" in path_data_dir:
            channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

        for f, file in enumerate(files):
            path_data = path_data_dir+file
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)

            file_path = path_save_results+embcode
            dists_contour_Casp3_current = np.load(file_path+"_dists_contour_Casp3_F3.npy")
            dists_contour_A12_current = np.load(file_path+"_dists_contour_A12.npy")
            dists_contour_F3_current = np.load(file_path+"_dists_contour_F3.npy")
            
            dists_centroid_Casp3_current = np.load(file_path+"_dists_centroid_Casp3_F3.npy")
            dists_centroid_A12_current = np.load(file_path+"_dists_centroid_A12.npy")
            dists_centroid_F3_current = np.load(file_path+"_dists_centroid_F3.npy")
            
            dists_contour_Casp3 = [*dists_contour_Casp3, *dists_contour_Casp3_current]
            dists_contour_A12 = [*dists_contour_A12, *dists_contour_A12_current]
            dists_contour_F3 = [*dists_contour_F3, *dists_contour_F3_current]
            
            dists_centroid_Casp3 = [*dists_centroid_Casp3, *dists_centroid_Casp3_current]
            dists_centroid_A12 = [*dists_centroid_A12, *dists_centroid_A12_current]
            dists_centroid_F3 = [*dists_centroid_F3, *dists_centroid_F3_current]


        ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/WT/'.format(TIME)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/WT/'.format(TIME)
        path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/{}_apoptosis/{}/WT/'.format(apo_stage, TIME)


        ### GET FULL FILE NAME AND FILE CODE ###
        files = get_file_names(path_data_dir)

        channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
        if "96hr" in path_data_dir:
            channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

        # for f, file in enumerate(files):
        #     path_data = path_data_dir+file
        #     file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)

        #     file_path = path_save_results+embcode
        #     dists_contour_Casp3_current = np.load(file_path+"_dists_contour_Casp3_F3.npy")
        #     dists_contour_A12_current = np.load(file_path+"_dists_contour_A12.npy")
        #     dists_contour_F3_current = np.load(file_path+"_dists_contour_F3.npy")
            
        #     dists_centroid_Casp3_current = np.load(file_path+"_dists_centroid_Casp3_F3.npy")
        #     dists_centroid_A12_current = np.load(file_path+"_dists_centroid_A12.npy")
        #     dists_centroid_F3_current = np.load(file_path+"_dists_centroid_F3.npy")
            
        #     dists_contour_Casp3 = [*dists_contour_Casp3, *dists_contour_Casp3_current]
        #     dists_contour_A12 = [*dists_contour_A12, *dists_contour_A12_current]
        #     dists_contour_F3 = [*dists_contour_F3, *dists_contour_F3_current]
            
        #     dists_centroid_Casp3 = [*dists_centroid_Casp3, *dists_centroid_Casp3_current]
        #     dists_centroid_A12 = [*dists_centroid_A12, *dists_centroid_A12_current]
        #     dists_centroid_F3 = [*dists_centroid_F3, *dists_centroid_F3_current]
            
        dists_contour = np.array([*dists_contour_A12, *dists_contour_F3])
        dists_centroid = np.array([*dists_centroid_A12, *dists_centroid_F3])

        dists_contour = np.array(dists_contour_F3)
        dists_centroid = np.array(dists_centroid_F3)

        dists = dists_centroid / (dists_centroid + dists_contour)
        dists_apo = np.array(dists_centroid_Casp3) / (np.array(dists_centroid_Casp3) + np.array(dists_contour_Casp3))

        DISTS.append(dists)
        DISTS_CONT.append(dists_contour)
        DISTS_apo.append(dists_apo)
        DISTS_CONT_apo.append(dists_contour_Casp3)

    for t, TIME in enumerate(TIMES):
        dists = DISTS[t]
        dists_apo = DISTS_apo[t]

        sample_size = len(dists_apo)
        print(sample_size)
        max_samples = np.floor(len(dists) / sample_size).astype("int32")

        KS_dists = []
        for n in range(200):
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
        if t==0:
            ax[ap, t].set_ylabel(apo_stages_names[ap])
        if ap==0:
            ax[ap, t].set_title(TIMES[t])
        if ap==2:
            ax[ap, t].set_xlabel(r"KS-statistic ($D$)")
        if ap==0 and t==2:
            lab=r"$\{ D_b \}_{b=1}^B$"
        else:
            lab=None
        n_hist, bins, patches = ax[ap, t].hist(statistics, color="grey", alpha=0.7, bins=30, label=lab)
        ax[ap, t].vlines(statistic_casp3, 0, np.max(n_hist), label=r"$D_x$ \\ $P_{{D_x}}={:0.1f}$".format(frac_lower*100), color="k", lw=3) 
        ax[ap,t].legend(loc="upper right", framealpha=1.0)
        ax[ap, t].spines[['left', 'right', 'top']].set_visible(False)
        ax[ap, t].set_yticks([])
    
# plt.savefig(path_figures+"KS.svg")
# plt.savefig(path_figures+"KS.pdf")
plt.tight_layout()
plt.show()
