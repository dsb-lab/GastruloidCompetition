### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, get_file_names, correct_path
import numpy as np
from scipy import stats

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')


TIMES = ["48hr", "72hr", "96hr"]
CONDITIONS = ["WT", "KO"]

for COND in CONDITIONS:
    DISTS_F3 = []
    DISTS_A12 = []
    DISTS_Casp3 = []

    for TIME in TIMES:
        dists_contour_Casp3 = []
        dists_contour_A12 = []
        dists_contour_F3 = []
        
        dists_centroid_Casp3 = []
        dists_centroid_A12 = []
        dists_centroid_F3 = []

        ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/{}/'.format(TIME, COND)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/{}/'.format(TIME, COND)
        path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/early_apoptosis/{}/{}/'.format(TIME, COND)

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
            
            dists_contour_Casp3 = [*dists_contour_Casp3, *dists_contour_Casp3_current]
            dists_contour_A12 = [*dists_contour_A12, *dists_contour_A12_current]
            dists_contour_F3 = [*dists_contour_F3, *dists_contour_F3_current]
            
            dists_centroid_Casp3 = [*dists_centroid_Casp3, *dists_centroid_Casp3_current]
            dists_centroid_A12 = [*dists_centroid_A12, *dists_centroid_A12_current]
            dists_centroid_F3 = [*dists_centroid_F3, *dists_centroid_F3_current]


        dists_F3 = np.array(dists_centroid_F3) / (np.array(dists_centroid_F3) + np.array(dists_contour_F3))
        dists_A12 = np.array(dists_centroid_A12) / (np.array(dists_centroid_A12) + np.array(dists_contour_A12))
        dists_Casp3 = np.array(dists_centroid_Casp3) / (np.array(dists_centroid_Casp3) + np.array(dists_contour_Casp3))

        DISTS_F3.append(dists_F3)
        DISTS_A12.append(dists_A12)
        DISTS_Casp3.append(dists_Casp3)

        np.savetxt(path_save_results+"dists_F3.csv", dists_F3)
        np.savetxt(path_save_results+"dists_A12.csv", dists_A12)
        np.savetxt(path_save_results+"dists_Casp3.csv", dists_Casp3)

