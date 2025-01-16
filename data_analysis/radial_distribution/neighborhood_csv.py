### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift, correct_path
import numpy as np
import matplotlib.pyplot as plt

# test

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

import matplotlib as mpl
plt.rcParams.update({
    "text.usetex": True,
})
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath'
mpl.rc('font', size=14) 
mpl.rc('axes', labelsize=14) 
mpl.rc('xtick', labelsize=14) 
mpl.rc('ytick', labelsize=14) 
mpl.rc('legend', fontsize=14) 

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/thesis/figures/neighborhood/"
for number_of_neighs in [5 ,10, 15, 20, 30, 50, 75, 100, 200]:

    TIMES = ["48hr", "72hr", "96hr"]
    CONDS = ["WT", "KO"]

    filenames = []
    F3_neigh = []
    A12_neigh = []
    F3_apo_early_neigh = []
    F3_apo_mid_neigh = []
    F3_apo_late_neigh = []
    A12_apo_early_neigh = []
    A12_apo_mid_neigh = []
    A12_apo_late_neigh = []
    for ap, apo_stage in enumerate(["early", "mid", "late"]):
        for TTT, TIME in enumerate(TIMES):
            for CCC, COND in enumerate(CONDS):
                path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/{}/'.format(TIME, COND)
                path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/{}/'.format(TIME, COND)
            
                try: 
                    files = get_file_names(path_save_dir)
                except: 
                    import os
                    os.mkdir(path_save_dir)
                    
                ### GET FULL FILE NAME AND FILE CODE ###
                files_data = get_file_names(path_data_dir)

                channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
                if "96hr" in path_data_dir:
                    channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

                neighs_fates_F3_sum = np.zeros((len(files_data), 2))
                neighs_fates_A12_sum = np.zeros((len(files_data), 2))
                neighs_fates_Casp3_F3_sum = np.zeros((len(files_data), 2))
                neighs_fates_Casp3_A12_sum = np.zeros((len(files_data), 2))

                total_f3s = []
                total_a12s = []
            
                for f, file in enumerate(files_data):
                    path_data = path_data_dir+file
                    file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
                    path_save = correct_path(path_save_dir+embcode)
                    try: 
                        files = get_file_names(path_save)
                    except: 
                        import os
                        os.mkdir(path_save)

                    ### DEFINE ARGUMENTS ###
                    segmentation_args={
                        'method': 'stardist2D', 
                        'model': model, 
                        'blur': None, 
                        'min_outline_length':100,
                    }

                    concatenation3D_args = {
                        'distance_th_z': 3.0, # microns
                        'relative_overlap':False, 
                        'use_full_matrix_to_compute_overlap':True, 
                        'z_neighborhood':2, 
                        'overlap_gradient_th':0.3, 
                        'min_cell_planes': 2,
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
                        # 'plot_stack_dims': (256, 256), 
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
                        # 'plot_stack_dims': (256, 256), 
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
                    
                    ch = channel_names.index("Casp3")
                    chans = [ch]
                    for _ch in range(len(channel_names)):
                        if _ch not in chans:
                            chans.append(_ch)
                    batch_args = {
                        'name_format':"ch"+str(ch)+"_{}_"+apo_stage,
                        'extension':".tif",
                }
                    plot_args = {
                        'plot_layout': (1,1),
                        'plot_overlap': 1,
                        'masks_cmap': 'tab10',
                        # 'plot_stack_dims': (256, 256), 
                        'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
                        'channels':[ch],
                        'min_outline_length':75,
                    }
                    
                    ### DEFINE ARGUMENTS ###
                    segmentation_args={
                        'method': 'stardist2D', 
                        'model': model, 
                        'blur': [5,1], 
                        # 'n_tiles': (2,2),
                    }
                    CT_Casp3 = cellSegTrack(
                        path_data,
                        path_save,
                        segmentation_args=segmentation_args,
                        concatenation3D_args=concatenation3D_args,
                        error_correction_args=error_correction_args,
                        plot_args=plot_args,
                        batch_args=batch_args,
                        channels=chans
                    )

                    CT_Casp3.load()

                    ### CORRECT DISTRIBUTIONS ###
                    F3_dist = []
                    areas = []
                    for cell in CT_F3.jitcells: 
                        for zid, z in enumerate(cell.zs[0]):
                            mask = cell.masks[0][zid]
                            img = CT_F3.hyperstack[0,z, channel_names.index("F3")]
                            F3_dist.append(np.mean(img[mask[:,1], mask[:,0]]))

                    A12_dist = []
                    for cell in CT_A12.jitcells: 
                        for zid, z in enumerate(cell.zs[0]):
                            mask = cell.masks[0][zid]
                            areas.append(len(mask))
                            img = CT_A12.hyperstack[0,z, channel_names.index("A12")]
                            A12_dist.append(np.mean(img[mask[:,1], mask[:,0]]))

                    total_f3s.append(len(CT_F3.jitcells))
                    total_a12s.append(len(CT_A12.jitcells))

                    area = np.mean(areas)
                    dim = 2*np.sqrt(area/np.pi)
                                    
                    ## Now contract the shape as much as we want. 
                    F3_dist = np.array(F3_dist)
                    A12_dist = np.array(A12_dist)
                    
                    mdiff = np.mean(F3_dist) - np.mean(A12_dist)
                    
                    zres = CT_F3.metadata["Zresolution"]
                    xyres = CT_F3.metadata["XYresolution"]
                    
                    fates = []
                    centers = []
                    for cell in CT_F3.jitcells:
                        fates.append(0)
                        centers.append(cell.centers[0]*[zres, xyres, xyres])
                    for cell in CT_A12.jitcells:
                        fates.append(1)
                        centers.append(cell.centers[0]*[zres, xyres, xyres])

                    len_pre_casp3 = len(centers)
                    for cell in CT_Casp3.jitcells:
                        casp3 = []
                        a12 = []
                        f3 = []
                        for zid, z in enumerate(cell.zs[0]):
                            mask = cell.masks[0][zid]
                            img = CT_Casp3.hyperstack[0,z, channel_names.index("Casp3")]
                            casp3.append(np.mean(img[mask[:,1], mask[:,0]]))
                            
                            img = CT_Casp3.hyperstack[0,z, channel_names.index("A12")]
                            a12.append(np.mean(img[mask[:,1], mask[:,0]]))
                            
                            img = CT_Casp3.hyperstack[0,z, channel_names.index("F3")]
                            f3.append(np.mean(img[mask[:,1], mask[:,0]]))

                        centers.append(cell.centers[0]*[zres, xyres, xyres])

                        zz = np.int64(cell.centers[0][0])
                        idx = cell.zs[0].index(zz)
                        
                        cell_f3 = f3[idx]
                        cell_a12 = a12[idx]

                        if mdiff > 0:
                            cell_a12 += mdiff
                        else: 
                            cell_f3 -= mdiff 
                            
                        if cell_f3 > cell_a12:
                            fates.append(2)
                        else:
                            fates.append(3)
                        
                    centers = np.array(centers)
                    fates = np.array(fates)
                    

                    ### NEIGHBORHOOD COMPUTATION 
                    from sklearn.neighbors import NearestNeighbors
                    nbrs = NearestNeighbors(n_neighbors=number_of_neighs+1, algorithm='ball_tree').fit(centers)
                    distances, neighs = nbrs.kneighbors(centers)

                    dist_th = (dim*xyres)*10000.0 #microns
                    dist_th_near = (dim*xyres)*0.25
                    
                    true_neighs = []
                    true_dists = []
                    for p, neigh_p in enumerate(neighs):
                        true_neigh_p = []
                        true_dists_p = []
                        for neigh in neigh_p[1:]:
                            dist = np.linalg.norm(centers[p]-centers[neigh])
                            if dist < dist_th:
                                if dist > dist_th_near:
                                    if (neigh < len_pre_casp3) or (apo_stage=="early"):
                                        true_dists_p.append(dist)
                                        true_neigh_p.append(neigh)
                            if len(true_neigh_p) == number_of_neighs: break
                        true_dists.append(true_dists_p)
                        true_neighs.append(true_neigh_p)
                    
                    neighs_fates = []
                    for p, neigh_p in enumerate(true_neighs):
                        lbs = []
                        fts = []
                        for neigh in neigh_p:
                            fts.append(fates[neigh])
                        neighs_fates.append(fts)

                    neighs_fates_F3 = [n for i,n in enumerate(neighs_fates) if fates[i] == 0]
                    neighs_fates_A12 = [n for i,n in enumerate(neighs_fates) if fates[i] == 1]
                    neighs_fates_Casp3_F3 = [n for i,n in enumerate(neighs_fates) if fates[i] == 2]
                    neighs_fates_Casp3_A12 = [n for i,n in enumerate(neighs_fates) if fates[i] == 3]
                    
                    for n_fates in neighs_fates_F3:
                        for _f in n_fates:
                            if _f in [0]:
                                neighs_fates_F3_sum[f,0] += 1
                            elif _f in [1]:
                                neighs_fates_F3_sum[f,1] += 1
                                
                    for n_fates in neighs_fates_A12:
                        for _f in n_fates:
                            if _f in [0]:
                                neighs_fates_A12_sum[f,0] += 1
                            elif _f in [1]:
                                neighs_fates_A12_sum[f,1] += 1

                    for n_fates in neighs_fates_Casp3_F3:
                        for _f in n_fates:
                            if _f in [0]:
                                neighs_fates_Casp3_F3_sum[f,0] += 1
                            elif _f in [1]:
                                neighs_fates_Casp3_F3_sum[f,1] += 1

                    for n_fates in neighs_fates_Casp3_A12:
                        for _f in n_fates:
                            if _f in [0]:
                                neighs_fates_Casp3_A12_sum[f,0] += 1
                            elif _f in [1]:
                                neighs_fates_Casp3_A12_sum[f,1] += 1

                    f3_a12_norm = np.array([total_f3s[f], total_a12s[f]])
                    
                    # neighs_fates_A12_sum[f] /= f3_a12_norm
                    neighs_fates_A12_sum[f] /= np.sum(neighs_fates_A12_sum[f])
                    
                    # neighs_fates_F3_sum[f] /= f3_a12_norm
                    neighs_fates_F3_sum[f] /= np.sum(neighs_fates_F3_sum[f])

                    # neighs_fates_Casp3_F3_sum[f] /= f3_a12_norm
                    neighs_fates_Casp3_F3_sum[f] /= np.sum(neighs_fates_Casp3_F3_sum[f])

                    # neighs_fates_Casp3_A12_sum[f] /= f3_a12_norm
                    neighs_fates_Casp3_A12_sum[f] /= np.sum(neighs_fates_Casp3_A12_sum[f])
                    
                    filenames.append(file)
                    F3_neigh.append(neighs_fates_F3_sum[f,1])
                    A12_neigh.append(neighs_fates_A12_sum[f,1])

                    if apo_stage=="early":
                        F3_apo_early_neigh.append(neighs_fates_Casp3_F3_sum[f,1])
                        A12_apo_early_neigh.append(neighs_fates_Casp3_A12_sum[f,1])
                    elif apo_stage=="mid":
                        F3_apo_mid_neigh.append(neighs_fates_Casp3_F3_sum[f,1])
                        A12_apo_mid_neigh.append(neighs_fates_Casp3_A12_sum[f,1])
                    elif apo_stage=="late":
                        F3_apo_late_neigh.append(neighs_fates_Casp3_F3_sum[f,1])
                        A12_apo_late_neigh.append(neighs_fates_Casp3_A12_sum[f,1])
    
    import csv
    output_file = path_figures+"neigh_{}_A12frac_raw.csv".format(number_of_neighs)

    colnames = ["file", "F3", "A12", "apo_F3_early", "apo_A12_early", "apo_F3_mid", "apo_A12_mid", "apo_F3_late", "apo_A12_late"]
    full_data = [colnames]
    for f in range(len(filenames[0:22])):
        dat = [filenames[f], F3_neigh[f], A12_neigh[f], F3_apo_early_neigh[f], A12_apo_early_neigh[f], F3_apo_mid_neigh[f], A12_apo_mid_neigh[f], F3_apo_late_neigh[f], A12_apo_late_neigh[f]]
        full_data.append(dat)

    # Output CSV file path

    # Write to CSV
    with open(output_file, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerows(full_data)

