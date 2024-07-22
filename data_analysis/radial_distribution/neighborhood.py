### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_4Dstack, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift, correct_path
import numpy as np
import matplotlib.pyplot as plt

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
    fig, ax = plt.subplots(2,3, figsize=(12,6), sharey=True, sharex='col')
    for ap, apo_stage in enumerate(["early", "mid", "late"]):

        NEIGHS_F3 = [[], []]
        NEIGHS_A12 = [[], []]
        NEIGHS_F3_apo = [[], []]
        NEIGHS_A12_apo = [[], []]
            
        # ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
        TIMES = ["48hr", "72hr", "96hr"]
        CONDS = ["WT", "KO"]
        
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
                    CT_F3 = CellTracking(
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

                    CT_A12 = CellTracking(
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
                    CT_Casp3 = CellTracking(
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

                    import numpy as np

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
                    
                    import numpy as np
                
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
                    
                    from sklearn.neighbors import NearestNeighbors
                    nbrs = NearestNeighbors(n_neighbors=number_of_neighs+1, algorithm='ball_tree').fit(centers)
                    distances, neighs = nbrs.kneighbors(centers)

                    dist_th = (dim*xyres)*10000.0 #microns
                    dist_th_near = (dim*xyres)*0.5
                    
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
                    
                    neighs_fates_A12_sum[f] /= f3_a12_norm
                    neighs_fates_A12_sum[f] /= np.sum(neighs_fates_A12_sum[f])
                    
                    neighs_fates_F3_sum[f] /= f3_a12_norm
                    neighs_fates_F3_sum[f] /= np.sum(neighs_fates_F3_sum[f])

                    neighs_fates_Casp3_F3_sum[f] /= f3_a12_norm
                    neighs_fates_Casp3_F3_sum[f] /= np.sum(neighs_fates_Casp3_F3_sum[f])

                    neighs_fates_Casp3_A12_sum[f] /= f3_a12_norm
                    neighs_fates_Casp3_A12_sum[f] /= np.sum(neighs_fates_Casp3_A12_sum[f])
                    
                NEIGHS_F3[CCC].append(neighs_fates_F3_sum)
                NEIGHS_F3_apo[CCC].append(neighs_fates_Casp3_F3_sum)
                NEIGHS_A12[CCC].append(neighs_fates_A12_sum)
                NEIGHS_A12_apo[CCC].append(neighs_fates_Casp3_A12_sum)

        F3s_mean_F3_neigh_WT = np.array([np.nanmean(NEIGHS_F3[0][i][:, 0]) for i in range(len(TIMES))])
        F3s_mean_F3_neigh_KO = np.array([np.nanmean(NEIGHS_F3[1][i][:, 0]) for i in range(len(TIMES))])
        F3s_mean_A12_neigh_WT = np.array([np.nanmean(NEIGHS_F3[0][i][:, 1]) for i in range(len(TIMES))])
        F3s_mean_A12_neigh_KO = np.array([np.nanmean(NEIGHS_F3[1][i][:, 1]) for i in range(len(TIMES))])

        F3s_apo_mean_F3_neigh_WT = np.array([np.nanmean(NEIGHS_F3_apo[0][i][:, 0]) for i in range(len(TIMES))])
        F3s_apo_mean_F3_neigh_KO = np.array([np.nanmean(NEIGHS_F3_apo[1][i][:, 0]) for i in range(len(TIMES))])
        F3s_apo_mean_A12_neigh_WT = np.array([np.nanmean(NEIGHS_F3_apo[0][i][:, 1]) for i in range(len(TIMES))])
        F3s_apo_mean_A12_neigh_KO = np.array([np.nanmean(NEIGHS_F3_apo[1][i][:, 1]) for i in range(len(TIMES))])

        A12s_mean_F3_neigh_WT = np.array([np.nanmean(NEIGHS_A12[0][i][:, 0]) for i in range(len(TIMES))])
        A12s_mean_F3_neigh_KO = np.array([np.nanmean(NEIGHS_A12[1][i][:, 0]) for i in range(len(TIMES))])
        A12s_mean_A12_neigh_WT = np.array([np.nanmean(NEIGHS_A12[0][i][:, 1]) for i in range(len(TIMES))])
        A12s_mean_A12_neigh_KO = np.array([np.nanmean(NEIGHS_A12[1][i][:, 1]) for i in range(len(TIMES))])

        A12s_apo_mean_F3_neigh_WT = np.array([np.nanmean(NEIGHS_A12_apo[0][i][:, 0]) for i in range(len(TIMES))])
        A12s_apo_mean_F3_neigh_KO = np.array([np.nanmean(NEIGHS_A12_apo[1][i][:, 0]) for i in range(len(TIMES))])
        A12s_apo_mean_A12_neigh_WT = np.array([np.nanmean(NEIGHS_A12_apo[0][i][:, 1]) for i in range(len(TIMES))])
        A12s_apo_mean_A12_neigh_KO = np.array([np.nanmean(NEIGHS_A12_apo[1][i][:, 1]) for i in range(len(TIMES))])


        conds = np.array([48, 72, 96])

        ax[0, ap].set_title("{} apoptotis".format(apo_stage))
        ax[0, ap].plot(conds, F3s_mean_A12_neigh_WT, color=[0.9,0.0,0.9], lw=4, label="F3")
        ax[0, ap].scatter(conds, F3s_mean_A12_neigh_WT, color=[0.9,0.0,0.9], s=100, edgecolor="k", zorder=10)
        ax[0, ap].plot(conds, F3s_apo_mean_A12_neigh_WT, color=[0.5,0.0,0.5], lw=4, ls='--', label="F3 - apo")
        ax[0, ap].scatter(conds, F3s_apo_mean_A12_neigh_WT, color=[0.5,0.0,0.5], s=100, edgecolor="k", zorder=10)

        ax[1, ap].plot(conds, F3s_mean_A12_neigh_KO, color=[0.9,0.0,0.9], lw=4, label="F3")
        ax[1, ap].scatter(conds, F3s_mean_A12_neigh_KO, color=[0.9,0.0,0.9], s=100, edgecolor="k", zorder=10)
        ax[1, ap].plot(conds, F3s_apo_mean_A12_neigh_KO, color=[0.5,0.0,0.5], lw=4, ls='--', label="F3 - apo")
        ax[1, ap].scatter(conds, F3s_apo_mean_A12_neigh_KO, color=[0.5,0.0,0.5], s=100, edgecolor="k", zorder=10)

        ax[1, ap].set_xticks(conds)
        ax[1, ap].set_xlabel("Time (hr)")
        if ap==0:
            ax[0, ap].set_ylabel(r"$\overline{F_{A12-WT}}$ around F3s")
            ax[1, ap].set_ylabel(r"$\overline{F_{A12-KO}}$ around F3s")

        ax[0, ap].spines[['right', 'top']].set_visible(False)
        ax[1, ap].spines[['right', 'top']].set_visible(False)
       
        if ap==2:
            ax[1, ap].legend()
        
    plt.tight_layout()
    plt.savefig(path_figures+"neigborhod_{}.svg".format(number_of_neighs))
    plt.savefig(path_figures+"neigborhod_{}.pdf".format(number_of_neighs))
plt.show()