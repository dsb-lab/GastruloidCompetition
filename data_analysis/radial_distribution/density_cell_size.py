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

fig, ax = plt.subplots(2,3, figsize=(12,6), sharey=True, sharex='col')

number_of_neighs = 15

for ap, apo_stage in enumerate(["early", "mid", "late"]):
    densities_F3_all = [[[], [], []], [[], [], []]]
    densities_A12_all = [[[], [], []], [[], [], []]]
    densities_F3_all_apo = [[[], [], []], [[], [], []]]
    densities_A12_all_apo = [[[], [], []], [[], [], []]]
    gastruloid_sizes = [[[], [], []], [[], [], []]]
        
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
            files = get_file_names(path_data_dir)

            channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
            if "96hr" in path_data_dir:
                channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

            neighs_fates_F3_sum = np.zeros((len(files), 2))
            neighs_fates_A12_sum = np.zeros((len(files), 2))
            neighs_fates_Casp3_F3_sum = np.zeros((len(files), 2))
            neighs_fates_Casp3_A12_sum = np.zeros((len(files), 2))

            densities_F3 = []
            densities_A12 = []
            densities_F3_apo = []
            densities_A12_apo = []
            
            for f, file in enumerate(files):
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

                centers = []
                channels_centers = []
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
                    
                Casp3_F3 = 0
                Casp3_A12 = 0
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
                    channels_centers.append("F3")

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
                        Casp3_F3+=1
                    else:
                        fates.append(3)
                        Casp3_A12+=1
                    
                centers = np.array(centers)
                fates = np.array(fates)

                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=number_of_neighs+1, algorithm='ball_tree').fit(centers)
                distances, neighs = nbrs.kneighbors(centers)

                dist_th = (dim*xyres)*1000.0 #microns
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
                                if neigh < len_pre_casp3:
                                    true_dists_p.append(dist)
                                    true_neigh_p.append(neigh)
                        if len(true_neigh_p) == number_of_neighs: break
                    true_dists.append(true_dists_p)
                    true_neighs.append(true_neigh_p)

                densities = [1/(np.mean(dists)**2) for dists in true_dists]

                _densities_F3 = [densities[n] for n in range(len(fates)) if fates[n] == 0]
                _densities_A12 = [densities[n] for n in range(len(fates)) if fates[n] == 1]
                _densities_F3_apo = [densities[n] for n in range(len(fates)) if fates[n] == 2]
                _densities_A12_apo = [densities[n] for n in range(len(fates)) if fates[n] == 3]
                    
                densities_F3 = [*densities_F3, *_densities_F3]
                densities_A12 = [*densities_A12, *_densities_A12]
                densities_F3_apo = [*densities_F3_apo, *_densities_F3_apo]
                densities_A12_apo = [*densities_A12_apo, *_densities_A12_apo]

            densities_F3_all[CCC][TTT] = [*densities_F3_all[CCC][TTT], *densities_F3]
            densities_A12_all[CCC][TTT] = [*densities_A12_all[CCC][TTT], *densities_A12]
            densities_F3_all_apo[CCC][TTT] = [*densities_F3_all_apo[CCC][TTT], *densities_F3_apo]
            densities_A12_all_apo[CCC][TTT] = [*densities_A12_all_apo[CCC][TTT], *densities_A12_apo]
            gastruloid_sizes[CCC][TTT] = [*gastruloid_sizes[CCC][TTT], len(centers)]

    F3s_mean_dens_WT = np.array([np.nanmean(densities_F3_all[0][i]) for i in range(3)])
    F3s_mean_dens_KO = np.array([np.nanmean(densities_F3_all[1][i]) for i in range(3)])
    F3s_mean_dens_apo_WT = np.array([np.nanmean(densities_F3_all_apo[0][i]) for i in range(3)])
    F3s_mean_dens_apo_KO = np.array([np.nanmean(densities_F3_all_apo[1][i]) for i in range(3)])

    A12s_mean_dens_WT = np.array([np.nanmean(densities_A12_all[0][i]) for i in range(3)])
    A12s_mean_dens_KO = np.array([np.nanmean(densities_A12_all[1][i]) for i in range(3)])
    A12s_mean_dens_apo_WT = np.array([np.nanmean(densities_A12_all_apo[0][i]) for i in range(3)])
    A12s_mean_dens_apo_KO = np.array([np.nanmean(densities_A12_all_apo[1][i]) for i in range(3)])

    g_sizes_WT = np.array([np.nanmean(gastruloid_sizes[0][i]) for i in range(3)])
    g_sizes_KO = np.array([np.nanmean(gastruloid_sizes[1][i]) for i in range(3)])

    F3s_stds_dens_WT = np.array([np.nanstd(densities_F3_all[0][i]) for i in range(3)])
    F3s_stds_dens_KO = np.array([np.nanstd(densities_F3_all[1][i]) for i in range(3)])
    A12s_stds_dens_WT = np.array([np.nanstd(densities_A12_all[0][i]) for i in range(3)])
    A12s_stds_dens_KO = np.array([np.nanstd(densities_A12_all[1][i]) for i in range(3)])
    g_sizes_stds_WT = np.array([np.nanstd(gastruloid_sizes[0][i]) for i in range(3)])
    g_sizes_stds_KO = np.array([np.nanstd(gastruloid_sizes[1][i]) for i in range(3)])

    conds = np.array([48, 72, 96])

    ax[0, ap].set_title("{} apoptotis".format(apo_stage))
    ax[0, ap].plot(conds, F3s_mean_dens_WT, color="green", lw=3, label="F3")
    ax[0, ap].scatter(conds, F3s_mean_dens_WT, color="green", s=100, edgecolor="k", zorder=10)
    ax[0, ap].plot(conds, F3s_mean_dens_apo_WT, color="green", lw=3, ls='--', label="F3 - apo")
    ax[0, ap].scatter(conds, F3s_mean_dens_apo_WT, color="green", s=100, edgecolor="k", zorder=10)

    ax[0, ap].plot(conds, A12s_mean_dens_WT, color="magenta", lw=3, label="A12")
    ax[0, ap].scatter(conds, A12s_mean_dens_WT, color="magenta", s=100, edgecolor="k", zorder=10)
    ax[0, ap].plot(conds, A12s_mean_dens_apo_WT, color="magenta", lw=3, ls='--', label="A12 - apo")
    ax[0, ap].scatter(conds, A12s_mean_dens_apo_WT, color="magenta", s=100, edgecolor="k", zorder=10)

    ax[1, ap].plot(conds, F3s_mean_dens_KO, color="green", lw=3, label="F3")
    ax[1, ap].scatter(conds, F3s_mean_dens_KO, color="green", s=100, edgecolor="k", zorder=10)
    ax[1, ap].plot(conds, F3s_mean_dens_apo_KO, color="green", lw=3, ls='--', label="F3 - apo")
    ax[1, ap].scatter(conds, F3s_mean_dens_apo_KO, color="green", s=100, edgecolor="k", zorder=10)

    ax[1, ap].plot(conds, A12s_mean_dens_KO, color="magenta", lw=3, label="A12")
    ax[1, ap].scatter(conds, A12s_mean_dens_KO, color="magenta", s=100, edgecolor="k", zorder=10)
    ax[1, ap].plot(conds, A12s_mean_dens_apo_KO, color="magenta", lw=3, ls='--', label="A12 - apo")
    ax[1, ap].scatter(conds, A12s_mean_dens_apo_KO, color="magenta", s=100, edgecolor="k", zorder=10)

    ax[1, ap].set_xticks(conds)
    ax[1, ap].set_xlabel("Time (hr)")
    if ap==0:
        ax[0, ap].set_ylabel(r"mean $\rho_\mathrm{{local}}$  $\mu \mathrm{{m}}^{{-2}}$")
        ax[1, ap].set_ylabel(r"mean $\rho_\mathrm{{local}}$  $\mu \mathrm{{m}}^{{-2}}$")

    ax[0, ap].spines[['right', 'top']].set_visible(False)
    ax[1, ap].spines[['right', 'top']].set_visible(False)

    miny = np.min([*F3s_mean_dens_KO,*A12s_mean_dens_KO,*F3s_mean_dens_apo_KO, *A12s_mean_dens_apo_KO, *F3s_mean_dens_WT,*A12s_mean_dens_WT])
    maxy = np.max([*F3s_mean_dens_KO,*A12s_mean_dens_KO,*F3s_mean_dens_apo_KO, *A12s_mean_dens_apo_KO, *F3s_mean_dens_WT,*A12s_mean_dens_WT])
    

# ax[0,3].axis(False)
# ax[0,2].legend()
# ax[1,3].plot(conds, g_sizes_WT, color="black", label="WT", lw=3)
# ax[1,3].scatter(conds, g_sizes_WT, color="black", s=100, edgecolor="k", zorder=10)
# ax[1,3].plot(conds, g_sizes_KO, color="grey", label="KO", lw=3)
# ax[1,3].scatter(conds, g_sizes_KO, color="grey", s=100, edgecolor="k", zorder=10)
# ax[1,3].set_xticks(conds)

# ax[1,3].legend()
# ax[1,3].set_ylabel("mean \# of cells")
plt.tight_layout()
plt.show()
