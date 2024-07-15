### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift, correct_path
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')
densities_F3_all = [[[], [], []], [[], [], []]]
densities_A12_all = [[[], [], []], [[], [], []]]
densities_F3_all_apo = [[[], [], []], [[], [], []]]
densities_A12_all_apo = [[[], [], []], [[], [], []]]
gastruloid_sizes = [[[], [], []], [[], [], []]]
for apo_stage in ["early"]:
    path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/figures/neighbors/{}/".format(apo_stage)
    try: 
        files = get_file_names(path_figures)
    except: 
        import os
        os.mkdir(path_figures)
        
    # ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
    TIMES = ["48hr", "72hr", "96hr"]
    CONDS = ["WT", "KO"]
    
    for TTT, TIME in enumerate(TIMES):
        path_figures_time = "{}{}/".format(path_figures, TIME)
        try: 
            files = get_file_names(path_figures_time)
        except: 
            import os
            os.mkdir(path_figures_time)
                
        for CCC, COND in enumerate(CONDS):
            path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/{}/'.format(TIME, COND)
            path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/{}/'.format(TIME, COND)
            
            for number_of_neighs in [10]:

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
                        
                    # CT_F3.plot_tracking()
                    # CT_A12.plot_tracking()
                    # CT_Casp3.plot_tracking()
                    
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
                            areas.append(len(mask))
                        channels_centers.append("F3")
                        centers.append(cell.centers[0])
                            
                    A12_dist = []
                    for cell in CT_A12.jitcells: 
                        for zid, z in enumerate(cell.zs[0]):
                            mask = cell.masks[0][zid]
                            areas.append(len(mask))
                            img = CT_A12.hyperstack[0,z, channel_names.index("A12")]
                            A12_dist.append(np.mean(img[mask[:,1], mask[:,0]]))
                        centers.append(cell.centers[0])
                        channels_centers.append("A12")

                    area = np.mean(areas)
                    dim = 2*np.sqrt(area/np.pi)
                    
                    import numpy as np
                    from scipy import stats
                    
                    def compute_distance_xy(x1, x2, y1, y2):
                        """
                        Parameters
                        ----------
                        x1 : number
                            x coordinate of point 1
                        x2 : number
                            x coordinate of point 2
                        y1 : number
                            y coordinate of point 1
                        y2 : number
                            y coordinate of point 2

                        Returns
                        -------
                        dist : number
                            euclidean distance between points (x1, y1) and (x2, y2)
                        """
                        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        return dist

                    ## Now contract the shape as much as we want. 
                    F3_dist = np.array(F3_dist)
                    A12_dist = np.array(A12_dist)
                    
                    mdiff = np.mean(F3_dist) - np.mean(A12_dist)
                    if mdiff > 0:
                        A12_dist += mdiff
                    else: 
                        F3_dist -= mdiff 
                    
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
                        if f3[idx] > a12[idx]:
                            fates.append(2)
                            Casp3_F3+=1
                        else:
                            fates.append(3)
                            Casp3_A12+=1
                        


                    print()
                    print(TIME)
                    print(COND)
                    print("F3", len(CT_F3.jitcells)+Casp3_F3)
                    print("A12", len(CT_A12.jitcells)+Casp3_A12)
                    
                    print("F3 casp3", Casp3_F3)
                    print("A12 casp3", Casp3_A12)
                    
                    print("F3 casp3 %", 100 * (Casp3_F3 / (len(CT_F3.jitcells)+Casp3_F3)))
                    print("A12 casp3 %", 100 * (Casp3_A12 / (len(CT_A12.jitcells)+Casp3_A12)))
                    print()

                    centers = np.array(centers)
                    fates = np.array(fates)

                    from scipy.spatial import Delaunay

                    def find_neighbors(pindex, triang):
                        neighbors = list()
                        for simplex in triang.simplices:
                            if pindex in simplex:
                                neighbors.extend([simplex[i] for i in range(len(simplex)) if simplex[i] != pindex])
                                '''
                                this is a one liner for if a simplex contains the point we`re interested in,
                                extend the neighbors list by appending all the *other* point indices in the simplex
                                '''
                        #now we just have to strip out all the dulicate indices and return the neighbors list:
                        return list(set(neighbors))

                    # For the caspase cells, check fate of neighbors as a percentage

                    # tri = Delaunay(centers)
                    # neighs = []
                    # for p, point in enumerate(centers):
                    #     neighs_p = find_neighbors(p, tri)
                    #     neighs.append(neighs_p)

                    from sklearn.neighbors import NearestNeighbors
                    nbrs = NearestNeighbors(n_neighbors=number_of_neighs+1, algorithm='ball_tree').fit(centers)
                    distances, neighs = nbrs.kneighbors(centers)

                    dist_th = (dim*xyres)*5.0 #microns
                    dist_th_near = (dim*xyres)*0.4

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

                    densities = [1/(np.mean(dists))**2 for dists in true_dists]

                    neighs_fates = []
                    for p, neigh_p in enumerate(true_neighs):
                        lbs = []
                        fts = []
                        for neigh in neigh_p:
                            fts.append(fates[neigh])
                        neighs_fates.append(fts)

                    neighs_n = [len(neighs_p) for neighs_p in true_neighs]
                    neighs_n_A12 = [n for i,n in enumerate(neighs_n) if fates[i] == 1]
                    neighs_n_F3 = [n for i,n in enumerate(neighs_n) if fates[i] == 0]
                    neighs_n_Casp3 = [n for i,n in enumerate(neighs_n) if fates[i] > 1]

                    y = [np.mean(x) for x in [neighs_n_A12, neighs_n_F3, neighs_n_Casp3]]
                    yerr = [np.std(x) for x in [neighs_n_A12, neighs_n_F3, neighs_n_Casp3]]
                    import matplotlib.pyplot as plt
                    # plt.bar([1,2,3], y, tick_label=["A12", "F3", "Casp3"], color=["magenta", "green", "yellow"], yerr=yerr, capsize=6)
                    # plt.ylabel("# of neighbors")
                    # plt.show()

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


                    neighs_fates_A12_sum[f] /= np.sum(neighs_fates_A12_sum[f])
                    neighs_fates_F3_sum[f] /= np.sum(neighs_fates_F3_sum[f])
                    neighs_fates_Casp3_F3_sum[f] /= np.sum(neighs_fates_Casp3_F3_sum[f])
                    print(neighs_fates_Casp3_A12_sum[f])
                    if np.sum(neighs_fates_Casp3_A12_sum[f])!=0:
                        neighs_fates_Casp3_A12_sum[f] /= np.sum(neighs_fates_Casp3_A12_sum[f])

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

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                bot = [np.mean(neighs_fates_A12_sum[:,0]), np.mean(neighs_fates_Casp3_A12_sum[:,0]), np.mean(neighs_fates_F3_sum[:,0]), np.mean(neighs_fates_Casp3_F3_sum[:,0])]
                bot_std = [np.std(neighs_fates_A12_sum[:,0]), np.std(neighs_fates_Casp3_A12_sum[:,0]), np.std(neighs_fates_F3_sum[:,0]), np.std(neighs_fates_Casp3_F3_sum[:,0])]
                ax.bar([1,2,3,4], bot, color="green", yerr=bot_std, capsize=6)

                top = [np.mean(neighs_fates_A12_sum[:,1]), np.mean(neighs_fates_Casp3_A12_sum[:,1]), np.mean(neighs_fates_F3_sum[:,1]), np.mean(neighs_fates_Casp3_F3_sum[:,1])]
                top_std = [np.std(neighs_fates_A12_sum[:,1]), np.std(neighs_fates_Casp3_A12_sum[:,1]), np.std(neighs_fates_F3_sum[:,1]), np.std(neighs_fates_Casp3_F3_sum[:,1])]
                ax.bar([1,2,3,4], top, bottom=bot, tick_label=["A12", "Casp3 - A12", "F3", "Casp3 - F3"], color="magenta", yerr=top_std, capsize=6)
                ax.set_ylabel("percentage of neighbors")
                plt.savefig("{}neighs_nneighs{}_{}.png".format(path_figures_time, number_of_neighs, COND))

                fig, ax = plt.subplots()
                ax.hist(densities_F3, density=True, color="green", bins=50, alpha=0.5)
                ax.hist(densities_A12, density=True, color="magenta", bins=50, alpha=0.5)
                ax.hist(densities_F3_apo, density=True, color="cyan", bins=50, alpha=0.5)
                ax.hist(densities_A12_apo, density=True, color="red", bins=50, alpha=0.5)
                ax.set_xlim(0.002, 0.022)
                plt.savefig("{}densities_nneighs{}_{}.png".format(path_figures_time, number_of_neighs, COND))
                # plt.show()

                if True:
                    print("A12", np.mean(neighs_fates_A12_sum, axis=0))
                    print("Casp3 - A12", np.mean(neighs_fates_Casp3_A12_sum, axis=0))
                    print("F3", np.mean(neighs_fates_F3_sum, axis=0))
                    print("Casp3 - F3", np.mean(neighs_fates_Casp3_F3_sum, axis=0))


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


alpha=0.3
fig, ax = plt.subplots(2,3, figsize=(16,8), sharex=True)

fig.suptitle("EARLY apoptotis")
ax[0,0].set_title("WT - WT 48hr")
ax[0,0].hist(densities_F3_all[0][0], color="green", bins=50, alpha=alpha)
ax[0,0].hist(densities_A12_all[0][0], color="magenta", bins=50, alpha=alpha)
ax[0,0].axvline(np.nanmean(densities_F3_all_apo[0][0]), color="green", lw=3, ls='--')
ax[0,0].axvline(np.nanmean(densities_A12_all_apo[0][0]), color="purple", lw=3, ls='--')
ax[0,0].axvline(np.nanmean(densities_F3_all[0][0]), color="green", lw=2)
ax[0,0].axvline(np.nanmean(densities_A12_all[0][0]), color="purple", lw=2)

ax[0,1].set_title("WT - WT 72hr")
ax[0,1].hist(densities_F3_all[0][1], color="green", bins=50, alpha=alpha)
ax[0,1].hist(densities_A12_all[0][1], color="magenta", bins=50, alpha=alpha)
ax[0,1].axvline(np.nanmean(densities_F3_all_apo[0][1]), color="green", lw=3, ls='--')
ax[0,1].axvline(np.nanmean(densities_A12_all_apo[0][1]), color="purple", lw=3, ls='--')
ax[0,1].axvline(np.nanmean(densities_F3_all[0][1]), color="green", lw=2)
ax[0,1].axvline(np.nanmean(densities_A12_all[0][1]), color="purple", lw=2)

ax[0,2].set_title("WT - WT 96hr")
ax[0,2].hist(densities_F3_all[0][2], color="green", bins=50, alpha=alpha, label="F3 - WT")
ax[0,2].hist(densities_A12_all[0][2], color="magenta", bins=50, alpha=alpha, label="A12 - WT")
ax[0,2].axvline(np.nanmean(densities_F3_all_apo[0][2]), color="green", lw=3, ls='--', label="F3 apo - WT")
ax[0,2].axvline(np.nanmean(densities_A12_all_apo[0][2]), color="purple", lw=3, ls='--', label="A12 apo - WT")
ax[0,2].axvline(np.nanmean(densities_F3_all[0][2]), color="green", lw=2, label="F3 - WT")
ax[0,2].axvline(np.nanmean(densities_A12_all[0][2]), color="purple", lw=2, label="A12 - WT")
ax[0,2].legend()

ax[1,0].set_title("WT - KO 48hr")
ax[1,0].hist(densities_F3_all[1][0], color="cyan", bins=50, alpha=alpha)
ax[1,0].hist(densities_A12_all[1][0], color="red", bins=50, alpha=alpha)
ax[1,0].axvline(np.nanmean(densities_F3_all[1][0]), color="blue", lw=2)
ax[1,0].axvline(np.nanmean(densities_A12_all[1][0]), color="red", lw=2)
ax[1,0].axvline(np.nanmean(densities_F3_all_apo[1][0]), color="blue", lw=3, ls='--')
ax[1,0].axvline(np.nanmean(densities_A12_all_apo[1][0]), color="red", lw=3, ls='--')

ax[1,1].set_title("WT - KO 72hr")
ax[1,1].hist(densities_F3_all[1][1], color="cyan", bins=50, alpha=alpha)
ax[1,1].hist(densities_A12_all[1][1], color="red", bins=50, alpha=alpha)
ax[1,1].axvline(np.nanmean(densities_F3_all[1][1]), color="blue", lw=2)
ax[1,1].axvline(np.nanmean(densities_A12_all[1][1]), color="red", lw=2)
ax[1,1].axvline(np.nanmean(densities_F3_all_apo[1][1]), color="blue", lw=3, ls='--')
ax[1,1].axvline(np.nanmean(densities_A12_all_apo[1][1]), color="red", lw=3, ls='--')

ax[1,2].set_title("WT - KO 96hr")
ax[1,2].hist(densities_F3_all[1][2], color="cyan", bins=50, alpha=alpha, label="F3 - KO")
ax[1,2].hist(densities_A12_all[1][2], color="red", bins=50, alpha=alpha, label="A12 - KO")
ax[1,2].axvline(np.nanmean(densities_F3_all[1][2]), color="blue", lw=2, label="F3 - KO")
ax[1,2].axvline(np.nanmean(densities_A12_all[1][2]), color="red", lw=2, label="A12 - KO")
ax[1,2].axvline(np.nanmean(densities_F3_all_apo[1][2]), color="blue", lw=3, label="F3 apo - KO", ls='--')
ax[1,2].axvline(np.nanmean(densities_A12_all_apo[1][2]), color="red", lw=3, label="A12 apo - KO", ls='--')
ax[1,2].legend()

plt.show()

# np.nanmean(densities_F3_all[1])
# np.nanmean(densities_F3_all[1])

# alpha=0.3
# fig, ax = plt.subplots(2,3, figsize=(14,7), sharex=True)
# fig.suptitle("EARLY apoptotis")

# ax[0,0].set_title("F3 - F3 48hr")
# ax[0,0].hist(densities_F3_all[0][0], color="green", bins=50, alpha=alpha)
# ax[0,0].hist(densities_F3_all[1][0], color="cyan", bins=50, alpha=alpha)
# ax[0,0].axvline(np.nanmean(densities_F3_all_apo[0][0]), color="green", lw=3)
# ax[0,0].axvline(np.nanmean(densities_F3_all_apo[1][0]), color="blue", lw=3)

# ax[0,1].set_title("F3 - F3 72hr")
# ax[0,1].hist(densities_F3_all[0][1], color="green", bins=50, alpha=alpha)
# ax[0,1].hist(densities_F3_all[1][1], color="cyan", bins=50, alpha=alpha)
# ax[0,1].axvline(np.nanmean(densities_F3_all_apo[0][1]), color="green", lw=3)
# ax[0,1].axvline(np.nanmean(densities_F3_all_apo[1][1]), color="blue", lw=3)

# ax[0,2].set_title("F3 - F3 96hr")
# ax[0,2].hist(densities_F3_all[0][2], color="green", bins=50, alpha=alpha, label="F3 - WT")
# ax[0,2].hist(densities_F3_all[1][2], color="cyan", bins=50, alpha=alpha, label="F3 - KO")
# ax[0,2].axvline(np.nanmean(densities_F3_all_apo[0][2]), color="green", lw=3, label="F3 apo - WT")
# ax[0,2].axvline(np.nanmean(densities_F3_all_apo[1][2]), color="blue", lw=3, label="F3 apo - KO")
# ax[0,2].legend()

# ax[1,0].set_title("A12 - A12 48hr")
# ax[1,0].hist(densities_A12_all[1][0], color="red", bins=50, alpha=alpha)
# ax[1,0].hist(densities_A12_all[0][0], color="magenta", bins=50, alpha=alpha)
# ax[1,0].axvline(np.nanmean(densities_A12_all_apo[0][0]), color="purple", lw=3)
# ax[1,0].axvline(np.nanmean(densities_A12_all_apo[1][0]), color="red", lw=3)

# ax[1,1].set_title("A12 - A12 72hr")
# ax[1,1].hist(densities_A12_all[1][1], color="red", bins=50, alpha=alpha)
# ax[1,1].hist(densities_A12_all[0][1], color="magenta", bins=50, alpha=alpha)
# ax[1,1].axvline(np.nanmean(densities_A12_all_apo[0][1]), color="purple", lw=3)
# ax[1,1].axvline(np.nanmean(densities_A12_all_apo[1][1]), color="red", lw=3)

# ax[1,2].set_title("A12 - A12 96hr")
# ax[1,2].hist(densities_A12_all[0][2], color="magenta", bins=50, alpha=alpha, label="A12 - WT")
# ax[1,2].hist(densities_A12_all[1][2], color="red", bins=50, alpha=alpha, label="A12 - KO")
# ax[1,2].axvline(np.nanmean(densities_A12_all_apo[0][2]), color="purple", lw=3, label="A12 apo - WT")
# ax[1,2].axvline(np.nanmean(densities_A12_all_apo[1][2]), color="red", lw=3, label="A12 apo - KO")
# ax[1,2].legend()

# plt.show()

fig, ax = plt.subplots(2,3, figsize=(16,8), sharex=True)

fig.suptitle("EARLY apoptotis")
ax[0,0].set_title("F3 - F3 48hr")
ax[0,0].hist(densities_F3_all[0][0], color="green", bins=50, alpha=alpha)
ax[1,0].hist(densities_A12_all[0][0], color="magenta", bins=50, alpha=alpha)
ax[0,0].axvline(np.nanmean(densities_F3_all_apo[0][0]), color="green", lw=3, ls='--')
ax[1,0].axvline(np.nanmean(densities_A12_all_apo[0][0]), color="purple", lw=3, ls='--')
ax[0,0].axvline(np.nanmean(densities_F3_all[0][0]), color="green", lw=2)
ax[1,0].axvline(np.nanmean(densities_A12_all[0][0]), color="purple", lw=2)

ax[0,1].set_title("F3 - F3 72hr")
ax[0,1].hist(densities_F3_all[0][1], color="green", bins=50, alpha=alpha)
ax[1,1].hist(densities_A12_all[0][1], color="magenta", bins=50, alpha=alpha)
ax[0,1].axvline(np.nanmean(densities_F3_all_apo[0][1]), color="green", lw=3, ls='--')
ax[1,1].axvline(np.nanmean(densities_A12_all_apo[0][1]), color="purple", lw=3, ls='--')
ax[0,1].axvline(np.nanmean(densities_F3_all[0][1]), color="green", lw=2)
ax[1,1].axvline(np.nanmean(densities_A12_all[0][1]), color="purple", lw=2)

ax[0,2].set_title("F3 - F3 96hr")
ax[0,2].hist(densities_F3_all[0][2], color="green", bins=50, alpha=alpha, label="F3 - WT")
ax[1,2].hist(densities_A12_all[0][2], color="magenta", bins=50, alpha=alpha, label="A12 - WT")
ax[0,2].axvline(np.nanmean(densities_F3_all_apo[0][2]), color="green", lw=3, ls='--', label="F3 apo - WT")
ax[1,2].axvline(np.nanmean(densities_A12_all_apo[0][2]), color="purple", lw=3, ls='--', label="A12 apo - WT")
ax[0,2].axvline(np.nanmean(densities_F3_all[0][2]), color="green", lw=2, label="F3 - WT")
ax[1,2].axvline(np.nanmean(densities_A12_all[0][2]), color="purple", lw=2, label="A12 - WT")

ax[1,0].set_title("A12 - A12 48hr")
ax[0,0].hist(densities_F3_all[1][0], color="cyan", bins=50, alpha=alpha)
ax[1,0].hist(densities_A12_all[1][0], color="red", bins=50, alpha=alpha)
ax[0,0].axvline(np.nanmean(densities_F3_all[1][0]), color="blue", lw=2)
ax[1,0].axvline(np.nanmean(densities_A12_all[1][0]), color="red", lw=2)
ax[0,0].axvline(np.nanmean(densities_F3_all_apo[1][0]), color="blue", lw=3, ls='--')
ax[1,0].axvline(np.nanmean(densities_A12_all_apo[1][0]), color="red", lw=3, ls='--')

ax[1,1].set_title("A12 - A12 72hr")
ax[0,1].hist(densities_F3_all[1][1], color="cyan", bins=50, alpha=alpha)
ax[1,1].hist(densities_A12_all[1][1], color="red", bins=50, alpha=alpha)
ax[0,1].axvline(np.nanmean(densities_F3_all[1][1]), color="blue", lw=2)
ax[1,1].axvline(np.nanmean(densities_A12_all[1][1]), color="red", lw=2)
ax[0,1].axvline(np.nanmean(densities_F3_all_apo[1][1]), color="blue", lw=3, ls='--')
ax[1,1].axvline(np.nanmean(densities_A12_all_apo[1][1]), color="red", lw=3, ls='--')

ax[1,2].set_title("A12 - A12 96hr")
ax[0,2].hist(densities_F3_all[1][2], color="cyan", bins=50, alpha=alpha, label="F3 - KO")
ax[1,2].hist(densities_A12_all[1][2], color="red", bins=50, alpha=alpha, label="A12 - KO")
ax[0,2].axvline(np.nanmean(densities_F3_all[1][2]), color="blue", lw=2, label="F3 - KO")
ax[1,2].axvline(np.nanmean(densities_A12_all[1][2]), color="red", lw=2, label="A12 - KO")
ax[0,2].axvline(np.nanmean(densities_F3_all_apo[1][2]), color="blue", lw=3, label="F3 apo - KO", ls='--')
ax[1,2].axvline(np.nanmean(densities_A12_all_apo[1][2]), color="red", lw=3, label="A12 apo - KO", ls='--')
ax[1,2].legend()
ax[0,2].legend()

plt.show()


F3s_mean_dens_WT = np.array([np.nanmean(densities_F3_all[0][i]) for i in range(3)])
F3s_mean_dens_KO = np.array([np.nanmean(densities_F3_all[1][i]) for i in range(3)])
A12s_mean_dens_WT = np.array([np.nanmean(densities_A12_all[0][i]) for i in range(3)])
A12s_mean_dens_KO = np.array([np.nanmean(densities_A12_all[1][i]) for i in range(3)])
g_sizes_WT = np.array([np.nanmean(gastruloid_sizes[0][i]) for i in range(3)])
g_sizes_KO = np.array([np.nanmean(gastruloid_sizes[1][i]) for i in range(3)])


F3s_stds_dens_WT = np.array([np.nanstd(densities_F3_all[0][i]) for i in range(3)])
F3s_stds_dens_KO = np.array([np.nanstd(densities_F3_all[1][i]) for i in range(3)])
A12s_stds_dens_WT = np.array([np.nanstd(densities_A12_all[0][i]) for i in range(3)])
A12s_stds_dens_KO = np.array([np.nanstd(densities_A12_all[1][i]) for i in range(3)])
g_sizes_stds_WT = np.array([np.nanstd(gastruloid_sizes[0][i]) for i in range(3)])
g_sizes_stds_KO = np.array([np.nanstd(gastruloid_sizes[1][i]) for i in range(3)])

conds = np.array([48, 72, 96])
fig, ax = plt.subplots(figsize=(6,4))
axt = ax.twinx()
ax.plot(conds, F3s_mean_dens_WT, color="green", lw=3, label="F3 - WT")
ax.fill_between(conds, F3s_mean_dens_WT-F3s_stds_dens_WT, F3s_mean_dens_WT+F3s_stds_dens_WT, color="green", alpha=0.3)

ax.plot(conds, F3s_mean_dens_KO, color="cyan", lw=3, label="F3 - KO")
ax.fill_between(conds, F3s_mean_dens_KO-F3s_stds_dens_KO, F3s_mean_dens_KO+F3s_stds_dens_KO, color="cyan", alpha=0.3)

ax.plot(conds, A12s_mean_dens_WT, color="magenta", lw=3, label="A12 - WT")
ax.plot(conds, A12s_mean_dens_KO, color="red", lw=3, label="A12 - KO")
ax.set_xticks(conds)
ax.set_xlabel("Time (hr)")
ax.set_ylabel(r"$\rho_\mathrm{{local}}$  $\mu \mathrm{{m}}^{{-2}}$")
axt.plot(conds, g_sizes_WT, color="black", label="WT", lw=3)
axt.plot(conds, g_sizes_KO, color="grey", label="KO", lw=3)
axt.legend()
ax.legend()
axt.set_ylabel("mean \# of cells")
plt.tight_layout()
plt.show()

from scipy.stats import ttest_ind
x1 = [i for i in densities_F3_all[1][1] if not np.isnan(i)]
x2 = [i for i in densities_A12_all[1][1] if not np.isnan(i)]

res = ttest_ind(x1, x2)