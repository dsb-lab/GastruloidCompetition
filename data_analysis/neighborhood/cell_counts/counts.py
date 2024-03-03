### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift, correct_path
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/figures/neighbors/"
try: 
    files = get_file_names(path_figures)
except: 
    import os
    os.mkdir(path_figures)
    
# ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
TIMES = ["48hr"]
CONDS = ["WT", "KO"]
for TIME in TIMES:
    path_figures_time = "{}{}/".format(path_figures, TIME)
    try: 
        files = get_file_names(path_figures_time)
    except: 
        import os
        os.mkdir(path_figures_time)
            
    for COND in CONDS:
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
            F3_dist = []
            areas = []
            for cell in CT_F3.jitcells: 
                for zid, z in enumerate(cell.zs[0]):
                    mask = cell.masks[0][zid]
                    img = CT_F3.hyperstack[0,z, channel_names.index("F3")]
                    F3_dist.append(np.mean(img[mask[:,1], mask[:,0]]))
                    areas.append(len(mask))
                centers.append(cell.centers[0])
                    
            A12_dist = []
            for cell in CT_A12.jitcells: 
                for zid, z in enumerate(cell.zs[0]):
                    mask = cell.masks[0][zid]
                    areas.append(len(mask))
                    img = CT_A12.hyperstack[0,z, channel_names.index("A12")]
                    A12_dist.append(np.mean(img[mask[:,1], mask[:,0]]))
                centers.append(cell.centers[0])
            
            def expand_contract_hull(outline, inc=-1):
                newoutline = []
                midpointx = (max(outline[:, 0]) + min(outline[:, 0])) / 2
                midpointy = (max(outline[:, 1]) + min(outline[:, 1])) / 2

                for p in outline:
                    newp = [0, 0]

                    # Get angle between point and center
                    x = p[0] - midpointx
                    y = p[1] - midpointy
                    theta = np.arctan2(y, x)
                    xinc = inc * np.cos(theta)
                    yinc = inc * np.sin(theta)
                    newp[0] = x + xinc + midpointx
                    newp[1] = y + yinc + midpointy
                    newoutline.append(newp)
                return np.array(newoutline), midpointx, midpointy
            
            def inhull_Delaunay(hull, p):
                """
                Test if points in `p` are in `hull`

                `p` should be a `NxK` coordinates of `N` points in `K` dimensions
                `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
                coordinates of `M` points in `K`dimensions for which Delaunay triangulation
                will be computed
                """

                return hull.find_simplex(p) >= 0
            
            
            from scipy.spatial import ConvexHull, Delaunay
            centers = np.array(centers)
            centers_order = np.argsort(centers[:,0])
            new_centers = centers[centers_order]
            uniquez = np.unique(new_centers[:,0])
            hulls = []
            used_zs = []
            area = np.mean(areas)
            dim = 2*np.sqrt(area/np.pi)
            if TIME == "48hr":
                inc = -1.*dim
            elif TIME =="72hr":
                inc = -1.2*dim
            else:
                inc = -1.6*dim
            for z in uniquez:
                if z in range(int(min(uniquez+3)),int(max(uniquez-3))):
                    centers_z = centers[np.where(centers[:,0]==z)]
                    if len(centers_z) >= 4:
                        used_zs.append(z)
                        hull = ConvexHull(centers_z[:, 1:])
                        outline = centers_z[:, 1:][hull.vertices]
                        outline_contracted, a, b = expand_contract_hull(outline, inc=inc)
                        inner_hull = Delaunay(outline_contracted)
                        hulls.append(inner_hull)
                    
            cells_to_remove = []
            for c, cell in enumerate(CT_F3.jitcells):
                center = cell.centers[0]
                z = center[0]
                if z not in used_zs:
                    cells_to_remove.append(cell.label)
                else:
                    zid = used_zs.index(z) 
                    if not inhull_Delaunay(hulls[zid], center[1:]):
                        cells_to_remove.append(cell.label)
                
            # for lab in cells_to_remove:
            #     CT_F3._del_cell(lab)
            
            CT_F3.update_labels()

            cells_to_remove = []
            for c, cell in enumerate(CT_A12.jitcells):
                center = cell.centers[0]
                z = center[0]
                if z not in used_zs:
                    cells_to_remove.append(cell.label)
                else:
                    zid = used_zs.index(z) 
                    if not inhull_Delaunay(hulls[zid], center[1:]):
                        cells_to_remove.append(cell.label)
                
            # for lab in cells_to_remove:
            #     CT_A12._del_cell(lab)
            
            CT_A12.update_labels()

            cells_to_remove = []
            for c, cell in enumerate(CT_Casp3.jitcells):
                center = cell.centers[0]
                z = center[0]
                if z not in used_zs:
                    cells_to_remove.append(cell.label)
                else:
                    zid = used_zs.index(z) 
                    if not inhull_Delaunay(hulls[zid], center[1:]):
                        cells_to_remove.append(cell.label)
                
            # for lab in cells_to_remove:
            #     CT_Casp3._del_cell(lab)
            
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
    
            labs_rem = []
            for cell in CT_Casp3.jitcells:
                zc = int(cell.centers[0][0])
                zcid = cell.zs[0].index(zc)
                center2D = cell.centers[0][1:]

                mask = cell.masks[0][zcid]
                stack = CT_Casp3.hyperstack[0, zc, ch]
                
                dists = []
                vals = []
                for point in mask:
                    dist = compute_distance_xy(center2D[0], point[0], center2D[1], point[1])
                    dists.append(dist)
                    val = stack[point[1], point[0]]
                    vals.append(val)
                dists = np.array(dists)
                vals  = np.array(vals)
                idxs = np.where(dists < 8.0)[0]
                
                dists = dists[idxs]
                vals = vals[idxs]
                slope, intercept, r_value, p_value, std_err = stats.linregress(dists,vals)
                if slope < 0:
                    labs_rem.append(cell.label)
            
            
            for lab in labs_rem:
                CT_Casp3._del_cell(lab)
            
            CT_Casp3.update_labels()
            

            # CT_F3.plot_tracking()
            # CT_A12.plot_tracking()
            # CT_Casp3.plot_tracking()

            ## Now contract the shape as much as we want. 
            F3_dist = np.array(F3_dist)
            A12_dist = np.array(A12_dist)
            
            mdiff = np.mean(F3_dist) - np.mean(A12_dist)
            if mdiff > 0:
                A12_dist += mdiff
            else: 
                F3_dist -= mdiff 
                
            # fig, ax = plt.subplots()
            # _ = ax.hist(F3_dist, color=[0.9,0,0.9,0.5], bins=40, density=True, label="F3")
            # _ = ax.hist(A12_dist, color=[0,0.8,0,0.5], bins=40, density=True, label="A12")
            # plt.legend()
            # plt.show()
            
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
            pre_casp3 = len(centers)
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
            nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(centers)
            distances, neighs = nbrs.kneighbors(centers)

            dist_th = (dim*xyres)*20.0 #microns
            dist_th_near = (dim*xyres)*0.5

            true_neighs = []

            for p, neigh_p in enumerate(neighs):
                true_neigh_p = []
                for neigh in neigh_p[1:]:
                    dist = np.linalg.norm(centers[p]-centers[neigh])
                    if dist < dist_th:
                        if dist > dist_th_near:
                            if neigh < pre_casp3:
                                true_neigh_p.append(neigh)
                true_neighs.append(true_neigh_p)

            
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
            neighs_fates_Casp3_A12_sum[f] /= np.sum(neighs_fates_Casp3_A12_sum[f])



        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        bot = [np.mean(neighs_fates_A12_sum[:,0]), np.mean(neighs_fates_Casp3_A12_sum[:,0]), np.mean(neighs_fates_F3_sum[:,0]), np.mean(neighs_fates_Casp3_F3_sum[:,0])]
        bot_std = [np.std(neighs_fates_A12_sum[:,0]), np.std(neighs_fates_Casp3_A12_sum[:,0]), np.std(neighs_fates_F3_sum[:,0]), np.std(neighs_fates_Casp3_F3_sum[:,0])]
        ax.bar([1,2,3,4], bot, color="green", yerr=bot_std, capsize=6)

        top = [np.mean(neighs_fates_A12_sum[:,1]), np.mean(neighs_fates_Casp3_A12_sum[:,1]), np.mean(neighs_fates_F3_sum[:,1]), np.mean(neighs_fates_Casp3_F3_sum[:,1])]
        top_std = [np.std(neighs_fates_A12_sum[:,1]), np.std(neighs_fates_Casp3_A12_sum[:,1]), np.std(neighs_fates_F3_sum[:,1]), np.std(neighs_fates_Casp3_F3_sum[:,1])]
        ax.bar([1,2,3,4], top, bottom =bot, tick_label=["A12", "Casp3 - A12", "F3", "Casp3 - F3"], color="magenta", yerr=top_std, capsize=6)
        ax.set_ylabel("percentage of neighbors")
        plt.savefig("{}{}.png".format(path_figures_time, COND))

        if True:
            print("A12", np.mean(neighs_fates_A12_sum, axis=0))
            print("Casp3 - A12", np.mean(neighs_fates_Casp3_A12_sum, axis=0))
            print("F3", np.mean(neighs_fates_F3_sum, axis=0))
            print("Casp3 - F3", np.mean(neighs_fates_Casp3_F3_sum, axis=0))

