### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, get_file_names, correct_path
import numpy as np
from scipy import stats

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

from numba import njit

@njit
def compute_distance_xyz(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return dist

@njit
def compute_dists(points1, points2):
    dists = np.zeros((len(points1), len(points2)))
    for i, center in enumerate(points1):
        for j, cont in enumerate(points2):
            dists[i,j] = compute_distance_xyz(center, cont)
    return dists

EXPERIMENTS = ["2023_11_17_Casp3", "2024_03_Casp3"]
TIMES = ["48hr", "72hr", "96hr"]

EXP = EXPERIMENTS[1]
CONDS = ["WT", "KO"]

apo_stages = ["early", "mid", "late"]


for TIME in TIMES:
    print(TIME)
    for COND in CONDS:
        print(COND)
        if TIME=="96hr":
            if COND=="KO":
                binths = [[5,4],[5.0, 4.0],[6.0, 5.0]]
            elif COND=="WT":
                binths = [[7.5, 5.0],[7.5, 5.0],[7.5, 5.0]]
        
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/stacks/{}/{}/'.format(EXP, TIME, COND)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/ctobjects/{}/{}/'.format(EXP, TIME, COND)
        try: 
            files = get_file_names(path_save_dir)
        except: 
            import os
            os.mkdir(path_save_dir)

        ### GET FULL FILE NAME AND FILE CODE ###
        files_data = get_file_names(path_data_dir)
        channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]

        if "96hr" in path_data_dir:
            if EXP == "2023_11_17_Casp3":
                channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]
            
        for apo_stage in apo_stages:
            path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/{}/{}_apoptosis/{}/{}/'.format(EXP, apo_stage, TIME, COND)

            for f, file_data in enumerate(files_data):

                path_data = path_data_dir+file_data
                file, embcode = get_file_name(path_data_dir, file_data, allow_file_fragment=False, return_files=False, return_name=True)
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
                    # 'n_tiles': (2,2),
                }

                concatenation3D_args = {
                    'distance_th_z': 3.0, # microns
                    'relative_overlap':False, 
                    'use_full_matrix_to_compute_overlap':True, 
                    'z_neighborhood':2, 
                    'overlap_gradient_th':0.3, 
                    'min_cell_planes': 2
                }

                error_correction_args = {
                    'backup_steps': 10,
                    'line_builder_mode': 'points',
                }
                
                ch_F3 = channel_names.index("F3")
                
                batch_args = {
                    'name_format':"ch"+str(ch_F3)+"_{}",
                    'extension':".tif",
                }
                plot_args = {
                    'plot_layout': (1,1),
                    'plot_overlap': 1,
                    'masks_cmap': 'tab10',
                    'plot_stack_dims': (256, 256), 
                    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
                    'channels':[ch_F3],
                    'min_outline_length':75,
                }
                
                chans = [ch_F3]
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
                # CT_F3.plot_tracking()
                
                ch_A12 = channel_names.index("A12")
                batch_args = {
                    'name_format':"ch"+str(ch_A12)+"_{}",
                    'extension':".tif",
                }
                plot_args = {
                    'plot_layout': (1,1),
                    'plot_overlap': 1,
                    'masks_cmap': 'tab10',
                    'plot_stack_dims': (256, 256), 
                    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
                    'channels':[ch_A12],
                    'min_outline_length':75,
                }

                chans = [ch_A12]
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
                # CT_A12.plot_tracking()
                
                ch_Casp3 = channel_names.index("Casp3")

                batch_args = {
                    'name_format':"ch"+str(ch_Casp3)+"_{}_"+apo_stage,
                    'extension':".tif",
                }
                plot_args = {
                    'plot_layout': (1,1),
                    'plot_overlap': 1,
                    'masks_cmap': 'tab10',
                    'plot_stack_dims': (256, 256), 
                    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
                    # 'channels':[ch_A12, ch_F3, ch_Casp3],
                    'channels':[ch_Casp3],
                    'min_outline_length':75,
                }
                
                chans = [ch_Casp3]
                for _ch in range(len(channel_names)):
                    if _ch not in chans:
                        chans.append(_ch)

                ### DEFINE ARGUMENTS ###
                segmentation_args={
                    'method': 'stardist2D', 
                    'model': model, 
                    'blur': [3,1], 
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
                # CT_Casp3.plot_tracking(plot_args=plot_args)

                import numpy as np

                ### CORRECT DISTRIBUTIONS ###

                areas = []

                F3_dist = []
                for cell in CT_F3.jitcells: 
                    for zid, z in enumerate(cell.zs[0]):
                        mask = cell.masks[0][zid]
                        img = CT_F3.hyperstack[0,z, channel_names.index("F3")]
                        F3_dist.append(np.mean(img[mask[:,1], mask[:,0]]))
                        areas.append(len(mask))
                        
                A12_dist = []
                for cell in CT_A12.jitcells: 
                    for zid, z in enumerate(cell.zs[0]):
                        mask = cell.masks[0][zid]
                        areas.append(len(mask))
                        img = CT_A12.hyperstack[0,z, channel_names.index("A12")]
                        A12_dist.append(np.mean(img[mask[:,1], mask[:,0]]))

                area = np.mean(areas)
                dim = 2*np.sqrt(area/np.pi)

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

                points = []
                zs = []
                for cell in CT_F3.jitcells:
                    outlines = cell.outlines[0]
                    for zid, z in enumerate(cell.zs[0]):
                        for point in outlines[zid]:
                            point3D = [z, point[0], point[1]]
                            points.append(point3D)

                for cell in CT_A12.jitcells:
                    outlines = cell.outlines[0]
                    for zid, z in enumerate(cell.zs[0]):
                        for point in outlines[zid]:
                            point3D = [z, point[0], point[1]]
                            points.append(point3D)


                xyres = CT_F3.metadata["XYresolution"]
                zres  = CT_F3.metadata["Zresolution"]
                resolutions = np.array([zres, xyres, xyres])
                points = np.array(points)*resolutions

                from scipy.spatial import ConvexHull
                hull = ConvexHull(points)
                outline3D = points[hull.vertices]

                # Get embryo 3D Centroid
                centers = []
                for cell in CT_F3.jitcells:
                    center = cell.centers[0]
                    centers.append(center)
                    
                for cell in CT_A12.jitcells:
                    center = cell.centers[0]
                    centers.append(center)

                centers = np.array(centers)
                
                minz = int(np.min(centers[:,0]))
                maxz = int(np.max(centers[:,0]))

                centroid = np.mean(centers*resolutions, axis=0)

                from qlivecell import EmbryoSegmentation, tif_reader_5D
                hyperstack, metadata = tif_reader_5D(path_data)
                channels_seg = np.array([ch_A12, ch_F3, ch_Casp3])
                hyperstack_seg = np.sum(hyperstack[:,:,channels_seg, :, :].astype("int32"), axis=2)

                z_plot = np.rint(hyperstack_seg.shape[1]/2).astype("int64")
                
                if TIME=="96hr": 
                    binth = binths[f]
                else:
                    binth=7.5
                
                print(binth)
                ES = EmbryoSegmentation(
                        hyperstack_seg,
                        ksize=5,
                        ksigma=20,
                        binths=binth,
                        apply_biths_to_zrange_only=False,
                        checkerboard_size=10,
                        num_inter=100,
                        smoothing=20,
                        trange=None,
                        zrange=range(minz, maxz+1),
                        mp_threads=14,
                    )

                ES(hyperstack_seg)
                # ES.plot_segmentation(0, minz + 4)
                # ES.plot_segmentation(0, z_plot)
                # ES.plot_segmentation(0, maxz - 4)

    
                import numpy as np
                from skimage import measure

                contour_points3D = []
                for zid, z in enumerate(range(minz, maxz+1)):
                    contours = measure.find_contours(ES.LS[0][zid], 0.5)
                    contour = []
                    # Select the largest contour as the gastruloid contour
                    for cont in contours:
                        if len(cont)>len(contour):
                            contour = cont
                    for p in contour:
                        contour_points3D.append(np.array([z, p[1], p[0]]))
                contour_points3D = np.array(contour_points3D)
                
                centers_Casp3 = []
                centers_Casp3_F3 = []
                centers_Casp3_A12 = []
                for cell in CT_Casp3.jitcells:
                    centers_Casp3.append(cell.centers[0])
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

                    zz = np.int64(cell.centers[0][0])
                    idx = cell.zs[0].index(zz)
                    if mdiff > 0:
                        a12[idx] += mdiff
                    else: 
                        f3[idx] -= mdiff
    
                    if f3[idx] > a12[idx]:
                        centers_Casp3_F3.append(cell.centers[0])
                    else:
                        centers_Casp3_A12.append(cell.centers[0])

                centers_Casp3 = np.array(centers_Casp3)
                centers_Casp3_F3 = np.array(centers_Casp3_F3)
                centers_Casp3_A12 = np.array(centers_Casp3_A12)

                centers_A12 = []
                for cell in CT_A12.jitcells:
                    centers_A12.append(cell.centers[0])
                centers_A12 = np.array(centers_A12)

                centers_F3 = []
                for cell in CT_F3.jitcells:
                    centers_F3.append(cell.centers[0])
                centers_F3 = np.array(centers_F3)

                contour_points3D = contour_points3D*resolutions
                
                if len(centers_Casp3)!= 0:
                    centers_Casp3 = centers_Casp3*resolutions
                if len(centers_Casp3_F3)!= 0:
                    centers_Casp3_F3 = centers_Casp3_F3*resolutions
                if len(centers_Casp3_A12)!= 0:
                    centers_Casp3_A12 = centers_Casp3_A12*resolutions

                centers_A12 = centers_A12*resolutions
                centers_F3 = centers_F3*resolutions

                if len(centers_Casp3)!=0:
                    dists_Casp3 = compute_dists(np.array(centers_Casp3), np.array(contour_points3D))
                    closests_Casp3_ids = np.argmin(dists_Casp3, axis=1)
                    closests_contour_points_Casp3 = np.array([contour_points3D[i] for i in closests_Casp3_ids])
                else: 
                    closests_Casp3_ids = np.array([])
                    
                if len(centers_Casp3_F3)!=0:
                    dists_Casp3_F3 = compute_dists(np.array(centers_Casp3_F3), np.array(contour_points3D))
                    closests_Casp3_ids_F3 = np.argmin(dists_Casp3_F3, axis=1)
                    closests_contour_points_Casp3_F3 = np.array([contour_points3D[i] for i in closests_Casp3_ids_F3])
                else: 
                    closests_Casp3_ids_F3 = np.array([])
                    
                if len(centers_Casp3_A12)!=0:
                    dists_Casp3_A12 = compute_dists(np.array(centers_Casp3_A12), np.array(contour_points3D))
                    closests_Casp3_ids_A12 = np.argmin(dists_Casp3_A12, axis=1)
                    closests_contour_points_Casp3_A12 = np.array([contour_points3D[i] for i in closests_Casp3_ids_A12])
                else: 
                    closests_Casp3_ids_A12 = np.array([])
                    
                dists_A12 = compute_dists(np.array(centers_A12), np.array(contour_points3D))
                closests_A12_ids = np.argmin(dists_A12, axis=1)
                closests_contour_points_A12 = np.array([contour_points3D[i] for i in closests_A12_ids])

                dists_F3 = compute_dists(np.array(centers_F3), np.array(contour_points3D))
                closests_F3_ids = np.argmin(dists_F3, axis=1)
                closests_contour_points_F3 = np.array([contour_points3D[i] for i in closests_F3_ids])

                dists_contour_Casp3_current = [dists_Casp3[i, closests_Casp3_ids[i]] for i in range(len(centers_Casp3))]
                dists_contour_Casp3_current_F3 = [dists_Casp3_F3[i, closests_Casp3_ids_F3[i]] for i in range(len(centers_Casp3_F3))]
                dists_contour_Casp3_current_A12 = [dists_Casp3_A12[i, closests_Casp3_ids_A12[i]] for i in range(len(centers_Casp3_A12))]

                dists_contour_A12_current = [dists_A12[i, closests_A12_ids[i]] for i in range(len(centers_A12))]
                dists_contour_F3_current = [dists_F3[i, closests_F3_ids[i]] for i in range(len(centers_F3))]
                
                dists_centroid_Casp3_current = [compute_distance_xyz(center, centroid) for center in centers_Casp3]
                dists_centroid_Casp3_current_F3 = [compute_distance_xyz(center, centroid) for center in centers_Casp3_F3]
                dists_centroid_Casp3_current_A12 = [compute_distance_xyz(center, centroid) for center in centers_Casp3_A12]

                dists_centroid_A12_current = [compute_distance_xyz(center, centroid) for center in centers_A12]
                dists_centroid_F3_current = [compute_distance_xyz(center, centroid) for center in centers_F3]

                file_path = path_save_results+embcode
                np.save(file_path+"_dists_contour_Casp3", dists_contour_Casp3_current, allow_pickle=False)
                np.save(file_path+"_dists_contour_Casp3_F3", dists_contour_Casp3_current_F3, allow_pickle=False)
                np.save(file_path+"_dists_contour_Casp3_A12", dists_contour_Casp3_current_A12, allow_pickle=False)

                np.save(file_path+"_dists_contour_A12", dists_contour_A12_current, allow_pickle=False)
                np.save(file_path+"_dists_contour_F3", dists_contour_F3_current, allow_pickle=False)
                
                np.save(file_path+"_dists_centroid_Casp3", dists_centroid_Casp3_current, allow_pickle=False)
                np.save(file_path+"_dists_centroid_Casp3_F3", dists_centroid_Casp3_current_F3, allow_pickle=False)
                np.save(file_path+"_dists_centroid_Casp3_A12", dists_centroid_Casp3_current_A12, allow_pickle=False)

                np.save(file_path+"_dists_centroid_A12", dists_centroid_A12_current, allow_pickle=False)
                np.save(file_path+"_dists_centroid_F3", dists_centroid_F3_current, allow_pickle=False)
