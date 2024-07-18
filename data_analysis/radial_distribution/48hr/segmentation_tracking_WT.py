### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, get_file_names, correct_path
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

apo_stages = ["early", "mid", "late"]
for apo_stage in apo_stages:
    ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
    path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/48hr/WT/'
    path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/48hr/WT/'
    path_save_results='/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/radial_distribution/{}_apoptosis/48hr/WT/'.format(apo_stage)

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

    dists_contour_Casp3 = []
    dists_contour_A12 = []
    dists_contour_F3 = []
    dists_centroid_Casp3 = []
    dists_centroid_A12 = []
    dists_centroid_F3 = []

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
            # 'plot_stack_dims': (256, 256), 
            'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
            'channels':[ch_F3],
            'min_outline_length':75,
        }
        
        chans = [ch_F3]
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
            # 'plot_stack_dims': (256, 256), 
            'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
            'channels':[ch_A12],
            'min_outline_length':75,
        }

        chans = [ch_A12]
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
            # 'plot_stack_dims': (256, 256), 
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
        # CT_Casp3.plot_tracking(plot_args=plot_args)

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

        centers = np.array(centers)*resolutions
        centroid = np.mean(centers, axis=0)

        from embdevtools import EmbryoSegmentation, tif_reader_5D
        hyperstack, metadata = tif_reader_5D(path_data)
        channels_seg = np.array([ch_A12, ch_F3, ch_Casp3])
        hyperstack_seg = np.sum(hyperstack[:,:,channels_seg, :, :].astype("int32"), axis=2)

        z_plot = np.rint(hyperstack_seg.shape[1]/2).astype("int64")
        ES = EmbryoSegmentation(
                hyperstack_seg,
                ksize=5,
                ksigma=30,
                binths=7,
                apply_biths_to_zrange_only=False,
                checkerboard_size=10,
                num_inter=100,
                smoothing=20,
                trange=None,
                zrange=range(minz, maxz+1),
                mp_threads=None,
            )

        ES(hyperstack_seg)
        # ES.plot_segmentation(0, minz + 2)
        # ES.plot_segmentation(0, z_plot)
        # ES.plot_segmentation(0, maxz - 2)

        
        import numpy as np
        from skimage import measure

        contour_points3D = []
        for zid, z in enumerate(range(minz, maxz+1)):
            contours = measure.find_contours(ES.LS[0][z], 0.5)
            contour = []
            for cont in contours:
                if len(cont)>len(contour):
                    contour = cont
            for p in contour:
                contour_points3D.append(np.array([z, p[1], p[0]]))
        contour_points3D = np.array(contour_points3D)

        print("got contours")
        
        centers_Casp3 = []
        for cell in CT_Casp3.jitcells:
            centers_Casp3.append(cell.centers[0])
        centers_Casp3 = np.array(centers_Casp3)

        centers_A12 = []
        for cell in CT_A12.jitcells:
            centers_A12.append(cell.centers[0])
        centers_A12 = np.array(centers_A12)

        centers_F3 = []
        for cell in CT_F3.jitcells:
            centers_F3.append(cell.centers[0])
        centers_F3 = np.array(centers_F3)

        contour_points3D = contour_points3D*resolutions
        centers_Casp3 = centers_Casp3*resolutions
        centers_A12 = centers_A12*resolutions
        centers_F3 = centers_F3*resolutions

        print("got centers")

        dists_Casp3 = compute_dists(np.array(centers_Casp3), np.array(contour_points3D))
        closests_Casp3_ids = np.argmin(dists_Casp3, axis=1)
        closests_contour_points_Casp3 = np.array([contour_points3D[i] for i in closests_Casp3_ids])

        dists_A12 = compute_dists(np.array(centers_A12), np.array(contour_points3D))
        closests_A12_ids = np.argmin(dists_A12, axis=1)
        closests_contour_points_A12 = np.array([contour_points3D[i] for i in closests_A12_ids])

        dists_F3 = compute_dists(np.array(centers_F3), np.array(contour_points3D))
        closests_F3_ids = np.argmin(dists_F3, axis=1)
        closests_contour_points_F3 = np.array([contour_points3D[i] for i in closests_F3_ids])

        dists_contour_Casp3_current = [dists_Casp3[i, closests_Casp3_ids[i]] for i in range(len(centers_Casp3))]
        dists_contour_A12_current = [dists_A12[i, closests_A12_ids[i]] for i in range(len(centers_A12))]
        dists_contour_F3_current = [dists_F3[i, closests_F3_ids[i]] for i in range(len(centers_F3))]
        dists_centroid_Casp3_current = [compute_distance_xyz(center, centroid) for center in centers_Casp3]
        dists_centroid_A12_current = [compute_distance_xyz(center, centroid) for center in centers_A12]
        dists_centroid_F3_current = [compute_distance_xyz(center, centroid) for center in centers_F3]

        file_path = path_save_results+embcode
        np.save(file_path+"_dists_contour_Casp3", dists_contour_Casp3_current, allow_pickle=False)
        np.save(file_path+"_dists_contour_A12", dists_contour_A12_current, allow_pickle=False)
        np.save(file_path+"_dists_contour_F3", dists_contour_F3_current, allow_pickle=False)
        
        np.save(file_path+"_dists_centroid_Casp3", dists_centroid_Casp3_current, allow_pickle=False)
        np.save(file_path+"_dists_centroid_A12", dists_centroid_A12_current, allow_pickle=False)
        np.save(file_path+"_dists_centroid_F3", dists_centroid_F3_current, allow_pickle=False)
        
        dists_contour_Casp3 = [*dists_contour_Casp3, *dists_contour_Casp3_current]
        dists_contour_A12 = [*dists_contour_A12, *dists_contour_A12_current]
        dists_contour_F3 = [*dists_contour_F3, *dists_contour_F3_current]
        
        dists_centroid_Casp3 = [*dists_centroid_Casp3, *dists_centroid_Casp3_current]
        dists_centroid_A12 = [*dists_centroid_A12, *dists_centroid_A12_current]
        dists_centroid_F3 = [*dists_centroid_F3, *dists_centroid_F3_current]


    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1,2, figsize=(10,5))

    # ax[0].hist(dists_contour_F3, color="green", alpha=0.5, bins=50)
    # ax[0].hist(dists_contour_A12, color="magenta", alpha=0.5, bins=50)
    # # ax[0].hist(dists_contour_Casp3, color="yellow")
    # ax[0].set_xlabel("distance to closest embryo border")
    # ax[0].set_yticks([])
    # ax[1].hist(dists_centroid_F3, color="green", alpha=0.5, bins=50)
    # ax[1].hist(dists_centroid_A12, color="magenta", alpha=0.5, bins=50)
    # # ax[1].hist(dists_centroid_Casp3, color="yellow")
    # ax[1].set_xlim(0,100)
    # ax[1].set_xlabel("distance to embryo centroid")
    # ax[1].set_yticks([])
    # plt.show()

