### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, get_file_names, correct_path
import numpy as np
from scipy import stats

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')


### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/96hr/WT/'
path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/96hr/WT/'

try: 
    files = get_file_names(path_save_dir)
except: 
    import os
    os.mkdir(path_save_dir)

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

def compute_distance_xyz(x1, x2, y1, y2, z1, z2):
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
    z1 : number
        z coordinate of point 1
    z2 : number
        z coordinate of point 2
    Returns
    -------
    dist : number
        euclidean distance between points (x1, y1, z1) and (x2, y2, z2)
    """
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return dist

### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data_dir)

channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
if "96hr" in path_data_dir:
    channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

cells = []
masks_fluo_values = []
CENTERS = []
DISTSF3 = []
DISTSA12 = []
DISTSCasp3 = []

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


    # CT_F3.load()
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

    # CT_A12.load()
    # CT_A12.plot_tracking()
    
    ch_Casp3 = channel_names.index("Casp3")

    batch_args = {
        'name_format':"ch"+str(ch_Casp3)+"_{}_mid",
        'extension':".tif",
    }
    plot_args = {
        'plot_layout': (1,1),
        'plot_overlap': 1,
        'masks_cmap': 'tab10',
        # 'plot_stack_dims': (256, 256), 
        'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
        'channels':[1,0,ch_Casp3],
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
    for lab in CT_Casp3.unique_labels:
        CT_Casp3._del_cell(lab)
        
    CT_Casp3.plot_tracking(plot_args=plot_args)

    # import numpy as np
    # from scipy import stats
    # labs_rem = []
    # for cell in CT_Casp3.jitcells:
    #     cells.append(cell)
    #     zc = int(cell.centers[0][0])
    #     zcid = cell.zs[0].index(zc)
    #     center2D = cell.centers[0][1:]

    #     mask = cell.masks[0][zcid]
    #     stack = CT_Casp3.hyperstack[0, zc, ch_Casp3]
    #     masks_fluo_values.append(stack[mask[:, 0], mask[:, 1]])
        
    #     dists = []
    #     vals = []
    #     for point in mask:
    #         dist = compute_distance_xy(center2D[0], point[0], center2D[1], point[1])
    #         dists.append(dist)
    #         val = stack[point[1], point[0]]
    #         vals.append(val)
    #     dists = np.array(dists)
    #     vals  = np.array(vals)
    #     idxs = np.where(dists < 8.0)[0]
        
    #     dists = dists[idxs]
    #     vals = vals[idxs]
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(dists,vals)
    #     if slope < 0:
    #         labs_rem.append(cell.label)
    
    
    # for lab in labs_rem:
    #     CT_Casp3._del_cell(lab)
    
    # CT_Casp3.plot_tracking(plot_args=plot_args)
    # CT_Casp3.update_labels()
    
    # centers = []
    # for cell in CT_F3.jitcells:
    #     centers.append(cell.centers[0])

    # for cell in CT_A12.jitcells:
    #     centers.append(cell.centers[0])

    # centroid = np.mean(centers, axis=0)

    # distsF3 = []
    # for cell in CT_F3.jitcells:
    #     dist = compute_distance_xyz(centroid[0], cell.centers[0][0], centroid[1], cell.centers[0][1], centroid[2], cell.centers[0][2])
    #     distsF3.append(dist)

    # distsA12 = []
    # for cell in CT_A12.jitcells:
    #     dist = compute_distance_xyz(centroid[0], cell.centers[0][0], centroid[1], cell.centers[0][1], centroid[2], cell.centers[0][2])
    #     distsA12.append(dist)


    # distsCasp3 = []
    # for cell in CT_Casp3.jitcells:
    #     dist = compute_distance_xyz(centroid[0], cell.centers[0][0], centroid[1], cell.centers[0][1], centroid[2], cell.centers[0][2])
    #     distsCasp3.append(dist)

    # CENTERS.append(centers)
    # DISTSF3.append(distsF3)
    # DISTSA12.append(distsA12)
    # DISTSCasp3.append(distsCasp3)




# len(cells)
# len(masks_fluo_values)

# import numpy as np

# mean_vals = np.array([np.mean(vals) for vals in masks_fluo_values])
# sum_vals = np.array([np.sum(vals) for vals in masks_fluo_values])
# std_vals = np.array([np.std(vals) for vals in masks_fluo_values])
# areas = np.array([len(vals) for vals in masks_fluo_values]) * (CT_Casp3.CT_info.xyresolution**2)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# # ax.hist(mean_vals, bins=1000)
# ax.scatter(areas, mean_vals, s=5)
# ax.hlines(3, areas.min(), areas.max())
# # ax.vlines(65*CT_Casp3.CT_info.xyresolution, mean_vals.min(), mean_vals.max())
# ax.set_yscale("log")
# ax.set_xscale("log")
# plt.show()

# from scipy import stats

# for cell in CT_Casp3.jitcells:
#     center2D = cell.centers[0][1:]
#     zc = int(cell.centers[0][0])
#     zcid = cell.zs[0].index(zc)
#     mask = cell.masks[0][zcid]

#     outline = cell.outlines[0][zcid]
#     stack = CT_Casp3.hyperstack[0, zc, 3]

#     dists = []
#     vals = []
#     for point in mask:
#         dist = compute_distance_xy(center2D[0], point[0], center2D[1], point[1])
#         dists.append(dist)
#         val = stack[point[1], point[0]]
#         vals.append(val)
        

#     r = 20
#     import matplotlib.pyplot as plt
    
#     dists = np.array(dists)
#     vals  = np.array(vals)
#     idxs = np.where(dists < 8.0)[0]
    
#     dists = dists[idxs]
#     vals = vals[idxs]
#     slope, intercept, r_value, p_value, std_err = stats.linregress(dists,vals)

#     if slope > 0:
#         fig, ax = plt.subplots(1,2,figsize=(10, 5))
#         ax[0].imshow(stack)
#         ax[0].scatter(outline[:,0], outline[:,1], c="w", s=5)
#         ax[0].scatter([center2D[0]], [center2D[1]], c="w", s=5)
#         ax[0].set_xlim(center2D[0]-r,center2D[0]+r)
#         ax[0].set_ylim( center2D[1]-r,center2D[1]+r)
#         ax[0].set_title("cell label {}".format(cell.label))
    
#         ax[1].scatter(dists, vals, s=5)
#         ax[1].plot(dists, dists*slope + intercept)
#         ax[1].set_ylabel("pixel intensity")
#         ax[1].set_xlabel("distance to center")
#         plt.show()
#     elif slope > -0.5:
#         fig, ax = plt.subplots(1,2,figsize=(10, 5))
#         ax[0].imshow(stack)
#         ax[0].scatter(outline[:,0], outline[:,1], c="w", s=5)
#         ax[0].scatter([center2D[0]], [center2D[1]], c="w", s=5)
#         ax[0].set_xlim(center2D[0]-r,center2D[0]+r)
#         ax[0].set_ylim( center2D[1]-r,center2D[1]+r)
#         ax[0].set_title("cell label {}".format(cell.label))
    
#         ax[1].scatter(dists, vals, s=5)
#         ax[1].plot(dists, dists*slope + intercept)
#         ax[1].set_ylabel("pixel intensity")
#         ax[1].set_xlabel("distance to center")
#         plt.show()
#     else:
        
#         continue

# for f, file in enumerate(files):
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     ax.hist(DISTSF3[f], density=True, color="green", alpha=0.5)
#     ax.hist(DISTSA12[f], density=True, color="magenta", alpha=0.5)
#     ax.hist(DISTSCasp3[f], density=True, color="yellow", alpha=0.5)
#     plt.show()

