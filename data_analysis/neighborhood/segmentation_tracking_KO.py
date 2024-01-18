### LOAD PACKAGE ###
from embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/72hr/KO/'
path_save='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/'

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)
### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data)

CENTERS = []
FATES = []
LABS = []

for f, file in enumerate(files):
    file, embcode = get_file_embcode(path_data, file, allow_file_fragment=False, returnfiles=False)


    ### LOAD HYPERSTACKS ###
    channel = 0
    IMGS_A12, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
    ### LOAD HYPERSTACKS ###
    channel = 1
    IMGS_F3, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
    ### LOAD HYPERSTACKS ###
    channel = 3
    IMGS_Casp3, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)
    ### LOAD HYPERSTACKS ###
    channel = 2
    IMGS_DAPI, xyres, zres = read_img_with_resolution(path_data+file, stack=True, channel=channel)


    ### LOAD STARDIST MODEL ###
    from stardist.models import StarDist2D
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    ### DEFINE ARGUMENTS ###
    segmentation_args={
        'method': 'stardist2D', 
        'model': model, 
        'blur': [5,1], 
        # 'scale': 3
    }
            
    concatenation3D_args = {
        'distance_th_z': 3.0, 
        'relative_overlap':False, 
        'use_full_matrix_to_compute_overlap':True, 
        'z_neighborhood':2, 
        'overlap_gradient_th':0.3, 
        'min_cell_planes': 3,
    }

    tracking_args = {
        'time_step': 5, # minutes
        'method': 'greedy', 
        'z_th':5, 
        'dist_th' : 10.0,
    }

    plot_args = {
        'plot_layout': (1,1),
        'plot_overlap': 1,
        'masks_cmap': 'tab10',
        'plot_stack_dims': (512, 512), 
        'plot_centers':[True, True]
    }

    error_correction_args = {
        'backup_steps': 10,
        'line_builder_mode': 'lasso',
    }


    # ### PLOTTING ###
    import numpy as np
    _IMGS_A12 = np.asarray([[255*(IMG/IMG.max()) for IMG in IMGS_A12[0]]]).astype('uint8')
    _IMGS_F3 = np.asarray([[255*(IMG/IMG.max()) for IMG in IMGS_F3[0]]]).astype('uint8')
    _IMGS_Casp3 = np.asarray([[255*(IMG/IMG.max()) for IMG in IMGS_Casp3[0]]]).astype('uint8')

    IMGS_plot = construct_RGB(R=_IMGS_A12, G=_IMGS_F3, B=_IMGS_Casp3)

    ### CREATE CELLTRACKING CLASS ###
    CT_Casp3 = CellTracking(
        IMGS_Casp3, 
        path_save, 
        embcode+"ch_%d" %(channel+1), 
        xyresolution=xyres, 
        zresolution=zres,
        segmentation_args=segmentation_args,
        concatenation3D_args=concatenation3D_args,
        tracking_args = tracking_args, 
        error_correction_args=error_correction_args,    
        plot_args = plot_args,
    )

    ### RUN SEGMENTATION AND TRACKING ###
    CT_Casp3.run()
    CT_Casp3.plot_tracking(plot_args, stacks_for_plotting=IMGS_plot)



    segmentation_args={
        'method': 'stardist2D', 
        'model': model, 
        'blur': None, 
    }
            
    ### CREATE CELLTRACKING CLASS ###
    CT_A12 = CellTracking(
        IMGS_A12, 
        path_save, 
        embcode+"ch_%d" %(channel+1), 
        xyresolution=xyres, 
        zresolution=zres,
        segmentation_args=segmentation_args,
        concatenation3D_args=concatenation3D_args,
        tracking_args = tracking_args, 
        error_correction_args=error_correction_args,    
        plot_args = plot_args,
    )


    ### RUN SEGMENTATION AND TRACKING ###
    CT_A12.run()
    CT_A12.plot_tracking(plot_args, stacks_for_plotting=IMGS_plot)

    ### CREATE CELLTRACKING CLASS ###
    CT_F3 = CellTracking(
        IMGS_F3, 
        path_save, 
        embcode+"ch_%d" %(channel+1), 
        xyresolution=xyres, 
        zresolution=zres,
        segmentation_args=segmentation_args,
        concatenation3D_args=concatenation3D_args,
        tracking_args = tracking_args, 
        error_correction_args=error_correction_args,    
        plot_args = plot_args,
    )


    ### RUN SEGMENTATION AND TRACKING ###
    CT_F3.run()
    CT_F3.plot_tracking(plot_args, stacks_for_plotting=IMGS_plot)

    from embdevtools.celltrack.core.tools.cell_tools import remove_small_cells

    diam_th = 5.6 / xyres
    area_th = np.pi * (diam_th/2)**2 
    remove_small_cells(CT_A12.jitcells, area_th, CT_A12._del_cell, CT_A12.update_labels)
    remove_small_cells(CT_F3.jitcells, area_th, CT_F3._del_cell, CT_F3.update_labels)

    import numpy as np

    cells = [cell for sublist in [CT_A12.jitcells, CT_F3.jitcells] for cell in sublist]
    fates = [s for s, sublist in enumerate([CT_A12.jitcells, CT_F3.jitcells]) for cell in sublist]

    centers = []
    labs = []

    for c, cell in enumerate(cells):
        centers.append(cell.centers[0]*[zres, xyres, xyres])
        if c >= len(CT_A12.jitcells):
            labs.append(cell.label + CT_A12.max_label + 1)
        else:
            labs.append(cell.label)

    max_pre = np.max(labs)
    for cell in CT_Casp3.jitcells:
        cells.append(cell)
        casp3 = []
        a12 = []
        f3 = []
        for zid, z in enumerate(cell.zs[0]):
            mask = cell.masks[0][zid]
            casp3.append(np.mean(IMGS_Casp3[0][z][mask[:,1], mask[:,0]]))
            a12.append(np.mean(IMGS_A12[0][z][mask[:,1], mask[:,0]]))
            f3.append(np.mean(IMGS_F3[0][z][mask[:,1], mask[:,0]]))

        # idx = np.argmax(casp3)
        zz = np.int64(cell.centers[0][0])
        idx = cell.zs[0].index(zz)
        if f3[idx] > a12[idx]:
            fates.append(2)
        else:
            fates.append(3)
        
        centers.append(cell.centers[0]*[zres, xyres, xyres])
        labs.append(cell.label + max_pre + 1)


    centers = np.array(centers)
    labs = np.array(labs)

    CENTERS.append(centers)
    LABS.append(labs)
    FATES.append(fates)
    CASP3 = []
    A12 = []
    F3 = []
    na12 = 0
    nf3 = 0
    labs = []
    for cell in CT_Casp3.jitcells:
        casp3 = []
        a12 = []
        f3 = []
        for zid, z in enumerate(cell.zs[0]):
            mask = cell.masks[0][zid]
            casp3.append(np.mean(_IMGS_Casp3[0][z][mask[:,1], mask[:,0]]))
            a12.append(np.mean(_IMGS_A12[0][z][mask[:,1], mask[:,0]]))
            f3.append(np.mean(_IMGS_F3[0][z][mask[:,1], mask[:,0]]))

        # idx = np.argmax(casp3)
        zz = np.int64(cell.centers[0][0])
        idx = cell.zs[0].index(zz)
        if f3[idx] > a12[idx]:
            nf3 +=1
        else:
            na12 +=1
            labs.append(cell.label)
        CASP3.append(casp3[idx])
        A12.append(a12[idx])
        F3.append(f3[idx])

    print("A12", na12 / (len(CT_A12.jitcells)+na12))
    print("F3", nf3 / (len(CT_F3.jitcells)+nf3))


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

tris =  []
NEIGHS = []
for f in range(len(files)):
    centers = CENTERS[f]
    tri = Delaunay(centers)
    tris.append(tri)

    neighs = []
    for p, point in enumerate(centers):
        neighs_p = find_neighbors(p, tri)
        neighs.append(neighs_p)

    NEIGHS.append(neighs)


neighs_fates_A12_sum = np.zeros((len(files), 2))
neighs_fates_F3_sum = np.zeros((len(files), 2))
neighs_fates_Casp3_A12_sum = np.zeros((len(files), 2))
neighs_fates_Casp3_F3_sum = np.zeros((len(files), 2))
dist_th = 12 #microns
dist_th_near = 5
for f in range(len(files)):
    centers = CENTERS[f]
    neighs = NEIGHS[f]
    fates = FATES[f]
    labs = LABS[f]

    true_neighs = []

    for p, neigh_p in enumerate(neighs):
        true_neigh_p = []
        for neigh in neigh_p:
            dist = np.linalg.norm(centers[p]-centers[neigh])
            if dist < dist_th:
                if dist > dist_th_near:
                    true_neigh_p.append(neigh)
        true_neighs.append(true_neigh_p)

    neighs_labs = []
    neighs_fates = []
    for p, neigh_p in enumerate(true_neighs):
        lbs = []
        fts = []
        for neigh in neigh_p:
            lbs.append(labs[neigh])
            fts.append(fates[neigh])
        neighs_labs.append(lbs)
        neighs_fates.append(fts)


    neighs_n = [len(neighs_p) for neighs_p in true_neighs]
    neighs_n_A12 = [n for i,n in enumerate(neighs_n) if fates[i] == 0]
    neighs_n_F3 = [n for i,n in enumerate(neighs_n) if fates[i] == 1]
    neighs_n_Casp3 = [n for i,n in enumerate(neighs_n) if fates[i] > 1]

    y = [np.mean(x) for x in [neighs_n_A12, neighs_n_F3, neighs_n_Casp3]]
    print(y)
    yerr = [np.std(x) for x in [neighs_n_A12, neighs_n_F3, neighs_n_Casp3]]
    print(yerr)
    import matplotlib.pyplot as plt
    # plt.bar([1,2,3], y, tick_label=["A12", "F3", "Casp3"], color=["magenta", "green", "yellow"], yerr=yerr, capsize=6)
    # plt.ylabel("# of neighbors")
    # plt.show()

    neighs_fates_A12 = [n for i,n in enumerate(neighs_fates) if fates[i] == 0]
    neighs_fates_F3 = [n for i,n in enumerate(neighs_fates) if fates[i] == 1]
    neighs_fates_Casp3_F3 = [n for i,n in enumerate(neighs_fates) if fates[i] == 2]
    neighs_fates_Casp3_A12 = [n for i,n in enumerate(neighs_fates) if fates[i] == 3]


    for n_fates in neighs_fates_A12:
        for _f in n_fates:
            if _f in [0]:
                neighs_fates_A12_sum[f,0] += 1
            elif _f in [1]:
                neighs_fates_A12_sum[f,1] += 1

    for n_fates in neighs_fates_F3:
        for _f in n_fates:
            if _f in [0]:
                neighs_fates_F3_sum[f,0] += 1
            elif _f in [1]:
                neighs_fates_F3_sum[f,1] += 1

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

bot = [np.mean(neighs_fates_A12_sum[:,0]), np.mean(neighs_fates_Casp3_A12_sum[:,0]), np.mean(neighs_fates_F3_sum[:,0]), np.mean(neighs_fates_Casp3_F3_sum[:,0])]
bot_std = [np.std(neighs_fates_A12_sum[:,0]), np.std(neighs_fates_Casp3_A12_sum[:,0]), np.std(neighs_fates_F3_sum[:,0]), np.std(neighs_fates_Casp3_F3_sum[:,0])]

plt.bar([1,2,3,4], bot, color="magenta", yerr=bot_std, capsize=6)
top = [np.mean(neighs_fates_A12_sum[:,1]), np.mean(neighs_fates_Casp3_A12_sum[:,1]), np.mean(neighs_fates_F3_sum[:,1]), np.mean(neighs_fates_Casp3_F3_sum[:,1])]
top_std = [np.std(neighs_fates_A12_sum[:,1]), np.std(neighs_fates_Casp3_A12_sum[:,1]), np.std(neighs_fates_F3_sum[:,1]), np.std(neighs_fates_Casp3_F3_sum[:,1])]
plt.bar([1,2,3,4], top, bottom =bot, tick_label=["A12", "Casp3 - A12", "F3", "Casp3 - F3"], color="green", yerr=top_std, capsize=6)
plt.ylabel("percentage of neighbors")
plt.show()

if True:
    print("A12", np.mean(neighs_fates_A12_sum, axis=0))
    print("Casp3 - A12", np.mean(neighs_fates_Casp3_A12_sum, axis=0))
    print("F3", np.mean(neighs_fates_F3_sum, axis=0))
    print("Casp3 - F3", np.mean(neighs_fates_Casp3_F3_sum, axis=0))

