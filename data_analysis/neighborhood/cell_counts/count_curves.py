### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift
import numpy as np
import matplotlib.pyplot as plt


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

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
experiment = "2023_11_17_Casp3"
TIMES = ["48hr", "72hr", "96hr"]
CONDS = ["WT", "KO"]

F3_WT = []
A12_WT = []
Casp3_F3_WT =  []
Casp3_A12_WT =  []

F3_KO = []
A12_KO = []
Casp3_F3_KO =  []
Casp3_A12_KO =  []

for TIME in TIMES:
    for COND in CONDS:
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/stacks/{}/{}/'.format(experiment, TIME, COND)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/ctobjects/{}/{}/'.format(experiment, TIME, COND)

        try: 
            files = get_file_names(path_save_dir)
        except: 
            import os
            os.mkdir(path_save_dir)
            
        ### GET FULL FILE NAME AND FILE CODE ###
        files = get_file_names(path_data_dir)

        CENTERS = []
        FATES = []
        LABS = []

        # channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
        channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
        if "96hr" in path_data_dir:
            channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

        F3_counts = []
        Casp3_F3_counts = []
        A12_counts = []
        Casp3_A12_counts = []

        for f, file in enumerate(files):
            path_data = path_data_dir+file
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
            path_save = path_save_dir+embcode
            try: 
                files = get_file_names(path_save)
            except: 
                import os
                os.mkdir(path_save)

            ### DEFINE ARGUMENTS ###
            segmentation_args={
                'method': 'stardist2D', 
                'model': model, 
                'blur': [2,1], 
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
                'name_format':"ch"+str(ch)+"_{}_early",
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

            plot_args = {
                'plot_layout': (1,1),
                'plot_overlap': 1,
                'masks_cmap': 'tab10',
                'plot_stack_dims': (512, 512), 
                'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
                'channels':[1,0,3],
                'min_outline_length':75,
            }

            CT_Casp3.plot_tracking(plot_args=plot_args)

            F3_counts.append(len(CT_F3.jitcells))
            A12_counts.append(len(CT_A12.jitcells))

            import numpy as np

            ### CORRECT DISTRIBUTIONS ###

            centers = []
            areas = []

            F3_dist = []
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

            fates = []
            centers = []
            labels = []
            for cell in CT_F3.jitcells:
                fates.append(0)
                centers.append(cell.centers[0]*[zres, xyres, xyres])
                labels.append(cell.label)
            for cell in CT_A12.jitcells:
                fates.append(1)
                centers.append(cell.centers[0]*[zres, xyres, xyres])
                labels.append(cell.label)

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
                labels.append(cell.label)

                zz = np.int64(cell.centers[0][0])
                idx = cell.zs[0].index(zz)
                if f3[idx] > a12[idx]:
                    fates.append(2)
                    Casp3_F3+=1
                else:
                    fates.append(3)
                    Casp3_A12+=1
            
            Casp3_F3_counts.append(Casp3_F3)
            Casp3_A12_counts.append(Casp3_A12)

        if COND == "WT":
            F3_WT.append(F3_counts)
            A12_WT.append(A12_counts)
            Casp3_F3_WT.append(Casp3_F3_counts)
            Casp3_A12_WT.append(Casp3_A12_counts)

        elif COND == "KO":
            F3_KO.append(F3_counts)
            A12_KO.append(A12_counts)
            Casp3_F3_KO.append(Casp3_F3_counts)
            Casp3_A12_KO.append(Casp3_A12_counts)

pth_save_fig = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/figures/counts/"

import matplotlib.pyplot as plt

time = [48, 72, 96]
fig, ax = plt.subplots(2,2, sharex=True, figsize=(12, 6))

ax[0,0].set_title("WT")
ax[0,1].set_title("KO")

ax[1,0].set_xlabel("time (hr)")
ax[1,1].set_xlabel("time (hr)")

ax[0,0].set_ylabel("cell number")
ax[1,0].set_ylabel("Apo ration")

### WT ###

F3_WT_means = np.array([np.mean(data) for data in F3_WT])
F3_WT_stds = np.array([np.std(data) for data in F3_WT])

ax[0,0].plot(time, F3_WT_means, 'green', label="F3")
ax[0,0].fill_between(time, F3_WT_means-F3_WT_stds, F3_WT_means+F3_WT_stds, alpha=0.1, color="green")

A12_WT_means = np.array([np.mean(data) for data in A12_WT])
A12_WT_stds = np.array([np.std(data) for data in A12_WT])

ax[0,0].plot(time, A12_WT_means, 'magenta', label="A12")
ax[0,0].fill_between(time, A12_WT_means-A12_WT_stds, A12_WT_means+A12_WT_stds, alpha=0.1, color="magenta")

DATA = [np.array(Casp3_F3_WT[n])/F3_WT[n] for n in range(len(F3_WT))]
Casp3_F3_WT_means = np.array([np.mean(data) for data in DATA])
Casp3_F3_WT_stds = np.array([np.std(data) for data in DATA])

ax[1,0].plot(time, Casp3_F3_WT_means, 'cyan', label="F3 - apo")
ax[1,0].fill_between(time, Casp3_F3_WT_means-Casp3_F3_WT_stds, Casp3_F3_WT_means+Casp3_F3_WT_stds, alpha=0.1, color="cyan")

DATA = [np.array(Casp3_A12_WT[n])/A12_WT[n] for n in range(len(A12_WT))]
Casp3_A12_WT_means = np.array([np.mean(data) for data in DATA])
Casp3_A12_WT_stds = np.array([np.std(data) for data in DATA])

ax[1,0].plot(time, Casp3_A12_WT_means, 'red', label="A12 - apo")
ax[1,0].fill_between(time, Casp3_A12_WT_means-Casp3_A12_WT_stds, Casp3_A12_WT_means+Casp3_A12_WT_stds, alpha=0.1, color="red")


### KO ###

F3_KO_means = np.array([np.mean(data) for data in F3_KO])
F3_KO_stds = np.array([np.std(data) for data in F3_KO])

ax[0,1].plot(time, F3_KO_means, 'green', label="F3")
ax[0,1].fill_between(time, F3_KO_means-F3_KO_stds, F3_KO_means+F3_KO_stds, alpha=0.1, color="green")

A12_KO_means = np.array([np.mean(data) for data in A12_KO])
A12_KO_stds = np.array([np.std(data) for data in A12_KO])

ax[0,1].plot(time, A12_KO_means, 'magenta', label="A12")
ax[0,1].fill_between(time, A12_KO_means-A12_KO_stds, A12_KO_means+A12_KO_stds, alpha=0.1, color="magenta")

DATA = [np.array(Casp3_F3_KO[n])/F3_KO[n] for n in range(len(F3_KO))]
Casp3_F3_KO_means = np.array([np.mean(data) for data in DATA])
Casp3_F3_KO_stds = np.array([np.std(data) for data in DATA])

ax[1,1].plot(time, Casp3_F3_KO_means, 'cyan', label="F3 - apo")
ax[1,1].fill_between(time, Casp3_F3_KO_means-Casp3_F3_KO_stds, Casp3_F3_KO_means+Casp3_F3_KO_stds, alpha=0.1, color="cyan")

DATA = [np.array(Casp3_A12_KO[n])/A12_KO[n] for n in range(len(A12_KO))]
Casp3_A12_KO_means = np.array([np.mean(data) for data in DATA])
Casp3_A12_KO_stds = np.array([np.std(data) for data in DATA])

ax[1,1].plot(time, Casp3_A12_KO_means, 'red', label="A12 - apo")
ax[1,1].fill_between(time, Casp3_A12_KO_means-Casp3_A12_KO_stds, Casp3_A12_KO_means+Casp3_A12_KO_stds, alpha=0.1, color="red")

ax[1,1].legend()
ax[0,1].legend()

plt.show()

