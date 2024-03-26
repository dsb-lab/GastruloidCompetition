### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
TIMES = ["48hr", "60hr", "72hr", "96hr"]
CONDS = ["WT", "KO"]

F3_WT = []
A12_WT = []

F3_KO = []
A12_KO = []

for TIME in TIMES:
    for COND in CONDS:
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2024_03_Casp3/stacks/{}/{}/'.format(TIME, COND)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2024_03_Casp3/ctobjects/{}/{}/'.format(TIME, COND)

        if TIME != "96hr":
            continue
        if COND != "WT":
            continue
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

        channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]

        F3_counts = []
        A12_counts = []

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
            CT_F3.plot_tracking()
            
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

            F3_counts.append(len(CT_F3.jitcells))
            A12_counts.append(len(CT_A12.jitcells))

        if COND == "WT":
            F3_WT.append(F3_counts)
            A12_WT.append(A12_counts)
        elif COND == "KO":
            F3_KO.append(F3_counts)
            A12_KO.append(A12_counts)

pth_save_fig = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/figures/counts/"

import matplotlib.pyplot as plt

time = [48, 60, 72, 96]
fig, ax = plt.subplots(1,2, sharey=True, figsize=(12, 6))

ax[0].set_title("WT")
ax[1].set_title("KO")

ax[0].set_xlabel("time (hr)")
ax[1].set_xlabel("time (hr)")

ax[0].set_ylabel("cell number")

F3_WT_means = np.array([np.mean(data) for data in F3_WT])
F3_WT_stds = np.array([np.std(data) for data in F3_WT])

ax[0].plot(time, F3_WT_means, 'green', label="F3")
ax[0].fill_between(time, F3_WT_means-F3_WT_stds, F3_WT_means+F3_WT_stds, alpha=0.1, color="green")

A12_WT_means = np.array([np.mean(data) for data in A12_WT])
A12_WT_stds = np.array([np.std(data) for data in A12_WT])

ax[0].plot(time, A12_WT_means, 'magenta', label="A12")
ax[0].fill_between(time, A12_WT_means-A12_WT_stds, A12_WT_means+A12_WT_stds, alpha=0.1, color="magenta")

F3_KO_means = np.array([np.mean(data) for data in F3_KO])
F3_KO_stds = np.array([np.std(data) for data in F3_KO])

ax[1].plot(time, F3_KO_means, 'green', label="F3")
ax[1].fill_between(time, F3_KO_means-F3_KO_stds, F3_KO_means+F3_KO_stds, alpha=0.1, color="green")

A12_KO_means = np.array([np.mean(data) for data in A12_KO])
A12_KO_stds = np.array([np.std(data) for data in A12_KO])

ax[1].plot(time, A12_KO_means, 'magenta', label="A12")
ax[1].fill_between(time, A12_KO_means-A12_KO_stds, A12_KO_means+A12_KO_stds, alpha=0.1, color="magenta")

plt.legend()
plt.show()

time = [48, 60, 72, 96]
fig, ax = plt.subplots(1,2, figsize=(12, 6))


ax[0].set_title("F3")
ax[1].set_title("A12")

ax[0].set_xlabel("time (hr)")
ax[1].set_xlabel("time (hr)")

ax[0].set_ylabel("cell number")
ax[1].set_ylabel("cell number")

ax[0].plot(time, F3_WT_means, 'green', label="F3 - WT")
ax[0].fill_between(time, F3_WT_means-F3_WT_stds, F3_WT_means+F3_WT_stds, alpha=0.1, color="green")

ax[0].plot(time, F3_KO_means, 'cyan', label="F3 - KO")
ax[0].fill_between(time, F3_KO_means-F3_KO_stds, F3_KO_means+F3_KO_stds, alpha=0.1, color="cyan")

ax[1].plot(time, A12_WT_means, 'magenta', label="A12 - WT")
ax[1].fill_between(time, A12_WT_means-A12_WT_stds, A12_WT_means+A12_WT_stds, alpha=0.1, color="magenta")

ax[1].plot(time, A12_KO_means, 'red', label="A12 - KO")
ax[1].fill_between(time, A12_KO_means-A12_KO_stds, A12_KO_means+A12_KO_stds, alpha=0.1, color="red")

ax[0].legend()
ax[1].legend()

plt.show()