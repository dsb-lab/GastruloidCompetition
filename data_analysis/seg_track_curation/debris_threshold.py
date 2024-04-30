### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB
from embdevtools import remove_small_cells, plot_cell_sizes, correct_path


experiment_code = '2024_03_Casp3'

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
TIMES = ["48hr", "60hr", "72hr", "96hr"]
CONDS = ["WT", "KO"]
path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/figures/{}".format(experiment_code)
F3_WT_total = []
A12_WT_total = []

F3_KO_total = []
A12_KO_total = []

F3_WT_debris = []
A12_WT_debris = []

F3_KO_debris = []
A12_KO_debris = []

F3_WT_cells = []
A12_WT_cells = []

F3_KO_cells = []
A12_KO_cells = []

for TIME in TIMES:
    path_figures_time = "{}{}/".format(path_figures, TIME)
    try: 
        files = get_file_names(path_figures_time)
    except: 
        import os
        os.mkdir(path_figures_time)
            
    for COND in CONDS:
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/stacks/{}/{}/'.format(experiment_code, TIME, COND)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/ctobjects/{}/{}/'.format(experiment_code,TIME, COND)


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


        total_cells = []
        true_cells  = []
        debris = []
        
        bws = [25, 25, 25, 25, 25]
        bins = [50, 50, 50, 50, 50]
        for f, file in enumerate(files):
            path_data = path_data_dir+file
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
            path_save = correct_path(path_save_dir+embcode)

            ### DEFINE ARGUMENTS ###
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
                'channels':[ch]
            }
            
            chans = [ch]
            for _ch in range(len(channel_names)):
                if _ch not in chans:
                    chans.append(_ch)
            CT_F3 = CellTracking(
                path_data,
                path_save,
                error_correction_args=error_correction_args,
                plot_args=plot_args,
                batch_args=batch_args,
                channels=chans
            )

            CT_F3.load()
            # CT_F3.plot_tracking()
            total_cells.append(len(CT_F3.jitcells))
            plot_cell_sizes(CT_F3, bw=bws[f], bins=bins[f], xlim=(0,200))
            remove_small_cells(CT_F3, 35, update_labels=False)
            # plot_cell_sizes(CT_F3, bw=bws[f], bins=bins[f], xlim=(0,200))
            true_cells.append(len(CT_F3.jitcells))
            debris.append(total_cells[-1] - true_cells[-1])

        if COND == "WT":
            F3_WT_cells.append(true_cells)
            F3_WT_debris.append(debris)
            F3_WT_total.append(total_cells)
        elif COND == "KO":
            F3_KO_cells.append(true_cells)
            F3_KO_debris.append(debris)
            F3_KO_total.append(total_cells)

        total_cells = []
        true_cells  = []
        debris = []
        bws = [30, 25, 30, 30, 20]
        bins = [50, 50, 50, 50, 50]
        for f, file in enumerate(files):
            path_data = path_data_dir+file
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
            path_save = correct_path(path_save_dir+embcode)

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
                'channels':[ch]
            }

            chans = [ch]
            for _ch in range(len(channel_names)):
                if _ch not in chans:
                    chans.append(_ch)

            CT_A12 = CellTracking(
                path_data,
                path_save,
                error_correction_args=error_correction_args,
                plot_args=plot_args,
                batch_args=batch_args,
                channels=chans
            )

            CT_A12.load()
            total_cells.append(len(CT_A12.jitcells))
            plot_cell_sizes(CT_A12, bw=bws[f], bins=bins[f], xlim=(0,400))
            remove_small_cells(CT_A12, 35, update_labels=False)
            true_cells.append(len(CT_A12.jitcells))
            debris.append(total_cells[-1] - true_cells[-1])

        if COND == "WT":
            A12_WT_cells.append(true_cells)
            A12_WT_debris.append(debris)
            A12_WT_total.append(total_cells)
        elif COND == "KO":
            A12_KO_cells.append(true_cells)
            A12_KO_debris.append(debris)
            A12_KO_total.append(total_cells)

        # total_cells.append([])
        # true_cells.append([])
        # bws = [50, 55, 45, 50, 50]
        # bins = [50, 50, 50, 50, 50]
        # for f, file in enumerate(files):
        #     path_data = path_data_dir+file
        #     file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
        #     path_save = correct_path(path_save_dir+embcode)

        #     ch = channel_names.index("Casp3")

        #     batch_args = {
        #         'name_format':"ch"+str(ch)+"_{}",
        #         'extension':".tif",
        #     }
        #     plot_args = {
        #         'plot_layout': (1,1),
        #         'plot_overlap': 1,
        #         'masks_cmap': 'tab10',
        #         # 'plot_stack_dims': (256, 256), 
        #         'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
        #         'channels':[ch]
        #     }
            
        #     chans = [ch]
        #     for _ch in range(len(channel_names)):
        #         if _ch not in chans:
        #             chans.append(_ch)

        #     CT_Casp3 = CellTracking(
        #         path_data,
        #         path_save,
        #         error_correction_args=error_correction_args,
        #         plot_args=plot_args,
        #         batch_args=batch_args,
        #         channels=chans
        #     )

        #     CT_Casp3.load()
        #     total_cells[-1].append(len(CT_Casp3.jitcells))
        #     # plot_cell_sizes(CT_Casp3, bw=bws[f], bins=bins[f], path_save="{}/debris/{}/{}hr_{}_{}_{}".format(path_figures, TIME, TIME, COND, channel_names[ch],f), xlim=(0,400))
        #     remove_small_cells(CT_Casp3, 65, update_labels=True)

        # import numpy as np
        # print(np.array(total_cells) - np.array(true_cells))
        # print(true_cells)
        

import numpy as np

import matplotlib.pyplot as plt

time = [48, 60, 72, 96]
fig, ax = plt.subplots(2,2, sharey=False, figsize=(14, 9))

ax[0,0].set_title("cells - WT")
ax[0,1].set_title("cells - KO")

ax[0,0].set_xlabel("time (hr)")
ax[0,1].set_xlabel("time (hr)")

ax[0,0].set_ylabel("cell number")

F3_WT_cells_means = np.array([np.mean(data) for data in F3_WT_cells])
F3_WT_cells_stds = np.array([np.std(data) for data in F3_WT_cells])

ax[0,0].plot(time, F3_WT_cells_means, 'green', label="F3 - cells - WT")
ax[0,0].fill_between(time, F3_WT_cells_means-F3_WT_cells_stds, F3_WT_cells_means+F3_WT_cells_stds, alpha=0.1, color="green")

A12_WT_cells_means = np.array([np.mean(data) for data in A12_WT_cells])
A12_WT_cells_stds = np.array([np.std(data) for data in A12_WT_cells])

ax[0,0].plot(time, A12_WT_cells_means, 'magenta', label="A12 - cells - WT")
ax[0,0].fill_between(time, A12_WT_cells_means-A12_WT_cells_stds, A12_WT_cells_means+A12_WT_cells_stds, alpha=0.1, color="magenta")

F3_KO_cells_means = np.array([np.mean(data) for data in F3_KO_cells])
F3_KO_cells_stds = np.array([np.std(data) for data in F3_KO_cells])

ax[0,1].plot(time, F3_KO_cells_means, 'green', label="F3 - cells - KO")
ax[0,1].fill_between(time, F3_KO_cells_means-F3_KO_cells_stds, F3_KO_cells_means+F3_KO_cells_stds, alpha=0.1, color="green")

A12_KO_cells_means = np.array([np.mean(data) for data in A12_KO_cells])
A12_KO_cells_stds = np.array([np.std(data) for data in A12_KO_cells])

ax[0,1].plot(time, A12_KO_cells_means, 'magenta', label="A12 - cells - KO")
ax[0,1].fill_between(time, A12_KO_cells_means-A12_KO_cells_stds, A12_KO_cells_means+A12_KO_cells_stds, alpha=0.1, color="magenta")


ax[1,0].set_title("debris - WT")
ax[1,1].set_title("debris - KO")

ax[1,0].set_xlabel("time (hr)")
ax[1,1].set_xlabel("time (hr)")

ax[1,0].set_ylabel("debris number")

F3_WT_debris_means = np.array([np.mean(data) for data in F3_WT_debris])
F3_WT_debris_stds = np.array([np.std(data) for data in F3_WT_debris])

ax[1,0].plot(time, F3_WT_debris_means, 'green', label="F3 - debris - WT")
ax[1,0].fill_between(time, F3_WT_debris_means-F3_WT_debris_stds, F3_WT_debris_means+F3_WT_debris_stds, alpha=0.1, color="green")

A12_WT_debris_means = np.array([np.mean(data) for data in A12_WT_debris])
A12_WT_debris_stds = np.array([np.std(data) for data in A12_WT_debris])

ax[1,0].plot(time, A12_WT_debris_means, 'magenta', label="A12 - debris - WT")
ax[1,0].fill_between(time, A12_WT_debris_means-A12_WT_debris_stds, A12_WT_debris_means+A12_WT_debris_stds, alpha=0.1, color="magenta")

F3_KO_debris_means = np.array([np.mean(data) for data in F3_KO_debris])
F3_KO_debris_stds = np.array([np.std(data) for data in F3_KO_debris])

ax[1,1].plot(time, F3_KO_debris_means, 'green', label="F3 - debris - KO")
ax[1,1].fill_between(time, F3_KO_debris_means-F3_KO_debris_stds, F3_KO_debris_means+F3_KO_debris_stds, alpha=0.1, color="green")

A12_KO_debris_means = np.array([np.mean(data) for data in A12_KO_debris])
A12_KO_debris_stds = np.array([np.std(data) for data in A12_KO_debris])

ax[1,1].plot(time, A12_KO_debris_means, 'magenta', label="A12 - debris - KO")
ax[1,1].fill_between(time, A12_KO_debris_means-A12_KO_debris_stds, A12_KO_debris_means+A12_KO_debris_stds, alpha=0.1, color="magenta")

ax[0, 0].legend()
ax[0, 1].legend()
ax[1, 0].legend()
ax[1, 1].legend()

plt.tight_layout()
plt.show()

time = [48, 60, 72, 96]
fig, ax = plt.subplots(2,2, figsize=(14, 9))


ax[0, 0].set_title("F3 - cells")
ax[0, 1].set_title("A12 - cells")

ax[0, 0].set_xlabel("time (hr)")
ax[0, 1].set_xlabel("time (hr)")

ax[0, 0].set_ylabel("cell number")
ax[0, 1].set_ylabel("cell number")

ax[0, 0].plot(time, F3_WT_cells_means, 'green', label="F3 - WT")
ax[0, 0].fill_between(time, F3_WT_cells_means-F3_WT_cells_stds, F3_WT_cells_means+F3_WT_cells_stds, alpha=0.1, color="green")

ax[0, 0].plot(time, F3_KO_cells_means, 'cyan', label="F3 - KO")
ax[0, 0].fill_between(time, F3_KO_cells_means-F3_KO_cells_stds, F3_KO_cells_means+F3_KO_cells_stds, alpha=0.1, color="cyan")

ax[0, 1].plot(time, A12_WT_cells_means, 'magenta', label="A12 - WT")
ax[0, 1].fill_between(time, A12_WT_cells_means-A12_WT_cells_stds, A12_WT_cells_means+A12_WT_cells_stds, alpha=0.1, color="magenta")

ax[0, 1].plot(time, A12_KO_cells_means, 'red', label="A12 - KO")
ax[0, 1].fill_between(time, A12_KO_cells_means-A12_KO_cells_stds, A12_KO_cells_means+A12_KO_cells_stds, alpha=0.1, color="red")

ax[0, 0].legend()
ax[0, 1].legend()


ax[1, 0].set_title("F3 - debris")
ax[1, 1].set_title("A12 - debris")

ax[1, 0].set_xlabel("time (hr)")
ax[1, 1].set_xlabel("time (hr)")

ax[1, 0].set_ylabel("debris number")
ax[1, 1].set_ylabel("debris number")

ax[1, 0].plot(time, F3_WT_debris_means, 'green', label="F3 - WT")
ax[1, 0].fill_between(time, F3_WT_debris_means-F3_WT_debris_stds, F3_WT_debris_means+F3_WT_debris_stds, alpha=0.1, color="green")

ax[1, 0].plot(time, F3_KO_debris_means, 'cyan', label="F3 - KO")
ax[1, 0].fill_between(time, F3_KO_debris_means-F3_KO_debris_stds, F3_KO_debris_means+F3_KO_debris_stds, alpha=0.1, color="cyan")

ax[1, 1].plot(time, A12_WT_debris_means, 'magenta', label="A12 - WT")
ax[1, 1].fill_between(time, A12_WT_debris_means-A12_WT_debris_stds, A12_WT_debris_means+A12_WT_debris_stds, alpha=0.1, color="magenta")

ax[1, 1].plot(time, A12_KO_debris_means, 'red', label="A12 - KO")
ax[1, 1].fill_between(time, A12_KO_debris_means-A12_KO_debris_stds, A12_KO_debris_means+A12_KO_debris_stds, alpha=0.1, color="red")

ax[1, 0].legend()
ax[1, 1].legend()

plt.tight_layout()
plt.show()