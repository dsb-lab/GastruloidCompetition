### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB
from embdevtools import remove_small_cells, plot_cell_sizes

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###

# ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
TIMES = ["48hr", "72hr", "96hr"]
CONDS = ["WT", "KO"]
path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/figures/"

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

        CENTERS = []
        FATES = []
        LABS = []

        channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
        if "96hr" in path_data_dir:
            channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

        total_cells = []
        true_cells  = []

        total_cells.append([])
        true_cells.append([])
        bws = [35, 30, 30, 30, 30]
        bins = [40, 25, 30, 30, 30]
        for f, file in enumerate(files):
            path_data = path_data_dir+file
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
            path_save = path_save_dir+embcode

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
            total_cells[-1].append(len(CT_F3.jitcells))
            # plot_cell_sizes(CT_F3, bw=bws[f], bins=bins[f], path_save="{}/debris/{}/{}hr_{}_{}_{}".format(path_figures, TIME, TIME, COND, channel_names[ch],f), xlim=(0,400))
            remove_small_cells(CT_F3, 57)
            CT_F3.update_labels()
            true_cells[-1].append(len(CT_F3.jitcells))


        total_cells.append([])
        true_cells.append([])
        bws = [30, 25, 30, 30, 20]
        bins = [50, 50, 50, 50, 50]
        for f, file in enumerate(files):
            path_data = path_data_dir+file
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
            path_save = path_save_dir+embcode

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
            total_cells[-1].append(len(CT_A12.jitcells))
            # plot_cell_sizes(CT_A12, bw=bws[f], bins=bins[f], path_save="{}/debris/{}/{}hr_{}_{}_{}".format(path_figures, TIME, TIME, COND, channel_names[ch],f), xlim=(0,400))
            remove_small_cells(CT_A12, 57)
            CT_A12.update_labels()
            true_cells[-1].append(len(CT_A12.jitcells))


        total_cells.append([])
        true_cells.append([])
        bws = [50, 55, 45, 50, 50]
        bins = [50, 50, 50, 50, 50]
        for f, file in enumerate(files):
            path_data = path_data_dir+file
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
            path_save = path_save_dir+embcode

            ch = channel_names.index("Casp3")

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

            CT_Casp3 = CellTracking(
                path_data,
                path_save,
                error_correction_args=error_correction_args,
                plot_args=plot_args,
                batch_args=batch_args,
                channels=chans
            )

            CT_Casp3.load()
            total_cells[-1].append(len(CT_Casp3.jitcells))
            # plot_cell_sizes(CT_Casp3, bw=bws[f], bins=bins[f], path_save="{}/debris/{}/{}hr_{}_{}_{}".format(path_figures, TIME, TIME, COND, channel_names[ch],f), xlim=(0,400))
            remove_small_cells(CT_Casp3, 57)
            CT_Casp3.update_labels()

        # import numpy as np
        # print(np.array(total_cells) - np.array(true_cells))
        # print(true_cells)