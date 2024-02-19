### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift
import numpy as np
import matplotlib.pyplot as plt

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
TIMES = ["48hr", "72hr", "96hr"]
CONDS = ["WT", "KO"]
for TIME in TIMES:
    for COND in CONDS:
        if TIME != "96hr": 
            continue
        if COND != "KO":
            continue

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

        Casp3_distributions_F3 = []
        Casp3_distributions_A12 = []
        Casp3_distributions_Casp3 = []

        F3_sizes = []
        A12_sizes = []
        Casp3_sizes = []

        for f, file in enumerate(files):
            Casp3_distributions_F3.append([])
            Casp3_distributions_A12.append([])
            Casp3_distributions_Casp3.append([])

            F3_sizes.append([])
            A12_sizes.append([])
            Casp3_sizes.append([])

            path_data = path_data_dir+file
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
            path_save = path_save_dir+embcode
            try: 
                files = get_file_names(path_save)
            except: 
                import os
                os.mkdir(path_save)

            ### LOAD STARDIST MODEL ###
            from stardist.models import StarDist2D
            model = StarDist2D.from_pretrained('2D_versatile_fluo')

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
            # CT_F3.plot_tracking()
            
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
            # CT_A12.plot_tracking()
            
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
                'channels':[ch],
                'min_outline_length':75,
            }
            
            chans = [ch]
            for _ch in range(len(channel_names)):
                if _ch not in chans:
                    chans.append(_ch)

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
            # CT_Casp3.plot_tracking()
            
            
            resultsF3 = extract_fluoro(CT_F3)
            resultsA12 = extract_fluoro(CT_A12)
            resultsCasp3 = extract_fluoro(CT_Casp3)
            
            drift_correction_F3, data_z_F3 = correct_drift(resultsF3, channel_names.index("F3"), plotting=False)
            drift_correction_A12, data_z_A12 = correct_drift(resultsA12, channel_names.index("A12"), plotting=False) 
            drift_correction_Casp3, data_z_Casp3 = correct_drift(resultsCasp3, channel_names.index("Casp3"), plotting=False)
            
                
            #### COMPUTE CASP3 DISTRIBUTIONS ####
            
            ch = channel_names.index("Casp3")
            # F3
            for cell in CT_F3.jitcells:
                center = cell.centers[0]
                z = np.int32(center[0])
                zid = cell.zs[0].index(z)
                
                mask = cell.masks[0][zid]
                
                img = CT_F3.hyperstack[0,z,ch]
                casp3_val = np.mean(img[mask[:, 1], mask[:, 0]])
                Casp3_distributions_F3[-1].append(casp3_val)
                F3_sizes[-1].append(len(mask))

            # A12
            for cell in CT_A12.jitcells:
                center = cell.centers[0]
                z = np.int32(center[0])
                zid = cell.zs[0].index(z)
                
                mask = cell.masks[0][zid]
                
                img = CT_A12.hyperstack[0,z,ch]
                casp3_val = np.mean(img[mask[:, 1], mask[:, 0]])
                Casp3_distributions_A12[-1].append(casp3_val)
                A12_sizes[-1].append(len(mask))
                
            # Casp3
            for cell in CT_Casp3.jitcells:
                center = cell.centers[0]
                z = np.int32(center[0])
                zid = cell.zs[0].index(z)
                
                mask = cell.masks[0][zid]
                
                img = CT_Casp3.hyperstack[0,z,ch]
                casp3_val = np.mean(img[mask[:, 1], mask[:, 0]])
                Casp3_distributions_Casp3[-1].append(casp3_val)
                Casp3_sizes[-1].append(len(mask))


            f3_casp3 = np.array(Casp3_distributions_F3[f])
            a12_casp3 = np.array(Casp3_distributions_A12[f])
            casp3_casp3 = np.array(Casp3_distributions_Casp3[f])

            thf3 = np.mean(f3_casp3)+1.5*np.std(f3_casp3)
            tha12 = np.mean(a12_casp3)+1.5*np.std(a12_casp3)
            
            
            fig, ax = plt.subplots()
            ax.set_title("{} {}".format(COND, TIME))
            y1, x, _ = ax.hist(f3_casp3, color=[0.9,0,0.9,0.5], bins=40, density=True, label="F3")
            y2, x, _ = ax.hist(a12_casp3, color=[0,0.8,0,0.5], bins=40, density=True, label="A12")
            y3, x, _ = ax.hist(casp3_casp3, color=[0.9,0.9,0.0,0.5], bins=40, density=True, label="Casp3")
            ax.vlines([thf3, tha12], np.min([y1,y2,y3]), np.max([y1,y2,y3]))
            ax.set_ylabel("density")
            ax.set_xlim(0,70)
            ax.set_xlabel("Casp3")
            ax.legend()
            plt.show()
                
            th_casp3 = np.mean([thf3, tha12])
            
            cells_to_remove = []
            for c, cell in enumerate(CT_Casp3.jitcells):
                center = cell.centers[0]
                z = np.int32(center[0])
                zid = cell.zs[0].index(z)
                
                mask = cell.masks[0][zid]
                
                img = CT_Casp3.hyperstack[0,z,ch]
                casp3_val = np.mean(img[mask[:, 1], mask[:, 0]])
                
                if casp3_val < th_casp3:
                    cells_to_remove.append(cell.label)
            
            # for lab in reversed(cells_to_remove):
            #     CT_Casp3._del_cell(lab)
            
            # CT_Casp3.update_labels()
            

