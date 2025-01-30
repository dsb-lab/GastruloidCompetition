### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
# experiment = "2023_11_17_Casp3"
experiment = "2024_03_Casp3"
TIMES = ["48hr", "60hr", "72hr", "96hr"]
CONDS = ["WT", "KO"]

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
        
        # channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
        channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
        # if "96hr" in path_data_dir:
        #     channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

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

            CT_F3.run()
            # CT_F3.plot(plot_args=plot_args)
            
            labs_to_rem = []
            for cell in CT_F3.jitcells:
                zc = int(cell.centers[0][0])
                zcid = cell.zs[0].index(zc)

                mask = cell.masks[0][zcid]
                area = len(mask)* CT_F3.CT_info.xyresolution**2
                if area < 36.7:
                    labs_to_rem.append(cell.label)
                
            for lab in labs_to_rem:
                CT_F3._del_cell(lab)   
            CT_F3.update_labels()
                        
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

            CT_A12.run()
            # CT_A12.plot(plot_args=plot_args)
            labs_to_rem = []
            for cell in CT_A12.jitcells:
                zc = int(cell.centers[0][0])
                zcid = cell.zs[0].index(zc)

                mask = cell.masks[0][zcid]
                area = len(mask)* CT_A12.CT_info.xyresolution**2
                if area < 36.7:
                    labs_to_rem.append(cell.label)
                
            for lab in labs_to_rem:
                CT_A12._del_cell(lab)   
            CT_A12.update_labels()
                         
            ch = channel_names.index("Casp3")
            chans = [ch]
            for _ch in range(len(channel_names)):
                if _ch not in chans:
                    chans.append(_ch)
            
            for stage in ["early", "mid", "late"]:
                print(stage)
                batch_args = {
                    'name_format':"ch"+str(ch)+"_{}_"+stage,
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
                    'blur': [1,1], 
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

                plot_args = {
                    'plot_layout': (1,1),
                    'plot_overlap': 1,
                    'masks_cmap': 'tab10',
                    'plot_stack_dims': (512, 512), 
                    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
                    'channels':[1, 0, 3],
                    'min_outline_length':75,
                    }

                # CT_Casp3.run()
                
                if stage=="late":
                    batch_args = {
                    'name_format':"ch"+str(ch)+"_{}_"+"early",
                    'extension':".tif",
                    }
                    CT_Casp3_early = cellSegTrack(
                        path_data,
                        path_save,
                        segmentation_args=segmentation_args,
                        concatenation3D_args=concatenation3D_args,
                        error_correction_args=error_correction_args,
                        plot_args=plot_args,
                        batch_args=batch_args,
                        channels=chans
                    )
                    CT_Casp3_early.load()
                    intens = []
                    for cell in CT_Casp3_early.jitcells:
                        zc = int(cell.centers[0][0])
                        zcid = cell.zs[0].index(zc)

                        mask = cell.masks[0][zcid]
                        ints = CT_Casp3.hyperstack[0,zc,ch,:,:][mask[:,1], mask[:,0]]
                        intens = [*intens, *ints]
                    
                    int_th = np.percentile(intens, 90)
                    batch_args = {
                    'name_format':"ch"+str(ch)+"_{}_"+"late",
                    'extension':".tif",
                    }
                    CT_Casp3.run()
                    labs_to_rem = []
                    for cell in CT_Casp3.jitcells:
                        zc = int(cell.centers[0][0])
                        zcid = cell.zs[0].index(zc)

                        mask = cell.masks[0][zcid]
                        area = len(mask)* CT_Casp3.CT_info.xyresolution**2
                        ints = np.mean(CT_Casp3.hyperstack[0,zc,ch,:,:][mask[:,1], mask[:,0]])
                        if area > 36.7:
                            labs_to_rem.append(cell.label)
                        if ints < int_th and cell.label not in labs_to_rem:
                            labs_to_rem.append(cell.label)
                            
                    for lab in labs_to_rem:
                        CT_Casp3._del_cell(lab)   
                    
                    plot_args = {
                    'plot_layout': (1,1),
                    'plot_overlap': 1,
                    'masks_cmap': 'tab10',
                    'plot_stack_dims': (512, 512), 
                    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
                    'channels':[1, 0, 3],
                    'min_outline_length':75,
                    }
                    # CT_Casp3.plot(plot_args=plot_args)                    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     