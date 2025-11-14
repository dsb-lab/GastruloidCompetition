### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, check_or_create_dir, get_file_names
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
CONDS = ["WT", "KO"]
repeats = ["n2", "n3", "n4", "n3_Ab"]
repeats = ["n3_Ab"]


size_th_low = 31.9
size_th_high = 250.0
for COND in CONDS:
    for REP in repeats:
        
        path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/input/{}/{}/".format(COND,REP)
        path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/{}/{}/".format(COND,REP)

        check_or_create_dir(path_save_dir)
        
        files = get_file_names(path_data_dir)
        
        channel_names = ["A12", "p53", "F3", "DAPI"]

        for f, file in enumerate(files):
            
            path_data = path_data_dir+file
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
            path_save = path_save_dir+embcode
                
            check_or_create_dir(path_save)

            ### DEFINE ARGUMENTS ###
            segmentation_args={
                'method': 'stardist2D', 
                'model': model, 
                'blur': [2,1], 
                'min_outline_length':100,
            }

            concatenation3D_args = {
                'do_3Dconcatenation': False
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
                'plot_stack_dims': (256, 256), 
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

            CT_F3.load()
            
            labs_to_rem = []
            for cell in CT_F3.jitcells:
                zc = int(cell.centers[0][0])
                zcid = cell.zs[0].index(zc)

                mask = cell.masks[0][zcid]
                area = len(mask) * CT_F3.metadata["XYresolution"]**2
                if area < size_th_low:
                    labs_to_rem.append(cell.label)
                elif area > size_th_high:
                    labs_to_rem.append(cell.label)
                    
            for lab in labs_to_rem:
                CT_F3._del_cell(lab)  
                 
            CT_F3.update_labels()
            
            print()
            print("F3", len(CT_F3.jitcells))
            print(len(labs_to_rem))
            
            ch = channel_names.index("A12")
            batch_args = {
                'name_format':"ch"+str(ch)+"_{}",
                'extension':".tif",
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

            CT_A12.load()

            labs_to_rem = []
            for cell in CT_A12.jitcells:
                zc = int(cell.centers[0][0])
                zcid = cell.zs[0].index(zc)

                mask = cell.masks[0][zcid]
                area = len(mask) * CT_A12.metadata["XYresolution"]**2
                if area < size_th_low:
                    labs_to_rem.append(cell.label)
                elif area > size_th_high:
                    labs_to_rem.append(cell.label)
            for lab in labs_to_rem:
                CT_A12._del_cell(lab)  
                 
            CT_A12.update_labels()
            
            print("A12", len(CT_A12.jitcells))
            print(len(labs_to_rem))
