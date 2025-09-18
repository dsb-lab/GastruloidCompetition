### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, check_or_create_dir, get_file_names
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

size_th_low = [25.0, 25.5, 31.3, 29.0]
size_th_low = np.array([25.0, 25.5, 36.800000000000004, 24.3, 27.0, 31.3, 24.700000000000003, 29.0])+1
size_th_high = 250.0

CONDS = ["auxin_48-72_48", "auxin_48-72_72a", "auxin_48-72_72b" , "auxin_48-72_96", "auxin_72-96_96", "noauxin_72", "noauxin_96", "secondaryonly"]

for C, COND in enumerate(CONDS):
    path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
    path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
        
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
            if area < size_th_low[C]:
                labs_to_rem.append(cell.label)
            elif area > size_th_high:
                labs_to_rem.append(cell.label)
                
        for lab in labs_to_rem:
            CT_F3._del_cell(lab)  
                
        CT_F3.update_labels()
        
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
            if area < size_th_low[C]:
                labs_to_rem.append(cell.label)
            elif area > size_th_high:
                labs_to_rem.append(cell.label)
        for lab in labs_to_rem:
            CT_A12._del_cell(lab)  
                
        CT_A12.update_labels()
        