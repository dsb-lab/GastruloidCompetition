from qlivecell import correct_drift, cellSegTrack, extract_fluoro, get_file_names, get_file_name
import os

path_data = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/merge/2024_07_22_DMSOmerged_Casp3/"
path_save = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/merge/stresults/"

channel_names = ["A12", "Casp3", "F3", "DAPI"]

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.makedirs(path_save)


### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

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
    'min_cell_planes': 1,
}

tracking_args = {
    'time_step': 5, # minutes
    'method': 'greedy', 
    'z_th':10, 
    'dist_th' : 10.0,
}

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    # 'plot_stack_dims': (512, 512), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[1]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'points',
}

files = get_file_names(path_data)
files = [file for file in files if ".tif" in file]
file = files[0]

path_data_file = path_data+file
file, embcode = get_file_name(path_data, file, allow_file_fragment=False, return_files=False, return_name=True)
path_save_file = path_save+embcode
            
ch_F3 = channel_names.index("F3")
batch_args = {
    'name_format':"ch"+str(ch_F3)+"_{}",
    'extension':".tif",
}

chans = [ch_F3]
for _ch in range(len(channel_names)):
    if _ch not in chans:
        chans.append(_ch)

cST_F3 = cellSegTrack(
    path_data_file,
    path_save_file,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    tracking_args=tracking_args,
    error_correction_args=error_correction_args,
    plot_args=plot_args,
    batch_args=batch_args,
    channels=chans
)

cST_F3.run()

ch_A12 = channel_names.index("A12")
batch_args = {
    'name_format':"ch"+str(ch_A12)+"_{}",
    'extension':".tif",
}

chans = [ch_A12]
for _ch in range(len(channel_names)):
    if _ch not in chans:
        chans.append(_ch)

cST_A12 = cellSegTrack(
    path_data_file,
    path_save_file,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    tracking_args=tracking_args,
    error_correction_args=error_correction_args,
    plot_args=plot_args,
    batch_args=batch_args,
    channels=chans
)

cST_A12.run()

resultsF3 = extract_fluoro(cST_F3)
resultsA12 = extract_fluoro(cST_A12)

drift_correction_F3, data_z_F3 = correct_drift(resultsF3, channel_names.index("F3"), plotting=False)
drift_correction_A12, data_z_A12 = correct_drift(resultsA12, channel_names.index("A12"), plotting=False) 
drift_correction_Casp3, data_z_Casp3 = correct_drift(resultsCasp3, channel_names.index("Casp3"), plotting=False)
