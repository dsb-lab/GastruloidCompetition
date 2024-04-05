### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, tif_reader_5D, plot_cell_sizes

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data_dir='/home/pablo/Downloads/'
path_save_dir='/home/pablo/Downloads/_test/'

CENTERS = []
FATES = []
LABS = []

channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]

file = "test72.tif"
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
    'blur': [0.1,1], 
    # 'n_tiles': (2,2),
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
    'min_outline_length':75
}

chans = [ch]
for _ch in range(len(channel_names)):
    if _ch not in chans:
        chans.append(_ch)

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    'blur': [2,1], 
    # 'n_tiles': (2,2),
}

CT = CellTracking(
    path_data,
    path_save,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    error_correction_args=error_correction_args,
    plot_args=plot_args,
    batch_args=batch_args,
    channels=chans
)

CT.run()
CT.plot_tracking()

plot_cell_sizes(CT, bw=20, bins=50, path_save=path_save_dir, xlim=(0,150))

