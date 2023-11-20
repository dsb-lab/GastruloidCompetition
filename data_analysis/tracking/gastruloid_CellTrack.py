import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/src')

from embdevtools.embdevtools import get_file_embcode, read_img_with_resolution, CellTracking, save_4Dstack, save_3Dstack, get_default_args, construct_RGB

import os 
home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/apo/data1/cropped8bit/'
path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/apo/data1/CellTrackObjects/'

file, embcode, files = get_file_embcode(path_data,  "cropped", returnfiles=True)
IMGSnuc1, xyres, zres = read_img_with_resolution(path_data+file, channel=1, stack=True)
IMGSnuc2, xyres, zres = read_img_with_resolution(path_data+file, channel=0, stack=True)
IMGS = construct_RGB(R=IMGSnuc1, G=IMGSnuc2, B=None)

from cellpose import models
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/2h_claire_ERK-KTR_MKATE2/movies/cell_tracking/training_set_expanded_nuc/models/blasto')
segmentation_method = 'cellpose'
ch_seg = 0
# composed of internal segmentation arguments + arguments from the decided segmentation method
segmentation_args={
    'method': 'cellpose', 
    'model': model, 
    'blur': None, #[[5,5], 1], 
    'channels': [ch_seg+1,0],
    'channel_axis': 5,
    'flow_threshold': 0.4,
}

concatenation3d_args = {
    'distance_th_z': 3.0, 
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
    'min_cell_planes': 1,
}

train_segmentation_args = {
    'model_save_path': path_save,
    'model_name': None,
    'blur': [[5,5], 1]
}

tracking_args = {
    'time_step': 5, # minutes
    'method': 'hungarian', 
    'z_th':100, 
    'cost_attributes':['distance', 'volume', 'shape'], 
    'cost_ratios':[0.6,0.2,0.2]
}

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    'plot_stack_dims': (512, 512), 
    'plot_centers':[False, False]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'lasso',
}


CT = CellTracking(
    IMGS, 
    path_save, 
    embcode+"%d" %ch_seg, 
    xyresolution=xyres, 
    zresolution=zres, 
    loadcells=True,
    segmentation_args=segmentation_args,
    segment3D=False,
    concatenation3D_args=concatenation3d_args,
    train_segmentation_args = train_segmentation_args,
    tracking_args = tracking_args, 
    error_correction_args=error_correction_args,    
    plot_args = plot_args,
)

# CT()
CT.plot_tracking(plot_args, stacks_for_plotting=IMGS)
