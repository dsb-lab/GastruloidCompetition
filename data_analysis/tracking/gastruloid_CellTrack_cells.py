import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/celltrack/src/celltrack')

from celltrack import CellTracking, get_file_embcode, read_img_with_resolution, load_cells


import os

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/PH3/movies/'
path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/PH3/CellTrackObjects/'

# path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/movies_registered/'
# path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/lightsheet/CellTrackObjects/'

files = os.listdir(path_data)

file, embcode = get_file_embcode(path_data, 'WT_48')
print(file)

ch = 1
IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=ch)
IMGS = IMGS[0:1, 25:]

cells, CT_info = load_cells(path_save, embcode + '_ch%d' %ch)

CT = CellTracking(IMGS, path_save, embcode + '_ch%d' %ch, CELLS=cells, CT_info=CT_info
                    , stacks_for_plotting=None
                    , plot_layout=(1,1)
                    , plot_overlap=1
                    , masks_cmap='tab10'
                    , min_outline_length=50
                    , neighbors_for_sequence_sorting=50
                    , backup_steps=5
                    , cell_distance_axis="xy"
                    , movement_computation_method="center"
                    , mean_substraction_cell_movement=False
                    , line_builder_mode='lasso')

CT.plot_tracking(plot_layout=(1,1), plot_overlap=1, plot_stack_dims=(512, 512))
# CT.plot_cell_movement(substract_mean=False)
CT.plot_masks3D_Imagej(cell_selection=False)

