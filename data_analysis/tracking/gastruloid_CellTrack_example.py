import sys
sys.path.append('/home/pablo/Desktop/PhD/projects/embdevtools/celltrack')

from celltrack import CellTracking, get_file_embcode, read_img_with_resolution
from cellpose import models

import os

home = os.path.expanduser('~')
path_data=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/KI67/movies/'

path_save=home+'/Desktop/PhD/projects/Data/gastruloids/joshi/competition/KI67/CellTrackObjects/'

files = os.listdir(path_data)
# model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/blastocysts/movies/2h_claire_ERK-KTR_MKATE2/cell_tracking/training_set_expanded_nuc/models/blasto')
model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/gastruloids/cellpose/train_sets/joshi/confocal/models/CP_20230510_115154')
# model  = models.Cellpose(gpu=True, model_type='nuclei')

file, embcode = get_file_embcode(path_data, 2)
print(file)
IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)

### CHANNEL 1 ###

CT = CellTracking(IMGS, path_save, embcode
                    , model=model
                    , trainedmodel=True
                    , channels=[0,0]
                    , flow_th_cellpose=0.4
                    , distance_th_z=3.0
                    , xyresolution=xyres # microns per pixel
                    , zresolution =zres
                    , relative_overlap=False
                    , use_full_matrix_to_compute_overlap=True
                    , z_neighborhood=2
                    , overlap_gradient_th=0.15
                    , plot_layout=(2,2)
                    , plot_overlap=1
                    , masks_cmap='tab10'
                    , min_outline_length=200
                    , neighbors_for_sequence_sorting=7
                    , plot_tracking_windows=1
                    , backup_steps=20
                    , time_step=6 # minutes
                    , cell_distance_axis="xy"
                    , movement_computation_method="center"
                    , mean_substraction_cell_movement=False
                    , plot_outline_width=0
                    , blur_args=None)

CT()
CT.plot_tracking(plot_stack_dims = (1024, 1024), plot_layout=(1,1), plot_outline_width=1)

# for cellid, cell in reversed(list(enumerate(CT.cells))):
#     if cell.times != [i for i in range(10)]: CT.cells.pop(cellid)

# CT.update_labels()

# CT.plot_tracking(plot_stack_dims = (512, 512), plot_layout=(1,1), plot_outline_width=1)
# save_cells(CT, path_save, embcode+'_ch%d' %0)
# # CT.plot_cell_movement()
 
# # CT.plot_masks3D_Imagej(cell_selection=True, color=None, channel_name="0")

# # CT1 = CT
# img = IMGS[0,10]
# # img[img < 15] = 0
# import cv2
# img_blur = cv2.GaussianBlur(img, [5,5], 1) 
# # img_blur[img_blur < 10] = 0
# # img_blur = cv2.GaussianBlur(img_blur, [5,5], 1) 
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1,2)

# ax[0].imshow(img)
# ax[1].imshow(img_blur)
# plt.show()