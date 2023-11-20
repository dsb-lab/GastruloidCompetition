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

file, embcode = get_file_embcode(path_data, 'p53_120')
print(file)
ch = 2
IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=ch)
IMGS2 = IMGS[0:1, 25:]

import numpy
IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=0)
IMGS0 = IMGS[0:1, 25:]
        
IMGS, xyres, zres = read_img_with_resolution(path_data+file, channel=1)
IMGS1 = IMGS[0:1, 25:]
    
IMGS = numpy.zeros((IMGS1.shape[0], IMGS1.shape[1], IMGS1.shape[2], IMGS1.shape[3], 3))
IMGS[:,:,:,:,0] = IMGS0
IMGS[:,:,:,:,1] = IMGS1

cells, CT_info = load_cells(path_save, embcode + '_ch%d' %ch)
        
CT = CellTracking(IMGS2, path_save, embcode + '_ch%d' %ch, CELLS=cells, CT_info=CT_info
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

for cell in CT.jitcells:
    
    zc = int(cell.centers[0][0])
    unwanted = []
    for zid,z in enumerate(cell.zs[0]):
        if zc != z: unwanted.append(zid)
        
    for ele in sorted(unwanted, reverse = True):
        del cell.zs[0][ele]
        del cell.outlines[0][ele]
        del cell.masks[0][ele]
        
CT.plot_tracking(plot_layout=(1,1), plot_overlap=1, plot_stack_dims=(512, 512))
# CT.plot_cell_movement(substract_mean=False)
# CT.plot_masks3D_Imagej(cell_selection=False)

CH0 = []
for cell in CT.jitcells:
    z = int(cell.centers[0][0])
    zid = list(cell.zs[0]).index(z)
    mask = cell.masks[0][zid]
    msk = IMGS0[0,z][mask]/numpy.max(IMGS0[0,z])
    CH0.append(numpy.sum(msk))

CH1 = []
for cell in CT.jitcells:
    z = int(cell.centers[0][0])
    zid = list(cell.zs[0]).index(z)
    mask = cell.masks[0][zid]
    msk = IMGS1[0,z][mask]/numpy.max(IMGS1[0,z])
    CH1.append(numpy.sum(msk))

CH0 = numpy.array(CH0) - numpy.mean(CH0)
CH1 = numpy.array(CH1) - numpy.mean(CH1)
c0mito = 0
c1mito = 0
for c in range(len(CT.jitcells)):
    if CH0[c] > CH1[c]: 
        c0mito+=1
    else: c1mito +=1

cells, CT_info = load_cells(path_save, embcode + '_ch%d' %0)
c0total = len(cells) + c0mito

cells, CT_info = load_cells(path_save, embcode + '_ch%d' %1)
c1total = len(cells) + c1mito

print()
print(c0total)
print(c0mito)
print()
print(c1total)
print(c1mito)

print(c0mito*100 / c0total)
print(c1mito*100 / c1total)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(CH0)
ax.hist(CH1)
plt.show()