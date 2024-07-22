### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, get_file_names, correct_path
import numpy as np
from scipy import stats

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

from numba import njit

@njit
def compute_distance_xyz(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return dist

@njit
def compute_dists(points1, points2):
    dists = np.zeros((len(points1), len(points2)))
    for i, center in enumerate(points1):
        for j, cont in enumerate(points2):
            dists[i,j] = compute_distance_xyz(center, cont)
    return dists


### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/96hr/KO/'
path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/96hr/KO/'

try: 
    files = get_file_names(path_save_dir)
except: 
    import os
    os.mkdir(path_save_dir)

### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data_dir)

channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
if "96hr" in path_data_dir:
    channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

dists_contour_Casp3 = []
dists_contour_A12 = []
dists_contour_F3 = []
dists_centroid_Casp3 = []
dists_centroid_A12 = []
dists_centroid_F3 = []

binths = [10,10,10,10]
# for f, file in enumerate(files):
f=0
file=files[f]
path_data = path_data_dir+file
file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
path_save = correct_path(path_save_dir+embcode)
try: 
    files = get_file_names(path_save)
except: 
    import os
    os.mkdir(path_save)

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
    'min_cell_planes': 2
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'points',
}

ch_F3 = channel_names.index("F3")

batch_args = {
    'name_format':"ch"+str(ch_F3)+"_{}",
    'extension':".tif",
}
plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    # 'plot_stack_dims': (256, 256), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[ch_F3],
    'min_outline_length':75,
}

chans = [ch_F3]
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

ch_A12 = channel_names.index("A12")
batch_args = {
    'name_format':"ch"+str(ch_A12)+"_{}",
    'extension':".tif",
}
plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    # 'plot_stack_dims': (256, 256), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[ch_A12],
    'min_outline_length':75,
}

chans = [ch_A12]
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

ch_Casp3 = channel_names.index("Casp3")

# CT_Casp3.plot_tracking(plot_args=plot_args)

points = []
zs = []
for cell in CT_F3.jitcells:
    outlines = cell.outlines[0]
    for zid, z in enumerate(cell.zs[0]):
        for point in outlines[zid]:
            point3D = [z, point[0], point[1]]
            points.append(point3D)

for cell in CT_A12.jitcells:
    outlines = cell.outlines[0]
    for zid, z in enumerate(cell.zs[0]):
        for point in outlines[zid]:
            point3D = [z, point[0], point[1]]
            points.append(point3D)


xyres = CT_F3.metadata["XYresolution"]
zres  = CT_F3.metadata["Zresolution"]
resolutions = np.array([zres, xyres, xyres])
points = np.array(points)*resolutions

from scipy.spatial import ConvexHull
hull = ConvexHull(points)
outline3D = points[hull.vertices]

# Get embryo 3D Centroid
centers = []
for cell in CT_F3.jitcells:
    center = cell.centers[0]
    centers.append(center)
    
for cell in CT_A12.jitcells:
    center = cell.centers[0]
    centers.append(center)

centers = np.array(centers)
minz = int(np.min(centers[:,0]))
maxz = int(np.max(centers[:,0]))

centers = np.array(centers)*resolutions
centroid = np.mean(centers, axis=0)

from embdevtools import EmbryoSegmentation, tif_reader_5D
hyperstack, metadata = tif_reader_5D(path_data)
channels_seg = np.array([ch_A12, ch_F3, ch_Casp3])
hyperstack_seg = np.sum(hyperstack[:,:,channels_seg, :, :].astype("int32"), axis=2)

z_plot = np.rint(hyperstack_seg.shape[1]/2).astype("int64")
ES = EmbryoSegmentation(
        hyperstack_seg,
        ksize=5,
        ksigma=20,
        binths=binths[f],
        apply_biths_to_zrange_only=False,
        checkerboard_size=10,
        num_inter=100,
        smoothing=20,
        trange=None,
        zrange=range(minz, maxz+1),
        mp_threads=14,
    )


ES(hyperstack_seg)

ES.plot_segmentation(0, minz + 2)
ES.plot_segmentation(0, z_plot-20)
ES.plot_segmentation(0, z_plot)
ES.plot_segmentation(0, z_plot+20)
ES.plot_segmentation(0, maxz - 2)

