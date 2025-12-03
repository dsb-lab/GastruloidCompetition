### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift, correct_path
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/figures/neighbors/"
try: 
    files = get_file_names(path_figures)
except: 
    import os
    os.mkdir(path_figures)
    
# ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
TIMES = ["48hr", "72hr", "96hr"]
CONDS = ["WT", "KO"]
TIME = TIMES[0]
COND = CONDS[1]

path_figures_time = "{}{}/".format(path_figures, TIME)
try: 
    files = get_file_names(path_figures_time)
except: 
    import os
    os.mkdir(path_figures_time)
        
path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/{}/'.format(TIME, COND)
path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/{}/'.format(TIME, COND)

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

neighs_fates_F3_sum = np.zeros((len(files), 2))
neighs_fates_A12_sum = np.zeros((len(files), 2))
neighs_fates_Casp3_F3_sum = np.zeros((len(files), 2))
neighs_fates_Casp3_A12_sum = np.zeros((len(files), 2))

f = 2
file = files[f]

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

CT_F3.load()

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

CT_A12.load()

ch = channel_names.index("Casp3")
chans = [ch]
for _ch in range(len(channel_names)):
    if _ch not in chans:
        chans.append(_ch)
batch_args = {
    'name_format':"ch"+str(ch)+"_{}_early",
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
    'blur': [5,1], 
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

CT_Casp3.load()
    
# CT_F3.plot_tracking()
# CT_A12.plot_tracking()
# CT_Casp3.plot_tracking()

import numpy as np

### CORRECT DISTRIBUTIONS ###

centers = []
areas = []

F3_dist = []
for cell in CT_F3.jitcells: 
    for zid, z in enumerate(cell.zs[0]):
        mask = cell.masks[0][zid]
        img = CT_F3.hyperstack[0,z, channel_names.index("F3")]
        F3_dist.append(np.mean(img[mask[:,1], mask[:,0]]))
        areas.append(len(mask))
    centers.append(cell.centers[0])
        
A12_dist = []
for cell in CT_A12.jitcells: 
    for zid, z in enumerate(cell.zs[0]):
        mask = cell.masks[0][zid]
        areas.append(len(mask))
        img = CT_A12.hyperstack[0,z, channel_names.index("A12")]
        A12_dist.append(np.mean(img[mask[:,1], mask[:,0]]))
    centers.append(cell.centers[0])

area = np.mean(areas)
dim = 2*np.sqrt(area/np.pi)

import numpy as np
from scipy import stats

def compute_distance_xy(x1, x2, y1, y2):
    """
    Parameters
    ----------
    x1 : number
        x coordinate of point 1
    x2 : number
        x coordinate of point 2
    y1 : number
        y coordinate of point 1
    y2 : number
        y coordinate of point 2

    Returns
    -------
    dist : number
        euclidean distance between points (x1, y1) and (x2, y2)
    """
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

## Now contract the shape as much as we want. 
F3_dist = np.array(F3_dist)
A12_dist = np.array(A12_dist)

mdiff = np.mean(F3_dist) - np.mean(A12_dist)
if mdiff > 0:
    A12_dist += mdiff
else: 
    F3_dist -= mdiff 

zres = CT_F3.metadata["Zresolution"]
xyres = CT_F3.metadata["XYresolution"]

fates = []
centers = []
labels = []
for cell in CT_F3.jitcells:
    fates.append(0)
    centers.append(cell.centers[0]*[zres, xyres, xyres])
    labels.append(cell.label)
for cell in CT_A12.jitcells:
    fates.append(1)
    centers.append(cell.centers[0]*[zres, xyres, xyres])
    labels.append(cell.label)

Casp3_F3 = 0
Casp3_A12 = 0
len_pre_casp3 = len(centers)
for cell in CT_Casp3.jitcells:
    casp3 = []
    a12 = []
    f3 = []
    for zid, z in enumerate(cell.zs[0]):
        mask = cell.masks[0][zid]
        img = CT_Casp3.hyperstack[0,z, channel_names.index("Casp3")]
        casp3.append(np.mean(img[mask[:,1], mask[:,0]]))
        
        img = CT_Casp3.hyperstack[0,z, channel_names.index("A12")]
        a12.append(np.mean(img[mask[:,1], mask[:,0]]))
        
        img = CT_Casp3.hyperstack[0,z, channel_names.index("F3")]
        f3.append(np.mean(img[mask[:,1], mask[:,0]]))

    centers.append(cell.centers[0]*[zres, xyres, xyres])
    labels.append(cell.label)

    zz = np.int64(cell.centers[0][0])
    idx = cell.zs[0].index(zz)
    if f3[idx] > a12[idx]:
        fates.append(2)
        Casp3_F3+=1
    else:
        fates.append(3)
        Casp3_A12+=1


print()
print(TIME)
print(COND)
print("F3", len(CT_F3.jitcells)+Casp3_F3)
print("A12", len(CT_A12.jitcells)+Casp3_A12)

print("F3 casp3", Casp3_F3)
print("A12 casp3", Casp3_A12)

print("F3 casp3 %", 100 * (Casp3_F3 / (len(CT_F3.jitcells)+Casp3_F3)))
print("A12 casp3 %", 100 * (Casp3_A12 / (len(CT_A12.jitcells)+Casp3_A12)))
print()

centers = np.array(centers)
fates = np.array(fates)
labels = np.array(labels)

from scipy.spatial import Delaunay

def find_neighbors(pindex, triang):
    neighbors = list()
    for simplex in triang.simplices:
        if pindex in simplex:
            neighbors.extend([simplex[i] for i in range(len(simplex)) if simplex[i] != pindex])
            '''
            this is a one liner for if a simplex contains the point we`re interested in,
            extend the neighbors list by appending all the *other* point indices in the simplex
            '''
    #now we just have to strip out all the dulicate indices and return the neighbors list:
    return list(set(neighbors))

# For the caspase cells, check fate of neighbors as a percentage

# tri = Delaunay(centers)
# neighs = []
# for p, point in enumerate(centers):
#     neighs_p = find_neighbors(p, tri)
#     neighs.append(neighs_p)

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(centers)
distances, neighs = nbrs.kneighbors(centers)

dist_th = (dim*xyres)*5.0 #microns
dist_th_near = (dim*xyres)*0.4
neighs_n = 4
true_neighs = []
true_dists = []
for p, neigh_p in enumerate(neighs):
    true_neigh_p = []
    true_dists_p = []
    for neigh in neigh_p[1:]:
        dist = np.linalg.norm(centers[p]-centers[neigh])
        if dist < dist_th:
            if dist > dist_th_near:
                if neigh < len_pre_casp3:
                    true_dists_p.append(dist)
                    true_neigh_p.append(neigh)
        if len(true_neigh_p) == neighs_n: break
    true_dists.append(true_dists_p)
    true_neighs.append(true_neigh_p)

neighs_fates = []
neighs_labels = []
for p, neigh_p in enumerate(true_neighs):
    lbs = []
    fts = []
    for neigh in neigh_p:
        fts.append(fates[neigh])
        lbs.append(labels[neigh])
    neighs_fates.append(fts)
    neighs_labels.append(lbs)

lab = 50

ch_F3 = channel_names.index("F3")
ch_A12 = channel_names.index("A12")
ch_Casp3 = channel_names.index("Casp3")


if fates[lab] == 0:
    CT = CT_F3
    chans = [ch_F3, ch_Casp3]
elif fates[lab] == 1:
    CT = CT_A12
    chans = [ch_A12, ch_Casp3]

else: raise Exception

label = labels[lab]
neighs = true_neighs[lab]
n_labs = [labels[l] for l in neighs if fates[l]==fates[lab]]

cell = CT._get_cell(label)


plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    'plot_stack_dims': (512, 512), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels': chans,
    'min_outline_length':75,
}

# CT.save_cells = True
# CT.plot_tracking(plot_args=plot_args, block_plot=False)
# CT_Casp3.plot_tracking(plot_args=plot_args, block_plot=False)


dists_F3 = []
dists_A12 = []
dists_Casp3_F3 = []
dists_Casp3_A12 = []

for n, dsts in enumerate(true_dists):
    fate = fates[n]
    if fate==0:
        dists_F3 = [*dists_F3, *dsts]
    elif fate==1:
        dists_A12 = [*dists_A12, *dsts]
    elif fate==2:
        dists_Casp3_F3 = [*dists_Casp3_F3, *dsts]
    elif fate==3:
        dists_Casp3_A12 = [*dists_Casp3_A12, *dsts]

import matplotlib.pyplot as plt

plt.hist(dists_F3, bins=50, color="green", edgecolor="k", density=True, alpha=0.5)
plt.hist(dists_A12, bins=50, color="magenta", edgecolor="k", density=True, alpha=0.5)
plt.hist(dists_Casp3_F3, bins=50, color="cyan", edgecolor="green", density=True, alpha=0.5)
plt.hist(dists_Casp3_A12, bins=50, color="red", edgecolor="magenta", density=True, alpha=0.5)

plt.show()