from qlivecell import get_file_name, cellSegTrack, get_file_names, tif_reader_5D
import matplotlib.pyplot as plt

import matplotlib as mpl
 
plt.rcParams.update({
    "text.usetex": True,
})
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath'
mpl.rc('font', size=18) 
mpl.rc('axes', labelsize=18) 
mpl.rc('xtick', labelsize=18) 
mpl.rc('ytick', labelsize=18) 
mpl.rc('legend', fontsize=18) 

from cellpose import models
from cellpose.utils import outlines_list
model = models.CellposeModel(gpu=True, model_type='cyto3')

### LOAD STARDIST MODEL ###
# from stardist.models import StarDist2D
# from csbdeep.utils import normalize
# model = StarDist2D.from_pretrained('2D_versatile_fluo')

channel_names = ["A12", "GPI-GFP", "YAP", "DAPI"]

### GET FULL FILE NAME AND FILE CODE ###
size_th = 30.0 #µm

ch = channel_names.index("DAPI")


segmentation_args={
'method': 'cellpose2D', 
'model': model, 
'blur': None, 
'channels': [ch+1, 0],
'diameter': 200,
}

# segmentation_args={
#     'method': 'stardist2D', 
#     'model': model, 
#     'blur': None, 
#     # 'n_tiles': (2,2),
# }
    
CONDITIONS = ["WT", "KO8", "KO25"]

from qlivecell.celltrack.core.tools.tools import mask_from_outline, get_outlines_masks_labels
import numpy as np
import skimage

COND = CONDITIONS[0]
path_data_dir = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/YAP/2025_02_02_AiryscMultipl_FastMediumQuality_Files/{}/".format(COND)
files = get_file_names(path_data_dir) 
files = [file for file in files if ".tif" in file]   
file = files[2]    
file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
path_data = path_data_dir+file

hyperstack, metadata = tif_reader_5D(path_to_file=path_data)
img = hyperstack[0, 3, -1]
# img = skimage.filters.gaussian(
#                 img, sigma=5, truncate=1
#             )
model = segmentation_args["model"]
# labels, _ = model.predict_instances(normalize(img), scale=0.15)
# outlines, masks, labs = get_outlines_masks_labels(labels)

masks, flows, styles = model.eval(img, channels=segmentation_args['channels'], diameter=segmentation_args['diameter'])
outlines = outlines_list(masks)

cells_to_rem = []
for o, outline in enumerate(outlines):
    if len(outline)<5:
        cells_to_rem.append(o)
        continue
    mask = mask_from_outline(outline)
    area = len(mask) * metadata["XYresolution"]**2
    if area < size_th:
        print(area)
        cells_to_rem.append(o)

cells_to_rem.sort(reverse=True)

for cell in cells_to_rem:
    outlines.pop(cell)
    pass

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(img)

for outline in outlines:
    ax.scatter(outline[:, 0], outline[:,1], s=0.01)

ax.axis('off')
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/YAP/segmentation_examples/nucleus.svg", dpi=300)
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/YAP/segmentation_examples/nucleus.png", dpi=300)
plt.show()
