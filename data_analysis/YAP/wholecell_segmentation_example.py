### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, get_file_names, tif_reader_5D, construct_RGB
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
model = models.CellposeModel(gpu=True, model_type='cyto3')

channel_names = ["A12", "GPI-GFP", "YAP", "DAPI"]

size_th = 50.0 #µm

ch1 = channel_names.index("GPI-GFP")
ch2 = channel_names.index("DAPI")

segmentation_args={
'method': 'cellpose2D', 
'model': model, 
'blur': None, 
'channels': [ch1+1,ch2+1],
'diameter': 240,
}

CONDITIONS = ["WT", "KO8", "KO25"]

from cellpose.utils import outlines_list
from qlivecell.celltrack.core.tools.tools import mask_from_outline

COND = CONDITIONS[0]
path_data_dir = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/YAP/2025_02_02_AiryscMultipl_FastMediumQuality_Files/{}/".format(COND)
files = get_file_names(path_data_dir) 
files = [file for file in files if ".tif" in file]   
file = files[2]    
file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
path_data = path_data_dir+file

hyperstack, metadata = tif_reader_5D(path_to_file=path_data)
img = hyperstack[0, 3]

model = segmentation_args["model"]
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

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(img[-1])

for outline in outlines:
    ax.scatter(outline[:, 0], outline[:,1], s=1)

ax.axis('off')
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/YAP/segmentation_examples/wholecell.svg", dpi=300)
plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/YAP/segmentation_examples/wholecell.png", dpi=300)
plt.show()
