### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, norm_stack_per_z, get_intenity_profile, get_file_names, construct_RGB, extract_fluoro, tif_reader_5D
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
# from stardist.models import StarDist2D
# model = StarDist2D.from_pretrained('2D_versatile_fluo')

from cellpose import models
# model  = models.CellposeModel(gpu=True, pretrained_model='/home/pablo/Desktop/PhD/projects/Data/models/cellpose/blasto')

model = models.CellposeModel(gpu=True, model_type='cyto3')
channel_names = ["A12", "GPI-GFP", "YAP", "DAPI"]

path_data_dir = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/YAP/2025_02_02_AiryscMultipl_FastMediumQuality_Files/"
path_save_dir = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/YAP/segmentation_results/"

try: 
    files = get_file_names(path_save_dir)
except: 
    import os
    os.mkdir(path_save_dir)

### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data_dir)
areas = []
                
for file in files:
    if not ".tif" in file: continue
    
    file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
    print(file)
    
    path_data = path_data_dir+file
    path_save = path_save_dir+embcode
    try: 
        files = get_file_names(path_save)
    except: 
        import os
        os.mkdir(path_save)
        
    ### DEFINE ARGUMENTS ###
    # segmentation_args={
    #     'method': 'stardist2D', 
    #     'model': model, 
    #     'blur': [10,5], 
    # }

    ch1 = channel_names.index("GPI-GFP")
    ch2 = channel_names.index("DAPI")

    segmentation_args={
    'method': 'cellpose2D', 
    'model': model, 
    'blur': None, 
    'channels': [ch1+1,ch2+1],
    # 'flow_threshold': 0.4,
    'diameter': 240,
    }

    concatenation3D_args = {
        'distance_th_z': 5.0, # microns
        'relative_overlap':False, 
        'use_full_matrix_to_compute_overlap':True, 
        'z_neighborhood':2, 
        'overlap_gradient_th':0.1, 
        'min_cell_planes': 1,
    }

    error_correction_args = {
        'backup_steps': 10,
        'line_builder_mode': 'points',
    }

    chans = [ch2]
    for _ch in range(len(channel_names)):
        if _ch not in chans:
            chans.append(_ch)

    batch_args = {
        'name_format':"ch"+str(ch1)+"_{}",
        'extension':".tif",
    }

    plot_args = {
        'plot_layout': (1,1),
        'plot_overlap': 1,
        'masks_cmap': 'tab10',
        'plot_stack_dims': (256, 256), 
        'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
        'channels':[ch2],
        # 'channels': chans_plot,
        'min_outline_length':75,
    }

    CT = cellSegTrack(
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
    # CT.plot(plot_args)
            
            # labs_to_rem = []
            # for cell in CT.jitcells:
            #     zc = int(cell.centers[0][0])
            #     zcid = cell.zs[0].index(zc)

            #     mask = cell.masks[0][zcid]
            #     area = len(mask) / CT.metadata["XYresolution"]**2
            #     if area < size_th:
            #         labs_to_rem.append(cell.label)
                
            # for lab in labs_to_rem:
            #     CT._del_cell(lab)  

            # CT.update_labels()
            
    for cell in CT.jitcells:
        zc = int(cell.centers[0][0])
        zcid = cell.zs[0].index(zc)

        msk = cell.masks[0][zcid]
        area = len(msk)
        areas.append(area)
                    

plt.hist(areas, bins=75)
plt.show()
hyperstack, metadata = tif_reader_5D(path_to_file=path_data)

img = hyperstack[0, 2]

model = segmentation_args["model"]
masks, flows, styles = model.eval(img, channels=[1,4], diameter= 240)

from cellpose.utils import outlines_list
outlines = outlines_list(masks)

fig, ax = plt.subplots()
ax.imshow(img[-1])

for outline in outlines:
    ax.scatter(outline[:, 0], outline[:,1], s=1)
plt.show()
