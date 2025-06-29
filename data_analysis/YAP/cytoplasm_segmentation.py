### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, norm_stack_per_z, get_intenity_profile, get_file_names, construct_RGB, extract_fluoro, tif_reader_5D
import numpy as np
import matplotlib.pyplot as plt

from cellpose import models

model = models.CellposeModel(gpu=True, model_type='cyto3')
channel_names = ["A12", "GPI-GFP", "YAP", "DAPI"]

path_data_dir = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/YAP/2025_02_02_AiryscMultipl_FastMediumQuality_Files/KO8_ABonly/"
path_save_dir = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/YAP/segmentation_results/KO8_ABonly/"

try: 
    files = get_file_names(path_save_dir)
except: 
    import os
    os.mkdir(path_save_dir)

### GET FULL FILE NAME AND FILE CODE ###
files = get_file_names(path_data_dir)
                
ch_cyto = channel_names.index("GPI-GFP")
ch_nuc = channel_names.index("DAPI")
ch_yap = channel_names.index("YAP")
ch_a12 = channel_names.index("A12")

nuc_quant = []
cyt_quant = []
A12_quant = []

cells_cyto = 0
cells_nuc = 0

cells_assigned = 0
cells_final = 0

cyt_nuc_ratio = 0.95
    
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

    hyperstack, metadata = tif_reader_5D(path_data)

    segmentation_args={
    'method': 'cellpose2D', 
    'model': model, 
    'blur': None, 
    'channels': [ch_cyto+1,ch_nuc+1],
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

    chans = [ch_nuc]
    for _ch in range(len(channel_names)):
        if _ch not in chans:
            chans.append(_ch)

    batch_args = {
        'name_format':"ch"+str(ch_cyto)+"_{}",
        'extension':".tif",
    }

    plot_args = {
        'plot_layout': (1,1),
        'plot_overlap': 1,
        'masks_cmap': 'tab10',
        'plot_stack_dims': (256, 256), 
        'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
        'channels':[ch_nuc],
        # 'channels': chans_plot,
        'min_outline_length':75,
    }

    CT_cyto = cellSegTrack(
        path_data,
        path_save,
        segmentation_args=segmentation_args,
        concatenation3D_args=concatenation3D_args,
        error_correction_args=error_correction_args,
        plot_args=plot_args,
        batch_args=batch_args,
        channels=chans
    )

    CT_cyto.run()
    # CT_cyto.plot(plot_args)

    segmentation_args={
    'method': 'cellpose2D', 
    'model': model, 
    'blur': None, 
    'channels': [ch_nuc+1, 0],
    # 'flow_threshold': 0.4,
    'diameter': 200,
    }

    chans = [ch_nuc]
    for _ch in range(len(channel_names)):
        if _ch not in chans:
            chans.append(_ch)

    batch_args = {
        'name_format':"ch"+str(ch_nuc)+"_{}",
        'extension':".tif",
    }


    CT_nuc = cellSegTrack(
        path_data,
        path_save,
        segmentation_args=segmentation_args,
        concatenation3D_args=concatenation3D_args,
        error_correction_args=error_correction_args,
        plot_args=plot_args,
        batch_args=batch_args,
        channels=chans
    )

    CT_nuc.run()
    # CT_nuc.plot(plot_args)

    from qlivecell.celltrack.core.tracking.tracking import greedy_tracking
    
    TLabels1 = [cell.label for cell in CT_cyto.jitcells]
    TLabels2 = [cell.label for cell in CT_nuc.jitcells]
    TLabels  = [TLabels1, TLabels2]

    TCenters1 = [cell.centers[0] for cell in CT_cyto.jitcells]
    TCenters2 = [cell.centers[0] for cell in CT_nuc.jitcells]
    TCenters  = [TCenters1, TCenters2]
    
    track_args = {
            "time_step": 1,
            "method": "greedy",
            "dist_th": 7.5,
            "z_th": 2,
        }
    
    xyres = CT_nuc.metadata["XYresolution"]
    zres = CT_nuc.metadata["Zresolution"]

    FinalLabels, label_correspondance = greedy_tracking(TLabels, TCenters, xyres, zres, track_args, lab_max=0)
    
    cells_cyto += len(CT_cyto.jitcells)
    cells_nuc += len(CT_nuc.jitcells)

    cells_assigned = 0
    cells_final = 0
    
    cyt_nuc_ratio = 0.95
    
    for lc in label_correspondance[1]:
        cell1 = CT_cyto._get_cell(lc[1])
        if cell1==None:continue
        cell2 = CT_nuc._get_cell(lc[0])

        z = int(cell1.centers[0][0])
        mask1 = cell1.masks[0][0]
        mask2 = cell2.masks[0][0]
        
        # Assume: mask1 and mask2 are (N, 2) arrays of 2D points
        mask1_view = mask1.view(dtype=[('', mask1.dtype)] * mask1.shape[1])
        mask2_view = mask2.view(dtype=[('', mask2.dtype)] * mask2.shape[1])
        
        cells_assigned += 1
        
        mask_coincidence = np.isin(mask2_view, mask1_view)
        mask_ratio = np.sum(mask_coincidence)/len(mask2)
        if mask_ratio < cyt_nuc_ratio: 
            continue
        
        if len(mask1)/2.0 > len(mask2):
            continue
        
        cells_final +=1
        
        img = hyperstack[0,z,ch_yap]
        nuc_quant.append(np.mean(img[mask2[:, 1], mask2[:, 0]]))
        
        img = hyperstack[0,z, ch_a12]
        A12_quant.append(np.mean(img[mask2[:, 1], mask2[:, 0]]))

        mask_coincidence_cyto = ~np.isin(mask1_view, mask2_view)
        cyto_mask = np.array([mask1[i] for i, j in enumerate(mask_coincidence_cyto) if j])
        img = hyperstack[0,z,ch_yap]
        cyt_quant.append(np.mean(img[cyto_mask[:, 1], cyto_mask[:, 0]]))
        
        # fig, ax = plt.subplots()
        
        # ax.imshow(CT_cyto.hyperstack[0,z,-1])
        # ax.scatter(cell1.outlines[0][0][:,0], cell1.outlines[0][0][:,1], label="cyto")
        # ax.scatter(cell2.outlines[0][0][:,0], cell2.outlines[0][0][:,1], label="nuc")
        # ax.set_xlim(np.min(cell1.outlines[0][0][:,0])-10, np.max(cell1.outlines[0][0][:,0])+10)
        # ax.set_ylim(np.min(cell1.outlines[0][0][:,1])-10, np.max(cell1.outlines[0][0][:,1])+10)
        # ax.legend()
        # plt.show()

print(cells_cyto)
print(cells_nuc)
print(cells_assigned)
print(cells_final)

plt.scatter(nuc_quant, cyt_quant)
plt.xlabel("nuclear")
plt.ylabel("cytoplasmic")
plt.show()

plt.hist(A12_quant, bins=50)
plt.show()

nuc_quant_WT = [nuc_quant[i] for i in range(len(nuc_quant)) if A12_quant[i]<10300]
nuc_quant_A12 = [nuc_quant[i] for i in range(len(nuc_quant)) if A12_quant[i]>=11000]

cyt_quant_WT = [cyt_quant[i] for i in range(len(cyt_quant)) if A12_quant[i]<10300]
cyt_quant_A12 = [cyt_quant[i] for i in range(len(cyt_quant)) if A12_quant[i]>=11000]

fig, ax = plt.subplots()
bins=50
ax.hist(np.array(nuc_quant)/cyt_quant, label="N/C", alpha=0.5, bins=bins, color="grey")
ax.legend()
plt.show()

fig, ax = plt.subplots()
bins=40
ax.hist(np.array(nuc_quant_WT), label="WT", alpha=0.5, bins=bins, color="green", density=True)
ax.hist(np.array(nuc_quant_A12), label="A12", alpha=0.5, bins=bins, color="purple", density=True)
ax.set_xlabel("nuclear YAP")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.hist(np.array(cyt_quant_WT), label="WT", alpha=0.5, bins=bins, color="green", density=True)
ax.hist(np.array(cyt_quant_A12), label="A12", alpha=0.5, bins=bins, color="purple", density=True)
ax.set_xlabel("cytoplasmic YAP")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.hist(np.array(nuc_quant_WT)/cyt_quant_WT, label="WT", alpha=0.5, bins=bins, color="green", density=True)
ax.hist(np.array(nuc_quant_A12)/cyt_quant_A12, label="A12", alpha=0.5, bins=15, color="purple", density=True)
ax.set_xlabel("N/C YAP")
ax.legend()
plt.show()

