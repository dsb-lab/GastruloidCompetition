### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift
import numpy as np
import matplotlib.pyplot as plt
from qlivecell import remove_small_cells, plot_cell_sizes, correct_path

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

experiment_code = '2023_11_17_Casp3'

### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
# TIMES = ["48hr", "60hr", "72hr", "96hr"]
TIMES = ["96hr"]
CONDS = ["KO"]
path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/thesis/figures/debris_threshold/"

intens_F3_cells = []
intens_F3_debris = []

intens_A12_cells = []
intens_A12_debris = []

for T, TIME in enumerate(TIMES):
    intens_F3_cells.append([])
    intens_F3_debris.append([])
    intens_A12_cells.append([])
    intens_A12_debris.append([])

    path_figures_time = "{}{}/".format(path_figures, TIME)
    try: 
        files = get_file_names(path_figures_time)
    except: 
        import os
        os.mkdir(path_figures_time)
            
    for C, COND in enumerate(CONDS):
        intens_F3_cells[T].append([])
        intens_F3_debris[T].append([])
        intens_A12_cells[T].append([])
        intens_A12_debris[T].append([])

        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/stacks/{}/{}/'.format(experiment_code, TIME, COND)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/ctobjects_debris/{}/{}/'.format(experiment_code,TIME, COND)

        channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
        if "96hr" in path_data_dir:
            channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

        try: 
            files = get_file_names(path_save_dir)
        except: 
            import os
            os.mkdir(path_save_dir)
            
        ### GET FULL FILE NAME AND FILE CODE ###
        files = get_file_names(path_data_dir)
        
        for f, file in enumerate(files):
            path_data = path_data_dir+file
            file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
            path_save = correct_path(path_save_dir+embcode)

           ### DEFINE ARGUMENTS ###
            segmentation_args={
                'method': 'stardist2D', 
                'model': model, 
                'blur': [1,1], 
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
                'channels':[ch_F3]
            }
            
            chans = [ch_F3]
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
                'channels':[ch_A12]
            }

            chans = [ch_A12]
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

            ch_Casp3 = channel_names.index("Casp3")


            F3 = []
            A12 = []
            for cell in CT_F3.jitcells:
                center = cell.centers[0]
                z = int(center[0])
                zid = cell.zs[0].index(z)

                mask = cell.masks[0][zid]
                outline = cell.outlines[0][zid]
                F3.append(np.mean(CT_F3.hyperstack[0,z,ch_Casp3,:,:][mask[:,1], mask[:,0]]))

            for cell in CT_A12.jitcells:
                center = cell.centers[0]
                z = int(center[0])
                zid = cell.zs[0].index(z)

                mask = cell.masks[0][zid]
                outline = cell.outlines[0][zid]
                A12.append(np.mean(CT_A12.hyperstack[0,z,ch_Casp3,:,:][mask[:,1], mask[:,0]]))
                
            intens_f3_debris = []
            intens_f3_cells = []
            for cell in CT_F3.jitcells:
                zc = int(cell.centers[0][0])
                zcid = cell.zs[0].index(zc)

                mask = cell.masks[0][zcid]
                ints = np.mean(CT_A12.hyperstack[0,zc,ch_Casp3,:,:][mask[:,1], mask[:,0]])
                area = len(mask)* CT_F3.CT_info.xyresolution**2
                if area < 33.34:
                    intens_f3_debris.append(ints)
                else:
                    intens_f3_cells.append(ints)

            intens_f3_debris = np.array(intens_f3_debris) - np.mean(F3)
            intens_f3_cells = np.array(intens_f3_cells) - np.mean(F3)

            intens_a12_debris = []
            intens_a12_cells = []
            for cell in CT_A12.jitcells:
                zc = int(cell.centers[0][0])
                zcid = cell.zs[0].index(zc)

                mask = cell.masks[0][zcid]
                ints = np.mean(CT_A12.hyperstack[0,zc,ch_Casp3,:,:][mask[:,1], mask[:,0]])
                area = len(mask)* CT_A12.CT_info.xyresolution**2
                if area < 33.34:
                    intens_a12_debris.append(ints)
                else:
                    intens_a12_cells.append(ints)

            intens_a12_debris = np.array(intens_a12_debris) - np.mean(A12)
            intens_a12_cells = np.array(intens_a12_cells) - np.mean(A12)

            intens_F3_debris[T][C].append(intens_f3_debris)
            intens_F3_cells[T][C].append(intens_f3_cells)

            intens_A12_debris[T][C].append(intens_a12_debris)
            intens_A12_cells[T][C].append(intens_a12_cells)

import matplotlib as mpl
plt.rcParams.update({
    "text.usetex": True,
})
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath'
mpl.rc('font', size=16) 
mpl.rc('axes', labelsize=16) 
mpl.rc('xtick', labelsize=16) 
mpl.rc('ytick', labelsize=16) 
mpl.rc('legend', fontsize=16) 

debris_F3 = []
debris_A12 = []

cells_F3 = []
cells_A12 = []

for T, TIME in enumerate(TIMES):
    for C, COND in enumerate(CONDS):
        flat_list = [x for xs in intens_F3_debris[T][C] for x in xs]
        debris_F3 = [*debris_F3, *flat_list]

        flat_list = [x for xs in intens_A12_debris[T][C] for x in xs]
        debris_A12 = [*debris_A12, *flat_list]

        flat_list = [x for xs in intens_F3_cells[T][C] for x in xs]
        cells_F3 = [*cells_F3, *flat_list]

        flat_list = [x for xs in intens_A12_cells[T][C] for x in xs]
        cells_A12 = [*cells_A12, *flat_list]
        
fig, ax = plt.subplots(figsize=(4,4))
ax.hist(debris_F3, alpha=0.5, bins=200, density=True)
ax.hist(cells_F3, alpha=0.5, bins=200, density=True)
plt.tight_layout()
plt.show()

a = range(10)
np.percentile(a, 10)

np.nanmean(debris_F3)
np.nanmean(cells_F3)
np.nanmean(debris_A12)
np.nanmean(cells_A12)