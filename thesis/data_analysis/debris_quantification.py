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
TIMES = ["48hr", "72hr", "96hr"]
CONDS = ["WT", "KO"]
path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/thesis/figures/debris_threshold/"

areas_F3 = []

areas_A12 = []

for T, TIME in enumerate(TIMES):
    areas_F3.append([])
    areas_A12.append([])

    path_figures_time = "{}{}/".format(path_figures, TIME)
    try: 
        files = get_file_names(path_figures_time)
    except: 
        import os
        os.mkdir(path_figures_time)
            
    for C, COND in enumerate(CONDS):
        areas_F3[T].append([])
        areas_A12[T].append([])

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
                'channels':[ch]
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
                'channels':[ch]
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

            areas_f3 = []
            for cell in CT_F3.jitcells:
                zc = int(cell.centers[0][0])
                zcid = cell.zs[0].index(zc)

                msk = cell.masks[0][zcid]
                area = len(msk)
                areas_f3.append(area)

            areas_f3 = np.array(areas_f3) * CT_F3.CT_info.xyresolution**2

            areas_a12 = []
            for cell in CT_A12.jitcells:
                zc = int(cell.centers[0][0])
                zcid = cell.zs[0].index(zc)

                msk = cell.masks[0][zcid]
                area = len(msk)
                areas_a12.append(area)

            areas_a12 = np.array(areas_a12) * CT_F3.CT_info.xyresolution**2 
            
            areas_F3[T][C].append(areas_f3)
            areas_A12[T][C].append(areas_a12)

  
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

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/thesis/figures/counts/"

debris_F3 = []
debris_F3_std = []

debris_A12 = []
debris_A12_std = []

for T, TIME in enumerate(TIMES):
    debris_F3.append([])
    debris_F3_std.append([])
    debris_A12.append([])
    debris_A12_std.append([])
    for C, COND in enumerate(CONDS):
        deb_f3_wt = []
        for f in range(len(areas_F3[T][C])):
            data = np.array(areas_F3[T][C][f])
            deb_f3_wt.append(np.sum(data < 33.34)/len(data))
        debris_F3[T].append(np.mean(deb_f3_wt))
        debris_F3_std[T].append(np.std(deb_f3_wt))

        deb_a12_wt = []
        for f in range(len(areas_A12[T][C])):
            data = np.array(areas_A12[T][C][f])
            deb_a12_wt.append(np.sum(data < 33.34)/len(data))
        debris_A12[T].append(np.mean(deb_a12_wt))
        debris_A12_std[T].append(np.std(deb_a12_wt))

debris_F3 = np.array(debris_F3)
debris_F3_std = np.array(debris_F3_std)

debris_A12 = np.array(debris_A12)
debris_A12_std = np.array(debris_A12_std)

fig, ax = plt.subplots(2,1, figsize=(3.5,6))
ax[0].set_title("Debris")
ax[0].plot(range(3), debris_F3[:, 0], color=[0.0,0.8,0.0], lw=2, zorder=1)
ax[0].fill_between(range(3), debris_F3[:, 0]-debris_F3_std[:, 0], debris_F3[:, 0]+debris_F3_std[:, 0], color=[0.0,0.8,0.0], alpha=0.3)
ax[0].scatter(range(3), debris_F3[:, 0], color=[0.0,0.6,0.0], edgecolor='k', s=50, zorder=2)

ax[0].plot(range(3), debris_A12[:, 0], color=[0.8,0.0,0.8], lw=2, zorder=1)
ax[0].fill_between(range(3), debris_A12[:, 0]-debris_A12_std[:, 0], debris_A12[:, 0]+debris_A12_std[:, 0], color=[0.8,0.0,0.8], alpha=0.3)
ax[0].scatter(range(3), debris_A12[:, 0], color=[0.6,0.0,0.6], edgecolor='k', s=50, zorder=2)
ax[0].set_xticks([0,1,2])
ax[0].set_xticklabels([None,None,None])
ax[0].set_ylabel("debris fraction")

ax[1].plot(range(3), debris_F3[:, 1], color=[0.0,0.8,0.0], lw=2, zorder=1, label="F3")
ax[1].fill_between(range(3), debris_F3[:, 1]-debris_F3_std[:, 1], debris_F3[:, 1]+debris_F3_std[:, 1], color=[0.0,0.8,0.0], alpha=0.3)
ax[1].scatter(range(3), debris_F3[:, 1], color=[0.0,0.6,0.0], edgecolor='k', s=50, zorder=2)
ax[1].set_xticks([0,1,2])
ax[1].set_xticklabels(TIMES)

ax[1].plot(range(3), debris_A12[:, 1], color=[0.8,0.0,0.8], lw=2, zorder=1, label="A12")
ax[1].fill_between(range(3), debris_A12[:, 1]-debris_A12_std[:, 1], debris_A12[:, 1]+debris_A12_std[:, 1], color=[0.8,0.0,0.8], alpha=0.3)
ax[1].scatter(range(3), debris_A12[:, 1], color=[0.6,0.0,0.6], edgecolor='k', s=50, zorder=2)
ax[1].set_xticks([0,1,2])
ax[1].set_xticklabels(TIMES)
ax[1].set_ylabel("debris fraction")
ax[1].legend(loc="upper left")
ax[0].sharey(ax[1])

ax[0].spines[['right', 'top']].set_visible(False)
ax[1].spines[['right', 'top']].set_visible(False)

plt.tight_layout()
plt.savefig(path_figures+"debris.svg")
plt.savefig(path_figures+"debris.pdf")
plt.show()