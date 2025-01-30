### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift
import numpy as np
import matplotlib.pyplot as plt
from qlivecell import remove_small_cells, plot_cell_sizes, correct_path

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# experiment_code = '2023_11_17_Casp3'
experiment_code = "2024_03_Casp3"

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

        # channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
        # if "96hr" in path_data_dir:
        #     channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

        channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
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
mpl.rc('font', size=14) 
mpl.rc('axes', labelsize=14) 
mpl.rc('xtick', labelsize=14) 
mpl.rc('ytick', labelsize=14) 
mpl.rc('legend', fontsize=14) 

from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity

data1 = np.array([a for lst in areas_F3[-1][0] for a in lst])
data2 = np.array([a for lst in areas_A12[-1][0] for a in lst])
data3 = np.array([a for lst in areas_F3[-1][-1] for a in lst])
data4 = np.array([a for lst in areas_A12[-1][-1] for a in lst])

fig, ax = plt.subplots(1,4, figsize=(14,4))
ax[0].set_title("96 hr F3-WT")
ax[0].hist(data1, bins=30, color=[0.0, 0.8, 0.0], density=True, alpha=0.6, label="F3")
ax[0].set_xlabel(r"area ($\mu$m$^2$)")

x = np.arange(0, step=0.1, stop=np.max(data1))
bw = 12
modelo_kde = KernelDensity(kernel="linear", bandwidth=bw)
modelo_kde.fit(X=data1.reshape(-1, 1))
densidad_pred = np.exp(modelo_kde.score_samples(x.reshape((-1, 1))))
local_minima = argrelextrema(densidad_pred, np.less)[0]

x_th = np.ones(len(x)) * x[local_minima[0]]
y_th = np.linspace(0, np.max(densidad_pred), num=len(x))
ax[0].plot(x_th, y_th, c="k", ls="--",lw=2, label="debris th.")
handles1, labels1 = ax[0].get_legend_handles_labels(legend_handler_map=None)

ax[0].plot(x, densidad_pred, lw=5, label="kde", color=[0.0, 0.6, 0.0])
ax[0].set_xlabel("area (µm²)")
ax[0].set_yticks([])
ax[0].spines[['right', 'top', 'left']].set_visible(False)
ax[0].set_xlim(-1,150)

ax[1].set_title("96 hr A12-WT")
ax[1].hist(data2, bins=25, color=[0.8, 0.0, 0.8], alpha=0.6, density=True, label="A12")
handles2, labels2 = ax[1].get_legend_handles_labels(legend_handler_map=None)

ax[1].set_xlabel(r"area ($\mu$m$^2$)")
x = np.arange(0, step=0.1, stop=np.max(data2))
bw = 12
modelo_kde = KernelDensity(kernel="linear", bandwidth=bw)
modelo_kde.fit(X=data2.reshape(-1, 1))
densidad_pred = np.exp(modelo_kde.score_samples(x.reshape((-1, 1))))
local_minima = argrelextrema(densidad_pred, np.less)[0]

x_th = np.ones(len(x)) * x[local_minima[0]]
y_th = np.linspace(0, np.max(densidad_pred), num=len(x))
ax[1].plot(x_th, y_th, c="k", ls="--",lw=2)

ax[1].plot(x, densidad_pred, lw=5, label="kde", color=[0.6, 0.0, 0.6])
ax[1].set_xlabel("area (µm²)")
ax[1].set_yticks([])
ax[1].spines[['right', 'top', 'left']].set_visible(False)
ax[1].set_xlim(-1,150)

ax[2].set_title("96 hr F3-KO")
ax[2].hist(data3, bins=25, color=[0.0, 0.8, 0.0], alpha=0.6, density=True)
ax[2].set_xlabel(r"area ($\mu$m$^2$)")
x = np.arange(0, step=0.1, stop=np.max(data3))
bw = 12
modelo_kde = KernelDensity(kernel="linear", bandwidth=bw)
modelo_kde.fit(X=data3.reshape(-1, 1))
densidad_pred = np.exp(modelo_kde.score_samples(x.reshape((-1, 1))))
local_minima = argrelextrema(densidad_pred, np.less)[0]

x_th = np.ones(len(x)) * x[local_minima[0]]
y_th = np.linspace(0, np.max(densidad_pred), num=len(x))
ax[2].plot(x_th, y_th, c="k", ls="--",lw=2)

ax[2].plot(x, densidad_pred, lw=5, label="kde", color=[0.0, 0.6, 0.0])
ax[2].set_xlabel("area (µm²)")
ax[2].set_yticks([])
ax[2].spines[['right', 'top', 'left']].set_visible(False)
ax[2].set_xlim(-1,150)

ax[3].set_title("96 hr A12-KO")
ax[3].hist(data4, bins=25, color=[0.8, 0.0, 0.8], alpha=0.6, density=True)
ax[3].set_xlabel(r"area ($\mu$m$^2$)")

x = np.arange(0, step=0.1, stop=np.max(data4))
bw = 12
modelo_kde = KernelDensity(kernel="linear", bandwidth=bw)
modelo_kde.fit(X=data4.reshape(-1, 1))
densidad_pred = np.exp(modelo_kde.score_samples(x.reshape((-1, 1))))
local_minima = argrelextrema(densidad_pred, np.less)[0]

x_th = np.ones(len(x)) * x[local_minima[0]]
y_th = np.linspace(0, np.max(densidad_pred), num=len(x))
# ax[3].plot(x_th, y_th, c="k", ls="--",lw=2)

ax[3].plot(x, densidad_pred, lw=5, label="kde", color=[0.6, 0.0, 0.6])
ax[3].set_xlabel("area (µm²)")
ax[3].set_yticks([])
ax[3].spines[['right', 'top', 'left']].set_visible(False)
ax[3].set_xlim(-1,150)


import matplotlib.patches as mpatches

handles=[handles1[0], handles2[0], handles1[1]]
labels=[labels1[0], labels2[0], labels1[1]]

ax[3].legend(handles=handles, labels=labels, loc='upper right')

plt.tight_layout()
# plt.savefig(path_figures+"debris_threshold.svg")
# plt.savefig(path_figures+"debris_threshold.pdf")
plt.show()