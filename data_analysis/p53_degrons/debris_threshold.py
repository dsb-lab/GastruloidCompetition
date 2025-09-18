### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, check_or_create_dir, get_file_names
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

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


thresholds = []

# ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
CONDS = ["auxin_48-72_48", "auxin_48-72_72a", "auxin_48-72_72b" , "auxin_48-72_96", "auxin_72-96_96", "noauxin_72", "noauxin_96", "secondaryonly"]
# CONDS = ["auxin_48-72_72b"]


for COND in CONDS:        
    path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
    path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
        
    check_or_create_dir(path_save_dir)
    
    files = get_file_names(path_data_dir)
    
    channel_names = ["A12", "p53", "F3", "DAPI"]

    areas = []
    
    for f, file in enumerate(files):
    
        areas = []

        path_data = path_data_dir+file
        file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
        path_save = path_save_dir+embcode
            
        check_or_create_dir(path_save)

        ### DEFINE ARGUMENTS ###
        segmentation_args={
            'method': 'stardist2D', 
            'model': model, 
            'blur': [2,1], 
            'min_outline_length':100,
        }

        concatenation3D_args = {
            'do_3Dconcatenation': False
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
            'plot_stack_dims': (256, 256), 
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
        
        for cell in CT_F3.jitcells:
            zc = int(cell.centers[0][0])
            zcid = cell.zs[0].index(zc)

            mask = cell.masks[0][zcid]
            area = len(mask)* CT_F3.CT_info.xyresolution**2
            areas.append(area)

                    
        ch = channel_names.index("A12")
        batch_args = {
            'name_format':"ch"+str(ch)+"_{}",
            'extension':".tif",
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

        for cell in CT_A12.jitcells:
            zc = int(cell.centers[0][0])
            zcid = cell.zs[0].index(zc)

            mask = cell.masks[0][zcid]
            area = len(mask)* CT_A12.CT_info.xyresolution**2
            areas.append(area)


    fig, ax = plt.subplots(2,1, figsize=(5,8))
    threshold = 0
    data = np.array(areas)
    ax[0].hist(data, bins=200, color=[0.0, 0.8, 0.0], density=True, alpha=0.6,)

    x = np.arange(0, step=0.1, stop=np.max(data))
    bw = 7
    modelo_kde = KernelDensity(kernel="linear", bandwidth=bw)
    modelo_kde.fit(X=data.reshape(-1, 1))
    densidad_pred = np.exp(modelo_kde.score_samples(x.reshape((-1, 1))))
    ax[0].plot(x, densidad_pred, color="magenta")

    local_minima = argrelextrema(densidad_pred, np.less)[0]
    threshold = x[local_minima[0]]
    x_th = np.ones(len(x)) * x[local_minima[0]]
    y_th = np.linspace(0, np.max(densidad_pred), num=len(x))
    ax[0].plot(x_th, y_th, c="k", ls="--",lw=2, label="debris th.")

    thresholds.append(threshold)
    
    ax[0].set_ylabel("count")
    ax[1].set_ylabel("count")

    ax[1].hist(data, bins=200, color=[0.0, 0.8, 0.0], density=True, alpha=0.6)
    ax[1].set_xlabel(r"area ($\mu$m$^2$)")
    ax[1].plot(x, densidad_pred, color="magenta")
    ax[1].plot(x_th, y_th, c="k", ls="--",lw=2, label="debris th.")

    ax[1].set_xlim(-1, 75)
    plt.tight_layout()

    path_figs = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/"
    plt.savefig(path_figs+"debris_thresholds_{}.svg".format(COND))
    plt.savefig(path_figs+"debris_thresholds_{}.pdf".format(COND))
    plt.show()

print(thresholds)