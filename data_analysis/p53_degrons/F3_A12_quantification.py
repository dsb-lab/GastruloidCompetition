### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, check_or_create_dir, get_file_names

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

import numpy as np
import matplotlib.pyplot as plt

F3_all = [[] for z in range(10)]
F3_F3 = [[] for z in range(10)]
F3_A12 = [[] for z in range(10)]
F3_DAPI = [[] for z in range(10)]

A12_all = [[] for z in range(10)]
A12_F3 = [[] for z in range(10)]
A12_A12 = [[] for z in range(10)]
A12_DAPI = [[] for z in range(10)]

DAPI_all = [[] for z in range(10)]

colors = [[] for z in range(10)]

files_to_exclude = [
    "n2_F3(150)+WT(150)_72h_emiRFP-p53-mCh-DAPI_(40xSil)_Stack1.tif",
    "n2_F3(150)+WT(150)_72h_emiRFP-p53-mCh-DAPI_(40xSil)_Stack2.tif"
]

zs = []
all_files = []

CONDS = ["auxin48", "auxin72", "noauxin72" , "secondaryonly"]

for C, COND in enumerate(CONDS):
    path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
    path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
        
    check_or_create_dir(path_save_dir)

    files = get_file_names(path_data_dir)

    channel_names = ["A12", "p53", "F3", "DAPI"]

    for f, file in enumerate(files):
        
        if file in files_to_exclude: continue
        
        all_files.append(file)
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

        ch = channel_names.index("A12")
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
        zs.append(CT_A12.hyperstack.shape[1])
        
        ch_F3 = channel_names.index("F3")
        ch_A12 = channel_names.index("A12")
        ch_DAPI = channel_names.index("DAPI")

        for cell in CT_F3.jitcells:
            center = cell.centers[0]
            z = int(center[0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]

            F3_all[z].append(np.mean(CT_A12.hyperstack[0,z,ch_F3,:,:][mask[:,1], mask[:,0]]))
            A12_all[z].append(np.mean(CT_A12.hyperstack[0,z,ch_A12,:,:][mask[:,1], mask[:,0]]))
            DAPI_all[z].append(np.mean(CT_A12.hyperstack[0,z,ch_DAPI,:,:][mask[:,1], mask[:,0]]))

            F3_F3[z].append(np.mean(CT_A12.hyperstack[0,z,ch_F3,:,:][mask[:,1], mask[:,0]]))
            F3_A12[z].append(np.mean(CT_A12.hyperstack[0,z,ch_A12,:,:][mask[:,1], mask[:,0]]))
            F3_DAPI[z].append(np.mean(CT_A12.hyperstack[0,z,ch_DAPI,:,:][mask[:,1], mask[:,0]]))

            colors[z].append([0.0,0.8,0.0, 0.3])
            
        for cell in CT_A12.jitcells:
            center = cell.centers[0]
            z = int(center[0])
            zid = cell.zs[0].index(z)
            mask = cell.masks[0][zid]

            F3_all[z].append(np.mean(CT_A12.hyperstack[0,z,ch_F3,:,:][mask[:,1], mask[:,0]]))
            A12_all[z].append(np.mean(CT_A12.hyperstack[0,z,ch_A12,:,:][mask[:,1], mask[:,0]]))
            DAPI_all[z].append(np.mean(CT_A12.hyperstack[0,z,ch_DAPI,:,:][mask[:,1], mask[:,0]]))
            
            A12_F3[z].append(np.mean(CT_A12.hyperstack[0,z,ch_F3,:,:][mask[:,1], mask[:,0]]))
            A12_A12[z].append(np.mean(CT_A12.hyperstack[0,z,ch_A12,:,:][mask[:,1], mask[:,0]]))
            A12_DAPI[z].append(np.mean(CT_A12.hyperstack[0,z,ch_DAPI,:,:][mask[:,1], mask[:,0]]))                
            colors[z].append([0.8,0.0,0.8, 0.3])

for z in range(zs[0]):
    fig, ax = plt.subplots()
    ax.set_xlabel("mean H2B-mCherry [a.u.]")
    ax.set_ylabel("mean H2B-emiRFP [a.u.]")            
    ax.scatter(F3_all[z], A12_all[z],c=colors[z], edgecolors='none')
    # ax.set_xlim(-5, 20000)
    # ax.set_ylim(-5, 20000)
    # ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()

F3_means = np.array([np.mean(f3) for f3 in F3_F3])
F3_stds = np.array([np.std(f3) for f3 in F3_F3])

A12_means = np.array([np.mean(a12) for a12 in A12_A12])
A12_stds = np.array([np.std(a12) for a12 in A12_A12])

DAPI_means = np.array([np.mean(dapi) for dapi in DAPI_all])
DAPI_stds = np.array([np.std(dapi) for dapi in DAPI_all])

fig, ax = plt.subplots()

ax.plot(range(10), F3_means, color=[0.0,0.8,0.0, 1.0], label="H2B-mCherry on F3")
ax.fill_between(range(10), F3_means - F3_stds, F3_means + F3_stds, color=[0.0,0.8,0.0], alpha=0.2)

ax.plot(range(10), A12_means, color=[0.8,0.0,0.8, 1.0], label="H2B-emiRFP on A12")
ax.fill_between(range(10), A12_means - A12_stds, A12_means + A12_stds, color=[0.8,0.0,0.8], alpha=0.2)

ax.plot(range(10), DAPI_means, color="cyan", label="DAPI on all cells")
ax.fill_between(range(10), DAPI_means - DAPI_stds, DAPI_means + DAPI_stds, color="cyan", alpha=0.2)

ax.set_xlabel("z")
ax.set_ylabel("fluoro [a.u.]")
ax.title("Mean with std ribbon")
ax.legend()
plt.show()
