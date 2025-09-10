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
F3_p53 = [[] for z in range(10)]

F3_p53_WT = [[] for z in range(10)]
F3_p53_KO = [[] for z in range(10)]

A12_p53_WT = [[] for z in range(10)]
A12_p53_KO = [[] for z in range(10)]

A12_all = [[] for z in range(10)]
A12_F3 = [[] for z in range(10)]
A12_A12 = [[] for z in range(10)]
A12_DAPI = [[] for z in range(10)]
A12_p53 = [[] for z in range(10)]

DAPI_all = [[] for z in range(10)]

colors = [[] for z in range(10)]

zs = []

all_files = []

CONDS = ["auxin48", "auxin72", "noauxin72"]
F3_backgrounds = np.array([2496.39216396, 2717.58109258, 2523.45981179, 2206.13498618, 2097.02696383, 1921.61948137, 1828.3545534, 1700.845861, 1557.78809637, 1506.0134047])
A12_backgrounds = np.array([1615.10968548, 1539.48146973, 1527.88985255, 1458.18400646, 1404.53034921, 1352.31062397, 1306.44659299, 1262.34521779, 1203.13280638, 1176.99222449])

for C, COND in enumerate(CONDS):
    path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
    path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
     
    check_or_create_dir(path_save_dir)

    files = get_file_names(path_data_dir)

    channel_names = ["A12", "p53", "F3", "DAPI"]

    for f, file in enumerate(files):
            
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
        ch_p53 = channel_names.index("p53")
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
            F3_p53[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]] - F3_backgrounds[z]))
            
            if COND=="WT":
                F3_p53_WT[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]] - F3_backgrounds[z]))
            else:
                F3_p53_KO[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]] - F3_backgrounds[z]))

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
            A12_p53[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]] - A12_backgrounds[z]))
            
            if COND=="WT":
                A12_p53_WT[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]] - A12_backgrounds[z]))
            else:
                A12_p53_KO[z].append(np.mean(CT_A12.hyperstack[0,z,ch_p53,:,:][mask[:,1], mask[:,0]] - A12_backgrounds[z]))

            colors[z].append([0.8,0.0,0.8, 0.3])


all_vals = [[*F3_p53_WT[z], *F3_p53_KO[z]] for z in range(10)]
iqr_outlier_threshold = 3.5

extreme_threshold = []
# Overlay individual points (WT)
z_vals = np.arange(len(F3_p53_WT))
for z in z_vals:
    q1, q3 = np.percentile(np.array(all_vals[z]), [25, 75])
    iqr = q3 - q1
    upper = q3 + iqr_outlier_threshold * iqr
    extreme_threshold.append(upper)

print("Thresholds")
print(extreme_threshold)



