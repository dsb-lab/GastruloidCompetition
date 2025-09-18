### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, check_or_create_dir, get_file_names

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

import numpy as np
import matplotlib.pyplot as plt


def build_union_masks(CT_list):
    """
    Build per-z 2D boolean masks marking in-cell pixels from the union of all cells
    across provided CT objects (e.g., CT_F3 and CT_A12).
    Returns: list of length Z with arrays (Y, X) dtype=bool.
    """
    CT0 = CT_list[0]
    Z = CT0.hyperstack.shape[1]
    Y = CT0.hyperstack.shape[-2]
    X = CT0.hyperstack.shape[-1]
    Mz_list = [np.zeros((Y, X), dtype=bool) for _ in range(Z)]
    for CT in CT_list:
        for cell in CT.jitcells:
            z = int(cell.centers[0][0])
            if z < 0 or z >= Z:
                continue
            # find mask for this z
            try:
                zid = cell.zs[0].index(z)
            except ValueError:
                continue
            mask = cell.masks[0][zid]
            yy = mask[:, 1].astype(np.intp)
            xx = mask[:, 0].astype(np.intp)
            Mz_list[z][yy, xx] = True
    return Mz_list

def estimate_b0z_for_file(CT, Mz_list, ch_B, ch_C, s_global, q=0.2):
    # q=0.5 (median) if few high-C cells; q=0.1–0.2 if many might be high
    import numpy as np
    Z = CT.hyperstack.shape[1]
    b0z = np.full(Z, np.nan, dtype=np.float64)
    for z in range(Z):
        Mz = Mz_list[z]
        if not np.any(Mz): continue
        Bz = CT.hyperstack[0, z, ch_B, :, :].astype(np.float64)
        Cz = CT.hyperstack[0, z, ch_C, :, :].astype(np.float64)
        resid = (Cz - s_global * Bz)[Mz].ravel()
        if resid.size < 50: continue
        b0z[z] = float(np.quantile(resid, q))
    # fill empties from available planes
    if np.any(np.isnan(b0z)):
        b0z[np.isnan(b0z)] = np.nanmedian(b0z)
    return b0z


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

calibF3 = np.load("/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/secondaryonly/F3(150)+OsTIR9-40(25)_48h_emiRFP-2ndaryA488-mCh-DAPI_(40xSil)_Stack1/calibration_F3_to_p53.npz")
p53_F3_s_global = float(calibF3["s"])
p53_F3_0z = calibF3["b0z"]

calibA12 = np.load("/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/secondaryonly/F3(150)+OsTIR9-40(25)_48h_emiRFP-2ndaryA488-mCh-DAPI_(40xSil)_Stack1/calibration_A12_to_p53.npz")
p53_A12_s_global = float(calibA12["s"])
p53_A12_0z = calibA12["b0z"]

def correct_cell_pixels(CT_ref, mask, z, ch_B, ch_C, s, b0z):
    """Return per-pixel corrected C for one cell at plane z."""
    yy = mask[:, 1].astype(np.intp)
    xx = mask[:, 0].astype(np.intp)
    C_vals = CT_ref.hyperstack[0, z, ch_C, :, :][yy, xx].astype(np.float32)
    B_vals = CT_ref.hyperstack[0, z, ch_B, :, :][yy, xx].astype(np.float32)
    return C_vals - float(b0z[z]) - float(s) * B_vals

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

        Mz_list = build_union_masks([CT_F3])
        p53_F3_0z = estimate_b0z_for_file(CT_F3, Mz_list, ch_F3, ch_p53, p53_F3_s_global)
        
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
            Ccorr_vals = correct_cell_pixels(CT_F3, mask, z, ch_F3, ch_p53, p53_F3_s_global, p53_F3_0z)
            F3_p53[z].append(float(np.mean(Ccorr_vals)))


            colors[z].append([0.0,0.8,0.0, 0.3])
        
        Mz_list = build_union_masks([CT_A12])
        p53_A12_0z = estimate_b0z_for_file(CT_A12, Mz_list, ch_A12, ch_p53, p53_A12_s_global)
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
            Ccorr_vals = correct_cell_pixels(CT_A12, mask, z, ch_A12, ch_p53, p53_A12_s_global, p53_A12_0z)
            A12_p53[z].append(float(np.mean(Ccorr_vals)))
        

            colors[z].append([0.8,0.0,0.8, 0.3])


all_vals = F3_p53
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



