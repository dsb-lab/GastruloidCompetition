### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, check_or_create_dir, get_file_names, get_intenity_profile

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

import numpy as np
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

files_to_exclude = [
    "n2_F3(150)+WT(150)_72h_emiRFP-p53-mCh-DAPI_(40xSil)_Stack1.tif",
    "n2_F3(150)+WT(150)_72h_emiRFP-p53-mCh-DAPI_(40xSil)_Stack2.tif"
]

CONDS = ["WT", "KO"]

repeats = ["n2", "n3", "n4"]

zs = []

all_files = []

path_z_drift_figures_p53 = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53/z_drift_correction_p53/"
path_z_drift_figures_F3 = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53/z_drift_correction_F3/"
path_z_drift_figures_A12 = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53/z_drift_correction_A12/"


calibF3 = np.load("/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/secondaryonly/F3(150)+OsTIR9-40(25)_48h_emiRFP-2ndaryA488-mCh-DAPI_(40xSil)_Stack1/calibration_F3_to_p53.npz")
p53_F3_s_global = float(calibF3["s"])
p53_F3_0z = calibF3["b0z"]

calibA12 = np.load("/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/secondaryonly/F3(150)+OsTIR9-40(25)_48h_emiRFP-2ndaryA488-mCh-DAPI_(40xSil)_Stack1/calibration_A12_to_p53.npz")
p53_A12_s_global = float(calibA12["s"])
p53_A12_0z = calibA12["b0z"]


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


def correct_cell_pixels(CT_ref, mask, z, ch_B, ch_C, s, b0z):
    """Return per-pixel corrected C for one cell at plane z."""
    yy = mask[:, 1].astype(np.intp)
    xx = mask[:, 0].astype(np.intp)
    C_vals = CT_ref.hyperstack[0, z, ch_C, :, :][yy, xx].astype(np.float32)
    B_vals = CT_ref.hyperstack[0, z, ch_B, :, :][yy, xx].astype(np.float32)
    return C_vals - float(b0z[z]) - float(s) * B_vals


def get_intenity_profile_spill(CT, ch_spill, ch_quant, s_global, b0z, cell_number_threshold=2, fit_everything=True):
    intensity_per_z = np.zeros(CT.slices)
    intensity_per_z_n = np.zeros(CT.slices)
    for cell in CT.jitcells:
        zc = int(cell.centers[0][0])
        zcid = cell.zs[0].index(zc)
        msk = cell.masks[0][zcid]
        
        Ccorr_vals = correct_cell_pixels(CT, msk, zc, ch_spill, ch_quant, s_global, b0z)
        intensity = float(np.mean(Ccorr_vals))

        intensity_per_z_n[zc] += 1
        intensity_per_z[zc] += intensity

    # intensity_per_z = np.array([intensity_per_z[zc]  if intensity_per_z_n[zc]>cell_number_threshold else 0 for zc in range(len(intensity_per_z))])
    intensity_per_z_n[intensity_per_z_n < cell_number_threshold] = 0
    zs = np.where(intensity_per_z_n != 0)[0]
    data_z = intensity_per_z[zs] / intensity_per_z_n[zs]
    data_z_filled = []
    zs_filled = []
    for z in range(zs[0], zs[-1] + 1):
        if z in zs:
            zid = np.where(zs == z)[0][0]
            data_z_filled.append(data_z[zid])
            zs_filled.append(z)
        else:
            if z + 1 in zs:
                zid = np.where(zs == z + 1)[0][0]
                data_z_filled.append(np.mean([data_z[zid], data_z_filled[-1]]))
                zs_filled.append(z)
            else:
                data_z_filled.append(data_z_filled[-1])
                zs_filled.append(z)
    # Measure intensity profile along the z-axis
    intensity_profile = np.array(data_z_filled)

    # Define z-axis positions
    z_positions = np.array(zs_filled)

    return intensity_profile, z_positions


for COND in CONDS:
    for REP in repeats:
        
        path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/input/{}/{}/".format(COND,REP)
        path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/{}/{}/".format(COND,REP)

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

            correction_function, intensity_profile, z_positions = get_intenity_profile(CT_A12, ch_DAPI)
            intensity_profile_p53_F3, z_positions_p53 = get_intenity_profile_spill(CT_F3, ch_F3, ch_p53, p53_F3_s_global, p53_F3_0z, cell_number_threshold=5)
            intensity_profile_p53_A12, z_positions_p53 = get_intenity_profile_spill(CT_A12, ch_A12, ch_p53, p53_A12_s_global, p53_A12_0z, cell_number_threshold=5)
            _, intensity_profile_A12, z_positions = get_intenity_profile(CT_A12, ch_A12)
            _, intensity_profile_F3, z_positions = get_intenity_profile(CT_A12, ch_F3)

            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(intensity_profile, label="intensity prof. DAPI", c='k')
            ax.set_ylabel("DAPI [a.u.]")
            ax_p53 = ax.twinx()
            ax_p53.plot(z_positions, intensity_profile_p53_F3, label="intensity prof. p53-F3", c="green")
            ax_p53.plot(z_positions, intensity_profile_p53_A12, label="intensity prof. p53-A12", c="magenta")
            plt.tight_layout()
            plt.savefig(path_z_drift_figures_p53+embcode+".png")

            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(z_positions, intensity_profile, label="intensity prof. DAPI", c='k')
            ax.set_ylabel("DAPI [a.u.]")
            ax_p53 = ax.twinx()
            ax_p53.plot(z_positions, intensity_profile_F3, label="intensity prof. F3", c="green")            
            plt.tight_layout()
            plt.savefig(path_z_drift_figures_F3+embcode+".png")

            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(z_positions, intensity_profile, label="intensity prof. DAPI", c='k')
            ax.set_ylabel("DAPI [a.u.]")
            ax_p53 = ax.twinx()
            ax_p53.plot(z_positions, intensity_profile_A12, label="intensity prof. A12", c="magenta")
            plt.tight_layout()
            plt.savefig(path_z_drift_figures_A12+embcode+".png")
            