# =========================
# Spillover calibration (per z baseline) and per-pixel correction
# =========================

# ---- USER CONFIG ----
BLEED_FROM = "F3"   # <-- set to "F3" or "A12": the channel that bleeds into C ("p53")
C_CHANNEL  = "p53"  # your readout channel (C)

# Limit pixels per z-plane when fitting (to keep memory/time bounded).
# Set to None to use ALL in-cell pixels.
SAMPLE_PER_Z = 200_000

files_to_exclude = [
    "n3_F3(150)+KO(25)_72h_emiRFP-2ndaryCtrl488-mCh-DAPI_(40xSil)_Stack1.tif",
    # "n3_F3(150)+KO(25)_72h_emiRFP-2ndaryCtrl488-mCh-DAPI_(40xSil)_Stack2.tif"
]

# ---- Imports & plotting style (your settings) ----
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from qlivecell import get_file_name, cellSegTrack, check_or_create_dir, get_file_names
from stardist.models import StarDist2D

plt.rcParams.update({"text.usetex": True})
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath'
mpl.rc('font', size=18)
mpl.rc('axes', labelsize=18)
mpl.rc('xtick', labelsize=18)
mpl.rc('ytick', labelsize=18)
mpl.rc('legend', fontsize=18)

# ---- Channels (your order) ----
channel_names = ["A12", "p53", "F3", "DAPI"]
ch_B = channel_names.index(BLEED_FROM)
ch_C = channel_names.index(C_CHANNEL)

# ---- StarDist model (your choice) ----
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# =========================
# Helpers
# =========================

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

def sample_indices(n, k):
    if (k is None) or (n <= k):
        return slice(None)
    return np.random.choice(n, size=k, replace=False)

def update_normal_eq_sums(x, y, sums):
    """
    Update sufficient statistics for OLS fit y = b0 + s*x
    sums is a dict with keys: N, sumx, sumy, sumxx, sumxy
    """
    if x.size == 0:
        return
    sums["N"]     += x.size
    sums["sumx"]  += float(np.sum(x))
    sums["sumy"]  += float(np.sum(y))
    sums["sumxx"] += float(np.sum(x * x))
    sums["sumxy"] += float(np.sum(x * y))

def solve_b0_s_from_sums(sums):
    N, sumx, sumy, sumxx, sumxy = sums["N"], sums["sumx"], sums["sumy"], sums["sumxx"], sums["sumxy"]
    denom = (N * sumxx - sumx**2)
    if denom == 0 or N == 0:
        return 0.0, 0.0
    s  = (N * sumxy - sumx * sumy) / denom
    b0 = (sumy - s * sumx) / N
    return float(b0), float(s)

def estimate_b0z_for_file(CT_ref, Mz_list, ch_B, ch_C, s_global):
    """Compute per-z baseline b0(z) = median( C - s_global * B ) over in-cell pixels."""
    Z = CT_ref.hyperstack.shape[1]
    b0z = np.full(Z, np.nan, dtype=np.float64)
    for z in range(Z):
        Mz = Mz_list[z]
        if not np.any(Mz):
            continue
        Bz = CT_ref.hyperstack[0, z, ch_B, :, :].astype(np.float64)
        Cz = CT_ref.hyperstack[0, z, ch_C, :, :].astype(np.float64)
        x  = Bz[Mz].ravel()
        y  = Cz[Mz].ravel()
        resid = y - s_global * x
        b0z[z] = np.median(resid)
    # Fill any empty planes with the median of available planes (fallback to 0)
    if np.any(np.isnan(b0z)):
        if np.any(~np.isnan(b0z)):
            fill = np.nanmedian(b0z)
        else:
            fill = 0.0
        b0z[np.isnan(b0z)] = fill
    return b0z

def correct_cell_pixels(CT_ref, mask, z, ch_B, ch_C, s, b0z):
    """Return per-pixel corrected C for one cell at plane z."""
    yy = mask[:, 1].astype(np.intp)
    xx = mask[:, 0].astype(np.intp)
    C_vals = CT_ref.hyperstack[0, z, ch_C, :, :][yy, xx].astype(np.float32)
    B_vals = CT_ref.hyperstack[0, z, ch_B, :, :][yy, xx].astype(np.float32)
    return C_vals - float(b0z[z]) - float(s) * B_vals

# =========================
# Pass 1: Fit global s (B → C) on secondary-only, using in-cell pixels (union masks)
#         Also get a global intercept (rarely used), we’ll prefer b0(z) per file.
# =========================

# IMPORTANT: use the same acquisition settings in "secondaryonly" as in your experiments.
# We'll stream sufficient statistics to avoid loading everything in memory.

path_data_dir = f"/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/input/KO/n3_Ab/"
path_save_dir = f"/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/KO/n3_Ab/"
check_or_create_dir(path_save_dir)

files = get_file_names(path_data_dir)
# Prepare segmentation/plot args (as you had)
segmentation_args = {'method': 'stardist2D', 'model': model, 'blur': [2, 1], 'min_outline_length': 100}
concatenation3D_args = {'do_3Dconcatenation': False}
error_correction_args = {'backup_steps': 10, 'line_builder_mode': 'points'}

# Accumulate OLS sums across all files/z-planes
sums = {"N": 0, "sumx": 0.0, "sumy": 0.0, "sumxx": 0.0, "sumxy": 0.0}

for f, file in enumerate(files):
    if file in files_to_exclude:
        continue
    path_data = os.path.join(path_data_dir, file)
    file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
    path_save = os.path.join(path_save_dir, embcode)
    check_or_create_dir(path_save)

    # --- Build CT for F3 (for masks) ---
    ch = channel_names.index(BLEED_FROM)
    batch_args = {'name_format': "ch" + str(ch) + "_{}", 'extension': ".tif"}
    plot_args  = {'plot_layout': (1, 1), 'plot_overlap': 1, 'masks_cmap': 'tab10',
                    'plot_stack_dims': (256, 256), 'plot_centers': [False, False],
                    'channels': [ch], 'min_outline_length': 75}
    chans = [ch] + [i for i in range(len(channel_names)) if i != ch]
    CT = cellSegTrack(path_data, path_save,
                            segmentation_args=segmentation_args,
                            concatenation3D_args=concatenation3D_args,
                            error_correction_args=error_correction_args,
                            plot_args=plot_args, batch_args=batch_args, channels=chans)
    CT.load()

    # --- Union mask across both populations, per z ---
    Mz_list = build_union_masks([CT])

    Z = CT.hyperstack.shape[1]
    for z in range(Z):
        Mz = Mz_list[z]
        if not np.any(Mz):
            continue
        Bz = CT.hyperstack[0, z, ch_B, :, :].astype(np.float64)
        Cz = CT.hyperstack[0, z, ch_C, :, :].astype(np.float64)

        x = Bz[Mz].ravel()
        y = Cz[Mz].ravel()

        # Optional subsample for speed
        sel = sample_indices(x.size, SAMPLE_PER_Z)
        x = x[sel]; y = y[sel]

        update_normal_eq_sums(x, y, sums)

# Solve for global (session) s and an overall b0 (we'll replace b0 by per-z b0z later)
b0_global, s_global = solve_b0_s_from_sums(sums)
print(f"[Calibration] Estimated global spillover s ({BLEED_FROM} → {C_CHANNEL}): {s_global:.6g}")
print(f"[Calibration] Global intercept b0 (unused; we’ll use per-z): {b0_global:.6g}")

# =========================
# Pass 2: For each file, compute per-z b0z using s_global, and save calibration
# =========================
for f, file in enumerate(files):
    if file in files_to_exclude:
        continue
    path_data = os.path.join(path_data_dir, file)
    file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
    path_save = os.path.join(path_save_dir, embcode)
    check_or_create_dir(path_save)

    # Re-load CTs (simpler than caching; OK for typical file counts)
    ch = channel_names.index(BLEED_FROM)
    batch_args = {'name_format': "ch" + str(ch) + "_{}", 'extension': ".tif"}
    plot_args  = {'plot_layout': (1, 1), 'plot_overlap': 1, 'masks_cmap': 'tab10',
                    'plot_stack_dims': (256, 256), 'plot_centers': [False, False],
                    'channels': [ch], 'min_outline_length': 75}
    chans = [ch] + [i for i in range(len(channel_names)) if i != ch]
    CT = cellSegTrack(path_data, path_save,
                            segmentation_args=segmentation_args,
                            concatenation3D_args=concatenation3D_args,
                            error_correction_args=error_correction_args,
                            plot_args=plot_args, batch_args=batch_args, channels=chans)
    CT.load()

    Mz_list = build_union_masks([CT])

    b0z = estimate_b0z_for_file(CT, Mz_list, ch_B, ch_C, s_global)

    # Save calibration alongside the file’s outputs (so you can reuse later)
    calib_path = os.path.join(path_save, f"calibration_{BLEED_FROM}_to_{C_CHANNEL}.npz")
    np.savez(calib_path, s=s_global, b0z=b0z, bleed_from=BLEED_FROM, c_channel=C_CHANNEL)
    print(f"[Saved] {calib_path}  |  s={s_global:.6g}  median(b0z)={np.median(b0z):.6g}")

    # -------- OPTIONAL quick QC plot per file (collapsed across z) --------
    # Make a small scatter of (C - b0z[z]) vs B for a few planes, before vs after correction
    try:
        import random
        Z = CT.hyperstack.shape[1]
        planes = random.sample([z for z in range(Z) if np.any(Mz_list[z])], k=min(2, Z))
        fig, axs = plt.subplots(1, len(planes), figsize=(6*len(planes), 5))
        if len(planes) == 1:
            axs = [axs]
        for ax, z in zip(axs, planes):
            Mz = Mz_list[z]
            Bz = CT.hyperstack[0, z, ch_B, :, :].astype(np.float64)
            Cz = CT.hyperstack[0, z, ch_C, :, :].astype(np.float64)
            x = Bz[Mz].ravel()
            print(x)
            y = Cz[Mz].ravel()
            sel = sample_indices(x.size, 10_000)
            x = x[sel]; y = y[sel]
            y0 = y - b0z[z]
            ycorr = y - b0z[z] - s_global * x
            ax.scatter(x, y0, s=4, alpha=0.2, label="before")
            ax.scatter(x, ycorr, s=4, alpha=0.2, label="after")
            ax.set_title(f"{embcode}  z={z}")
            ax.set_xlabel(f"{BLEED_FROM} intensity (B)")
            ax.set_ylabel(f"{C_CHANNEL} (C) minus b0(z)")
            ax.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[QC plot skipped] {e}")

# =========================
# How to APPLY the correction when quantifying cells (per-pixel)
# =========================
# Example: for any CT (F3 or A12) and any cell (with its mask, z):
#   C_corr = C - b0z[z] - s_global * B
#
