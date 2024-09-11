from qlivecell import tif_reader_5D, cellSegTrack, extract_fluoro, get_file_names, get_file_name, get_intenity_profile, construct_RGB, EmbryoSegmentation

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

import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import SimpleITK as sitk

sigmas_avg = [2.0, 2.5, 3.0, 4.0, 5.0, 7.5]
sigma=2
weight_th=2

path_datas = ["/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/merge/2024_07_22_DMSOmerged_Casp3/", "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/merge/2024_08_13_DMSOmerged_Casp3_n2/"]

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/figures/figures_joshi_poster/merged/single_gastruloids_segmented/combined/"

channel_names = ["A12", "Casp3", "F3", "DAPI"]

ch_F3 = channel_names.index("F3")
ch_A12 = channel_names.index("A12")
ch_Casp3 = channel_names.index("Casp3")
ch_DAPI = channel_names.index("DAPI")


files_WT_48_6 = []
files_KO_48_6 = []
files_WT_48_24 = []
files_KO_48_24 = []

files_WT_72_6 = []
files_KO_72_6 = []
files_WT_72_24 = []
files_KO_72_24 = []

for path_data in path_datas:
    files = get_file_names(path_data)
    files = [file for file in files if ".tif" in file]

    _files_WT_48_6 = [path_data+file for file in files if all(map(file.__contains__, ["48+06", "F3+WT"]))]
    files_WT_48_6 = [*files_WT_48_6, *_files_WT_48_6]
    _files_KO_48_6 = [path_data+file for file in files if all(map(file.__contains__, ["48+06", "F3+KO"]))]
    files_KO_48_6 = [*files_KO_48_6, *_files_KO_48_6]
    _files_WT_48_24 = [path_data+file for file in files if all(map(file.__contains__, ["48+24", "F3+WT"]))]
    files_WT_48_24 = [*files_WT_48_24, *_files_WT_48_24]
    _files_KO_48_24 = [path_data+file for file in files if all(map(file.__contains__, ["48+24", "F3+KO"]))]
    files_KO_48_24 = [*files_KO_48_24, *_files_KO_48_24]

    _files_WT_72_6 = [path_data+file for file in files if all(map(file.__contains__, ["72+06", "F3+WT"]))]
    files_WT_72_6 = [*files_WT_72_6, *_files_WT_72_6]
    _files_KO_72_6 = [path_data+file for file in files if all(map(file.__contains__, ["72+06", "F3+KO"]))]
    files_KO_72_6 = [*files_KO_72_6, *_files_KO_72_6]
    _files_WT_72_24 = [path_data+file for file in files if all(map(file.__contains__, ["72+24", "F3+WT"]))]
    files_WT_72_24 = [*files_WT_72_24, *_files_WT_72_24]
    _files_KO_72_24 = [path_data+file for file in files if all(map(file.__contains__, ["72+24", "F3+KO"]))]
    files_KO_72_24 = [*files_KO_72_24, *_files_KO_72_24]

conditions = [files_WT_48_6, files_KO_48_6, files_WT_48_24, files_KO_48_24, files_WT_72_6, files_KO_72_6, files_WT_72_24, files_KO_72_24]
conditions_names = ["48+06_F3+WT", "48+06_F3+KO", "48+24_F3+WT", "48+24_F3+KO", "72+06_F3+WT", "72+06_F3+KO", "72+24_F3+WT", "72+24_F3+KO"]

for c, files_current in enumerate(conditions):
    cname = conditions_names[c]
    try: 
        files = get_file_names(path_figures+cname)
    except: 
        import os
        os.mkdir(path_figures+cname)
     
    print()
    print(cname)
    F3_PROJ = []
    A12_PROJ = []
    CASP3_PROJ = []
    borders = []
    for path_data_file in files_current:
        if "AbOnly" in path_data_file: continue
        if "Example" in path_data_file: continue
                    
        hyperstack, metadata = tif_reader_5D(path_data_file)
        hyperstack = hyperstack.astype("float32")
        ES = EmbryoSegmentation(
            hyperstack[:,:,ch_DAPI].astype("float32"),
            ksize=5,
            ksigma=20,
            # binths=[8, 3],
            apply_biths_to_zrange_only=False,
            checkerboard_size=10,
            num_inter=100,
            smoothing=20,
            mp_threads=13,
        )
        ES(hyperstack[:,:,ch_DAPI].astype("float32"))

        hyperstack, metadata = tif_reader_5D(path_data_file)
        hyperstack = hyperstack.astype("float32")
        for ch in range(hyperstack.shape[2]):
            stack = hyperstack[0,:,ch]
            for z in range(stack.shape[0]):
                st = stack[z]
                mask = ES.Backmask[0][z]
                if z < 7:
                    if len(mask) < len(ES.Backmask[0][z+1]): 
                        for zz in range(0, z+1):
                            st[:]=np.nan
                            hyperstack[0, zz, ch] = st
                        continue
                if z > 19:
                    if len(mask) < len(ES.Backmask[0][z-1]): 
                        for zz in range(z, stack.shape[0]):
                            st[:]=np.nan
                            hyperstack[0, zz, ch] = st
                        break
                if len(mask) < 50:
                    st[:]=np.nan
                    hyperstack[0, z, ch] = st
                else:
                    st[mask[:,1], mask[:,0]]=np.nan
                    hyperstack[0, z, ch] = st
        
        stack_f3_mean = np.nanmean(hyperstack[0,:,ch_F3], axis=0)
        stack_a12_mean = np.nanmean(hyperstack[0,:,ch_A12], axis=0)
        stack_casp3_mean = np.nanmean(hyperstack[0,:,ch_Casp3], axis=0)

        stack_f3_max = np.nanmax(hyperstack[0,:,ch_F3], axis=0)
        stack_a12_max = np.nanmax(hyperstack[0,:,ch_A12], axis=0)
        stack_casp3_max = np.nanmax(hyperstack[0,:,ch_Casp3], axis=0)

        # Desired final size
        desired_shape = (700, 700)

        # Calculate padding for each dimension
        padding = (
            ((desired_shape[0] - stack_f3_mean.shape[0]) // 2, (desired_shape[0] - stack_f3_mean.shape[0]) - (desired_shape[0] - stack_f3_mean.shape[0]) // 2),  # Padding for rows
            ((desired_shape[1] - stack_f3_mean.shape[1]) // 2, (desired_shape[1] - stack_f3_mean.shape[1]) - (desired_shape[1] - stack_f3_mean.shape[1]) // 2)   # Padding for columns
        )

        # Pad array symmetrically with zeros
        stack_f3_mean = np.pad(stack_f3_mean, padding, mode='constant', constant_values=0)
        stack_a12_mean = np.pad(stack_a12_mean, padding, mode='constant', constant_values=0)
        stack_casp3_mean = np.pad(stack_casp3_mean, padding, mode='constant', constant_values=0)

        stack_f3_max = np.pad(stack_f3_max, padding, mode='constant', constant_values=0)
        stack_a12_max = np.pad(stack_a12_max, padding, mode='constant', constant_values=0)
        stack_casp3_max = np.pad(stack_casp3_max, padding, mode='constant', constant_values=0)

        pointsx = [i for i in range(stack_f3_max.shape[0]) for j in range(stack_f3_max.shape[1]) if not np.isnan(stack_f3_max[i,j])]
        pointsy = [j for i in range(stack_f3_max.shape[0]) for j in range(stack_f3_max.shape[1]) if not np.isnan(stack_f3_max[i,j])]
        weights = [stack_f3_max[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
        weights_f3 = [w if w>weight_th else 0 for w in weights]
        xs_f3 = np.average(np.array(pointsx), weights=weights_f3)
        ys_f3 = np.average(np.array(pointsy), weights=weights_f3)

        weights = [stack_a12_max[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
        weights_a12 = [w if w>weight_th else 0 for w in weights ]
        xs_a12 = np.average(np.array(pointsx), weights=weights_a12)
        ys_a12 = np.average(np.array(pointsy), weights=weights_a12)

        stack = construct_RGB(R=stack_a12_mean/np.nanmax(stack_a12_mean), G=stack_f3_mean/np.nanmax(stack_f3_mean), B=stack_a12_mean/np.nanmax(stack_a12_mean))

        import SimpleITK as sitk
        import numpy as np

        if ys_a12 > ys_f3:
            x1 = xs_a12
            x2 = xs_f3
            y1 = ys_a12
            y2 = ys_f3
        else: 
            x2 = xs_a12
            x1 = xs_f3
            y2 = ys_a12
            y1 = ys_f3
            
        angle = np.arctan2(x2 - x1, y2 - y1)
        image = sitk.GetImageFromArray(stack_f3_mean)

        # Define the center of rotation (usually the center of the image)
        image_center = [y1, x1]

        # Create the rotation transform
        transform = sitk.Euler2DTransform()
        transform.SetCenter(image_center)

        transform.SetAngle(angle+np.pi)

        # Resample the image with the transformation
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)

        # Apply the rotation
        rotated_image = resampler.Execute(image)
        stack_corrected_f3_mean = sitk.GetArrayFromImage(rotated_image)

        image = sitk.GetImageFromArray(stack_f3_max)
        rotated_image = resampler.Execute(image)
        stack_corrected_f3_max = sitk.GetArrayFromImage(rotated_image)

        image = sitk.GetImageFromArray(stack_a12_mean)
        rotated_image = resampler.Execute(image)
        stack_corrected_a12_mean = sitk.GetArrayFromImage(rotated_image)

        image = sitk.GetImageFromArray(stack_a12_max)
        rotated_image = resampler.Execute(image)
        stack_corrected_a12_max = sitk.GetArrayFromImage(rotated_image)

        pointsx = [i for i in range(stack_corrected_f3_max.shape[0]) for j in range(stack_corrected_f3_max.shape[1]) if not np.isnan(stack_corrected_f3_max[i,j])]
        pointsy = [j for i in range(stack_corrected_f3_max.shape[0]) for j in range(stack_corrected_f3_max.shape[1]) if not np.isnan(stack_corrected_f3_max[i,j])]
        weights = [stack_corrected_f3_max[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
        weights_f3 = [w if w>weight_th else 0 for w in weights]
        xs_f3_corr = np.average(np.array(pointsx), weights=weights_f3)
        ys_f3_corr = np.average(np.array(pointsy), weights=weights_f3)

        weights = [stack_corrected_a12_max[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
        weights_a12 = [w if w>weight_th else 0 for w in weights ]
        xs_a12_corr = np.average(np.array(pointsx), weights=weights_a12)
        ys_a12_corr = np.average(np.array(pointsy), weights=weights_a12)

        image = sitk.GetImageFromArray(stack_casp3_mean)
        rotated_image = resampler.Execute(image)
        stack_corrected_casp3_mean = sitk.GetArrayFromImage(rotated_image)

        image = sitk.GetImageFromArray(stack_casp3_max)
        rotated_image = resampler.Execute(image)
        stack_corrected_casp3_max = sitk.GetArrayFromImage(rotated_image)

        if ys_a12_corr > ys_f3_corr:
            angle = np.pi
            image = sitk.GetImageFromArray(stack_f3_mean)
            # Define the center of rotation (usually the center of the image)
            image_center = [image.GetWidth() // 2, image.GetHeight() // 2]
            
            # Create the rotation transform
            transform = sitk.Euler2DTransform()
            transform.SetCenter(image_center)

            transform.SetAngle(angle)

            # Resample the image with the transformation
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(transform)

            # Apply the rotation
            image = sitk.GetImageFromArray(stack_corrected_f3_mean)
            rotated_image = resampler.Execute(image)
            stack_corrected_f3_mean = sitk.GetArrayFromImage(rotated_image)

            image = sitk.GetImageFromArray(stack_corrected_f3_max)
            rotated_image = resampler.Execute(image)
            stack_corrected_f3_max = sitk.GetArrayFromImage(rotated_image)
                
            image = sitk.GetImageFromArray(stack_corrected_a12_mean)
            rotated_image = resampler.Execute(image)
            stack_corrected_a12_mean = sitk.GetArrayFromImage(rotated_image)

            image = sitk.GetImageFromArray(stack_corrected_a12_max)
            rotated_image = resampler.Execute(image)
            stack_corrected_a12_max = sitk.GetArrayFromImage(rotated_image)

            image = sitk.GetImageFromArray(stack_corrected_casp3_mean)
            rotated_image = resampler.Execute(image)
            stack_corrected_casp3_mean = sitk.GetArrayFromImage(rotated_image)

            image = sitk.GetImageFromArray(stack_corrected_casp3_max)
            rotated_image = resampler.Execute(image)
            stack_corrected_casp3_max = sitk.GetArrayFromImage(rotated_image)

            pointsx = [i for i in range(stack_corrected_f3_max.shape[0]) for j in range(stack_corrected_f3_max.shape[1]) if not np.isnan(stack_corrected_f3_max[i,j])]
            pointsy = [j for i in range(stack_corrected_f3_max.shape[0]) for j in range(stack_corrected_f3_max.shape[1]) if not np.isnan(stack_corrected_f3_max[i,j])]
            weights = [stack_corrected_f3_max[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
            weights_f3 = [w if w>weight_th else 0 for w in weights]
            xs_f3_corr = np.average(np.array(pointsx), weights=weights_f3)
            ys_f3_corr = np.average(np.array(pointsy), weights=weights_f3)

            weights = [stack_corrected_a12_max[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
            weights_a12 = [w if w>weight_th else 0 for w in weights ]
            xs_a12_corr = np.average(np.array(pointsx), weights=weights_a12)
            ys_a12_corr = np.average(np.array(pointsy), weights=weights_a12)

        stack_corrected_casp3_mean[np.isnan(stack_corrected_casp3_mean)] = 0
        stack_corrected = construct_RGB(R=stack_corrected_a12_mean/np.nanmax(stack_corrected_a12_mean), G=stack_corrected_f3_mean/np.nanmax(stack_corrected_f3_mean), B=stack_corrected_a12_mean/np.nanmax(stack_corrected_a12_mean))
        stack_corrected_casp = construct_RGB(R=stack_corrected_casp3_mean/(np.nanmax(stack_corrected_casp3_mean)*0.7),G=stack_corrected_casp3_mean/(np.nanmax(stack_corrected_casp3_mean)*0.7))

        f3_proj_mean = np.nanmean(stack_corrected_f3_mean, axis=0)
        f3_proj_mean[np.where(f3_proj_mean<0.01)] = np.nan
        f3_proj_max = np.nanmean(stack_corrected_f3_max, axis=0)

        a12_proj_mean = np.nanmean(stack_corrected_a12_mean, axis=0)
        a12_proj_mean[np.where(a12_proj_mean<0.01)] = np.nan
        a12_proj_max = np.nanmean(stack_corrected_a12_max, axis=0)

        casp3_proj_mean = np.nanmean(stack_corrected_casp3_mean, axis=0)
        casp3_proj_mean[np.where(casp3_proj_mean<0.01)] = np.nan
        casp3_proj_max = np.nanmean(stack_corrected_casp3_max, axis=0)

        comb = gaussian_filter1d(f3_proj_max*a12_proj_max, sigma=sigma)

        F3_PROJ.append(f3_proj_mean)
        A12_PROJ.append(a12_proj_mean)
        CASP3_PROJ.append(casp3_proj_mean)
        borders.append(np.argmax(comb))

    F3_SIGNALS = np.ones((len(borders), len(F3_PROJ[0])*2 + 1))*np.nan
    A12_SIGNALS = np.ones_like(F3_SIGNALS)*np.nan
    CASP3_SIGNALS = np.ones_like(F3_SIGNALS)*np.nan

    for b in range(len(borders)):
        center = len(F3_PROJ[0])
        border = borders[b]
        F3_SIGNALS[b, center-border:center] = F3_PROJ[b][:border]
        F3_SIGNALS[b, center:2*center-border] = F3_PROJ[b][border:]

        A12_SIGNALS[b, center-border:center] = A12_PROJ[b][:border]
        A12_SIGNALS[b, center:2*center-border] = A12_PROJ[b][border:]

        CASP3_SIGNALS[b, center-border:center] = CASP3_PROJ[b][:border]
        CASP3_SIGNALS[b, center:2*center-border] = CASP3_PROJ[b][border:]

    center = len(F3_PROJ[0])

    f3_signal = np.nanmean(F3_SIGNALS, axis=0)
    f3_signal_std = np.nanstd(F3_SIGNALS, axis=0)

    a12_signal = np.nanmean(A12_SIGNALS, axis=0)
    a12_signal_std = np.nanstd(A12_SIGNALS, axis=0)

    casp3_signal = np.nanmean(CASP3_SIGNALS, axis=0)
    casp3_signal_std = np.nanstd(CASP3_SIGNALS, axis=0)

    x = [i for i in range(len(f3_signal))]

    fig, ax = plt.subplots(len(F3_SIGNALS)+1,1, figsize=(9,5*(len(F3_SIGNALS)+1)), sharex=True)

    ax_t = []
    for p in range(len(ax)-1):
        ax_t.append(ax[p].twinx())
        ax[p].set_yticks([])
        ax_t[p].plot(x, CASP3_SIGNALS[p], color=[238/255, 210/255, 0.0], lw=4)
        ax[p].axvline(center, c="black", lw=2, ls='--')
        
    axc = ax[-1].twinx()

    ax[-1].axvline(center, c="black", lw=2, ls='--')

    ax[-1].plot(x, f3_signal, color="green", lw=4)

    ax[-1].fill_between(x, f3_signal - f3_signal_std, f3_signal + f3_signal_std, color="green", alpha=0.2)

    ax[-1].plot(x, a12_signal, color="magenta", lw=4)
    ax[-1].fill_between(x, a12_signal - a12_signal_std, a12_signal + a12_signal_std, color="magenta", alpha=0.2)

    axc.plot(x, casp3_signal, color=[238/255, 210/255, 0.0], lw=4)
    axc.fill_between(x, casp3_signal - casp3_signal_std, casp3_signal + casp3_signal_std, color=[238/255, 210/255, 0.0], alpha=0.2)
    ax[-1].set_xlim(x[0],x[1])
    ax[-1].set_title(cname)

    pix_res = 100/metadata["XYresolution"]

    xticks_left = [i for i in np.arange(center, 0.0, -pix_res) if i >=0]
    xticks_right = [i for i in np.arange(center, len(x), pix_res)]
    xticks = np.unique([*xticks_left, *xticks_right])

    xticks_labels = np.rint((xticks - center)*metadata["XYresolution"]).astype("int32")
    ax[-1].set_xticks(xticks)
    ax[-1].set_xticklabels(xticks_labels)

    ax[-1].set_xlabel(r"distance to merge point ($\mu$m)")
    ax[-1].set_xlim(center-300, center+300)
    axc.set_xlim(center-300, center+300)
    ax[-1].set_yticks([])

    plt.tight_layout()
    plt.savefig(path_figures+cname+"/"+"general_with_examples.svg")
    plt.savefig(path_figures+cname+"/"+"general_with_examples.png")
    # plt.show()

    fig, ax = plt.subplots(figsize=(9,6))
    axc = ax.twinx()

    ax.axvline(center, c="black", lw=2, ls='--')

    ax.plot(x, f3_signal, color="green", lw=4)

    ax.fill_between(x, f3_signal - f3_signal_std, f3_signal + f3_signal_std, color="green", alpha=0.2)

    ax.plot(x, a12_signal, color="magenta", lw=4)
    ax.fill_between(x, a12_signal - a12_signal_std, a12_signal + a12_signal_std, color="magenta", alpha=0.2)

    axc.plot(x, casp3_signal, color=[238/255, 210/255, 0.0], lw=4)
    axc.fill_between(x, casp3_signal - casp3_signal_std, casp3_signal + casp3_signal_std, color=[238/255, 210/255, 0.0], alpha=0.2)
    ax.set_xlim(x[0],x[1])
    ax.set_title(cname)

    pix_res = 100/metadata["XYresolution"]

    xticks_left = [i for i in np.arange(center, 0.0, -pix_res) if i >=0]
    xticks_right = [i for i in np.arange(center, len(x), pix_res)]
    xticks = np.unique([*xticks_left, *xticks_right])

    xticks_labels = np.rint((xticks - center)*metadata["XYresolution"]).astype("int32")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels)

    ax.set_xlabel(r"distance to merge point ($\mu$m)")
    ax.set_xlim(center-300, center+300)
    axc.set_xlim(center-300, center+300)
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path_figures+cname+"/"+"general.svg")
    plt.savefig(path_figures+cname+"/"+"general.png")


    for sigma_avg in sigmas_avg:
        F3_SIGNALS = np.ones((len(borders), len(F3_PROJ[0])*2 + 1))*np.nan
        A12_SIGNALS = np.ones_like(F3_SIGNALS)*np.nan
        CASP3_SIGNALS = np.ones_like(F3_SIGNALS)*np.nan

        for b in range(len(borders)):
            center = len(F3_PROJ[0])
            border = borders[b]
            F3_SIGNALS[b, center-border:center] = F3_PROJ[b][:border]
            F3_SIGNALS[b, center:2*center-border] = F3_PROJ[b][border:]

            A12_SIGNALS[b, center-border:center] = A12_PROJ[b][:border]
            A12_SIGNALS[b, center:2*center-border] = A12_PROJ[b][border:]

            CASP3_SIGNALS[b, center-border:center] = CASP3_PROJ[b][:border]
            CASP3_SIGNALS[b, center:2*center-border] = CASP3_PROJ[b][border:]

            F3_SIGNALS[b] = gaussian_filter1d(F3_SIGNALS[b], sigma=sigma_avg)
            A12_SIGNALS[b] = gaussian_filter1d(A12_SIGNALS[b], sigma=sigma_avg)
            CASP3_SIGNALS[b] = gaussian_filter1d(CASP3_SIGNALS[b], sigma=sigma_avg)

        center = len(F3_PROJ[0])

        f3_signal = np.nanmean(F3_SIGNALS, axis=0)
        f3_signal_std = np.nanstd(F3_SIGNALS, axis=0)

        a12_signal = np.nanmean(A12_SIGNALS, axis=0)
        a12_signal_std = np.nanstd(A12_SIGNALS, axis=0)

        casp3_signal = np.nanmean(CASP3_SIGNALS, axis=0)
        casp3_signal_std = np.nanstd(CASP3_SIGNALS, axis=0)

        x = [i for i in range(len(f3_signal))]

        fig, ax = plt.subplots(len(F3_SIGNALS)+1,1, figsize=(9,5*(len(F3_SIGNALS)+1)), sharex=True)

        ax_t = []
        for p in range(len(ax)-1):
            ax_t.append(ax[p].twinx())
            ax[p].set_yticks([])
            ax_t[p].plot(x, CASP3_SIGNALS[p], color=[238/255, 210/255, 0.0], lw=4)
            ax[p].axvline(center, c="black", lw=2, ls='--')
            
        axc = ax[-1].twinx()

        ax[-1].axvline(center, c="black", lw=2, ls='--')

        ax[-1].plot(x, f3_signal, color="green", lw=4)

        ax[-1].fill_between(x, f3_signal - f3_signal_std, f3_signal + f3_signal_std, color="green", alpha=0.2)

        ax[-1].plot(x, a12_signal, color="magenta", lw=4)
        ax[-1].fill_between(x, a12_signal - a12_signal_std, a12_signal + a12_signal_std, color="magenta", alpha=0.2)

        axc.plot(x, casp3_signal, color=[238/255, 210/255, 0.0], lw=4)
        axc.fill_between(x, casp3_signal - casp3_signal_std, casp3_signal + casp3_signal_std, color=[238/255, 210/255, 0.0], alpha=0.2)
        ax[-1].set_xlim(x[0],x[1])
        ax[-1].set_title(cname)

        pix_res = 100/metadata["XYresolution"]

        xticks_left = [i for i in np.arange(center, 0.0, -pix_res) if i >=0]
        xticks_right = [i for i in np.arange(center, len(x), pix_res)]
        xticks = np.unique([*xticks_left, *xticks_right])

        xticks_labels = np.rint((xticks - center)*metadata["XYresolution"]).astype("int32")
        ax[-1].set_xticks(xticks)
        ax[-1].set_xticklabels(xticks_labels)

        ax[-1].set_xlabel(r"distance to merge point ($\mu$m)")
        ax[-1].set_xlim(center-300, center+300)
        axc.set_xlim(center-300, center+300)
        ax[-1].set_yticks([])

        plt.tight_layout()
        plt.savefig(path_figures+cname+"/"+"general_with_examples_sigma{}.svg".format(sigma_avg))
        plt.savefig(path_figures+cname+"/"+"general_with_examples_sigma{}.png".format(sigma_avg))
        # plt.show()

        fig, ax = plt.subplots(figsize=(9,6))
        axc = ax.twinx()

        ax.axvline(center, c="black", lw=2, ls='--')

        ax.plot(x, f3_signal, color="green", lw=4)

        ax.fill_between(x, f3_signal - f3_signal_std, f3_signal + f3_signal_std, color="green", alpha=0.2)

        ax.plot(x, a12_signal, color="magenta", lw=4)
        ax.fill_between(x, a12_signal - a12_signal_std, a12_signal + a12_signal_std, color="magenta", alpha=0.2)

        axc.plot(x, casp3_signal, color=[238/255, 210/255, 0.0], lw=4)
        axc.fill_between(x, casp3_signal - casp3_signal_std, casp3_signal + casp3_signal_std, color=[238/255, 210/255, 0.0], alpha=0.2)
        ax.set_xlim(x[0],x[1])
        ax.set_title(cname)

        pix_res = 100/metadata["XYresolution"]

        xticks_left = [i for i in np.arange(center, 0.0, -pix_res) if i >=0]
        xticks_right = [i for i in np.arange(center, len(x), pix_res)]
        xticks = np.unique([*xticks_left, *xticks_right])

        xticks_labels = np.rint((xticks - center)*metadata["XYresolution"]).astype("int32")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_labels)

        ax.set_xlabel(r"distance to merge point ($\mu$m)")
        ax.set_xlim(center-300, center+300)
        axc.set_xlim(center-300, center+300)
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(path_figures+cname+"/"+"general_sigma{}.svg".format(sigma_avg))
        plt.savefig(path_figures+cname+"/"+"general_sigma{}.png".format(sigma_avg))

    # plt.show()