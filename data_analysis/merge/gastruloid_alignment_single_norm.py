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

weight_th=2
path_data = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/merge/2024_07_22_DMSOmerged_Casp3/"
path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/figures/figures_joshi_poster/merged/single_gastruloids_norm/"

channel_names = ["A12", "Casp3", "F3", "DAPI"]
ch_F3 = channel_names.index("F3")
ch_A12 = channel_names.index("A12")
ch_Casp3 = channel_names.index("Casp3")
ch_DAPI = channel_names.index("DAPI")

files = get_file_names(path_data)
files = [file for file in files if ".tif" in file]

files_WT_48_6 = [file for file in files if all(map(file.__contains__, ["48+06", "F3+WT"]))]
files_KO_48_6 = [file for file in files if all(map(file.__contains__, ["48+06", "F3+KO"]))]
files_WT_48_24 = [file for file in files if all(map(file.__contains__, ["48+24", "F3+WT"]))]
files_KO_48_24 = [file for file in files if all(map(file.__contains__, ["48+24", "F3+KO"]))]

files_WT_72_6 = [file for file in files if all(map(file.__contains__, ["72+06", "F3+WT"]))]
files_KO_72_6 = [file for file in files if all(map(file.__contains__, ["72+06", "F3+KO"]))]
files_WT_72_24 = [file for file in files if all(map(file.__contains__, ["72+24", "F3+WT"]))]
files_KO_72_24 = [file for file in files if all(map(file.__contains__, ["72+24", "F3+KO"]))]

files_current = files_WT_48_6
for file in files_current:
    print(file)
    # file=files_current[1]
    path_data_file = path_data+file
                
    hyperstack, metadata = tif_reader_5D(path_data_file)

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

    fig, ax = plt.subplots(3, 1, figsize=(6,15), sharex=True)

    f3_proj_mean = np.nanmean(stack_corrected_f3_mean, axis=0)
    f3_proj_max = np.nanmean(stack_corrected_f3_max, axis=0)

    a12_proj_mean = np.nanmean(stack_corrected_a12_mean, axis=0)
    a12_proj_max = np.nanmean(stack_corrected_a12_max, axis=0)

    casp3_proj_mean = np.nanmean(stack_corrected_casp3_mean, axis=0) / (f3_proj_mean + a12_proj_mean)
    casp3_proj_max = np.nanmean(stack_corrected_casp3_max, axis=0) / (f3_proj_max + a12_proj_max)

    ax[0].imshow(stack_corrected)
    ax[0].scatter([ys_f3_corr],[xs_f3_corr], color="white", s=50)
    ax[0].scatter([ys_a12_corr],[xs_a12_corr], color="white", s=50)

    # ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    ax[1].imshow(stack_corrected_casp3_mean/np.max(stack_corrected_casp3_mean), cmap="viridis", vmax=0.6)
    ax[1].scatter([ys_f3_corr],[xs_f3_corr], color="white", s=50)
    ax[1].scatter([ys_a12_corr],[xs_a12_corr], color="white", s=50)

    # ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].plot(f3_proj_mean, c="green", lw=4)
    ax[2].plot(a12_proj_mean, c="magenta", lw=4)
    ax2 = ax[2].twinx()
    ax2.plot(casp3_proj_mean, c=[238/255, 210/255, 0.0], lw=4)
    comb = gaussian_filter1d(f3_proj_max*a12_proj_max, sigma=2)
    ax[2].plot(f3_proj_mean + a12_proj_mean, c="grey", lw=4)

    ax[0].axvline(np.nanargmax(comb), c="white", lw=4)
    ax[1].axvline(np.nanargmax(comb), c="white", lw=4)
    ax[2].axvline(np.nanargmax(comb), c="black", lw=4)
    ax[2].set_xlim(0,len(casp3_proj_mean))

    plt.tight_layout()
    plt.savefig(path_figures+file[:-4]+".png")
    # plt.show()

