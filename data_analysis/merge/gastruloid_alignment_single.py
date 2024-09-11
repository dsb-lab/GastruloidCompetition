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

show_yticks=True
weight_th=2
path_data = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/merge/2024_07_22_DMSOmerged_Casp3_n2/"
path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/figures/figures_joshi_poster/merged/single_gastruloids_segmented/dataset2/"

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

conditions = [files_WT_48_6, files_KO_48_6, files_WT_48_24, files_KO_48_24, files_WT_72_6, files_KO_72_6, files_WT_72_24, files_KO_72_24]
conditions_names = ["48+06_F3+WT", "48+06_F3+KO", "48+24_F3+WT", "48+24_F3+KO", "72+06_F3+WT", "72+06_F3+KO", "72+24_F3+WT", "72+24_F3+KO"]

for c, files_current in enumerate(conditions):
    cname = conditions_names[c]
    print(cname)
    try: 
        files = get_file_names(path_figures+cname)
    except: 
        import os
        os.mkdir(path_figures+cname)
        
    for file in files_current:
        path_data_file = path_data+file
                    
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
        # for z in range(hyperstack[0,:,0].shape[0]):
        #     print(z)
        #     ES.plot_segmentation(0, z)

        hyperstack, metadata = tif_reader_5D(path_data_file)
        hyperstack = hyperstack.astype("float32")
        for ch in range(hyperstack.shape[2]):
            stack = hyperstack[0,:,ch]

            for z in range(stack.shape[0]):
                st = stack[z]
                mask = ES.Backmask[0][z]
                if z < 5:
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
        # for z in range(hyperstack[0,:,0].shape[0]):
        #     plt.imshow(hyperstack[0,z,ch_DAPI])
        #     plt.show()

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

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # fig, ax = plt.subplots(figsize=(6,15))
        fig, ax1 = plt.subplots(figsize=(6,15))
        divider = make_axes_locatable(ax1)
        ax2 = divider.append_axes("bottom", size="100%", pad=0.1)
        ax3 = divider.append_axes("bottom", size="75%", pad=0.1)


        f3_proj_mean = np.nanmean(stack_corrected_f3_mean, axis=0)
        f3_proj_max = np.nanmean(stack_corrected_f3_max, axis=0)

        a12_proj_mean = np.nanmean(stack_corrected_a12_mean, axis=0)
        a12_proj_max = np.nanmean(stack_corrected_a12_max, axis=0)

        casp3_proj_mean = np.nanmean(stack_corrected_casp3_mean, axis=0)
        casp3_proj_max = np.nanmean(stack_corrected_casp3_max, axis=0)

        ax1.imshow(stack_corrected)
        # ax1.scatter([ys_f3_corr],[xs_f3_corr], color="white", s=50)
        # ax1.scatter([ys_a12_corr],[xs_a12_corr], color="white", s=50)

        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2.imshow(stack_corrected_casp3_mean/np.max(stack_corrected_casp3_mean), cmap="viridis", vmax=0.6)
        # ax2.scatter([ys_f3_corr],[xs_f3_corr], color="white", s=50)
        # ax2.scatter([ys_a12_corr],[xs_a12_corr], color="white", s=50)

        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3.plot(f3_proj_mean, c="green", lw=4)
        ax3.plot(a12_proj_mean, c="magenta", lw=4)
        ax33 = ax3.twinx()
        ax33.plot(casp3_proj_mean, c=[238/255, 210/255, 0.0], lw=4)
        comb = gaussian_filter1d(f3_proj_max*a12_proj_max, sigma=2)
        # ax3.plot(f3_proj_mean + a12_proj_mean, c="grey", lw=4)

        ax1.axvline(np.nanargmax(comb), c="white", lw=1, ls='--')
        ax2.axvline(np.nanargmax(comb), c="white", lw=1, ls='--')
        ax3.axvline(np.nanargmax(comb), c="black", lw=2, ls='--')
        ax3.set_xlim(0,len(casp3_proj_mean))

        pix_res = 100/metadata["XYresolution"]
        center_tick = np.nanargmax(comb).astype('float32')
        xticks_left = [i for i in np.arange(center_tick, 0.0, -pix_res) if i >=0]
        xticks_right = [i for i in np.arange(center_tick, stack_corrected_casp3_mean.shape[0], pix_res)]
        xticks = np.unique([*xticks_left, *xticks_right])
        xticks_labels = np.rint((xticks - center_tick)*metadata["XYresolution"]).astype("int32")
        ax3.set_xticks(xticks)
        ax3.set_xticklabels(xticks_labels)

        ax3.set_xlabel(r"distance to merge point ($\mu$m)")
        ax3.set_yticks([])
        # ax33.set_yticks([])
        ax2.set_yticks([])

        plt.tight_layout()
        plt.savefig(path_figures+cname+"/"+file[:-4]+".png")
        plt.savefig(path_figures+cname+"/"+file[:-4]+".svg")

        # plt.show()

