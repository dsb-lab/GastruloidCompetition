from qlivecell import tif_reader_5D, cellSegTrack, extract_fluoro, get_file_names, get_file_name, get_intenity_profile, construct_RGB
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
sigma=2
weight_th = 20
path_data = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/merge/2024_07_22_DMSOmerged_Casp3/"

channel_names = ["A12", "Casp3", "F3", "DAPI"]

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
files_current = files_WT_72_6

ch_F3 = channel_names.index("F3")
ch_A12 = channel_names.index("A12")
ch_Casp3 = channel_names.index("Casp3")

for c, files_current in enumerate(conditions):
    print(c)
    cname = conditions_names[c]
    F3_PROJ = []
    A12_PROJ = []
    CASP3_PROJ = []
    borders = []
    for file in files_current:
        path_data_file = path_data+file
                
        hyperstack, metadata = tif_reader_5D(path_data_file)

        stack_f3 = np.max(hyperstack[0,:,ch_F3], axis=0)
        stack_a12 = np.max(hyperstack[0,:,ch_A12], axis=0)
        stack_casp3 = np.max(hyperstack[0,:,ch_Casp3], axis=0)
        # Desired final size
        desired_shape = (700, 700)

        # Calculate padding for each dimension
        padding = (
            ((desired_shape[0] - stack_f3.shape[0]) // 2, (desired_shape[0] - stack_f3.shape[0]) - (desired_shape[0] - stack_f3.shape[0]) // 2),  # Padding for rows
            ((desired_shape[1] - stack_f3.shape[1]) // 2, (desired_shape[1] - stack_f3.shape[1]) - (desired_shape[1] - stack_f3.shape[1]) // 2)   # Padding for columns
        )

        # Pad array symmetrically with zeros
        stack_f3 = np.pad(stack_f3, padding, mode='constant', constant_values=0)
        stack_a12 = np.pad(stack_a12, padding, mode='constant', constant_values=0)
        stack_casp3 = np.pad(stack_casp3, padding, mode='constant', constant_values=0)
        
        pointsx = [i for i in range(stack_f3.shape[0]) for j in range(stack_f3.shape[1])]
        pointsy = [j for i in range(stack_f3.shape[0]) for j in range(stack_f3.shape[1])]
        weights = [stack_f3[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
        weights_f3 = [w if w>weight_th else 0 for w in weights]
        xs_f3 = np.average(np.array(pointsx), weights=weights_f3)
        ys_f3 = np.average(np.array(pointsy), weights=weights_f3)

        weights = [stack_a12[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
        weights_a12 = [w if w>weight_th else 0 for w in weights ]
        xs_a12 = np.average(np.array(pointsx), weights=weights_a12)
        ys_a12 = np.average(np.array(pointsy), weights=weights_a12)

        stack = construct_RGB(R=stack_a12, G=stack_f3)

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
        image = sitk.GetImageFromArray(stack_f3)

        # Define the center of rotation (usually the center of the image)
        image_center = [y1, x1]

        # Create the rotation transform
        transform = sitk.Euler2DTransform()
        transform.SetCenter(image_center)

        # Set the angle to rotate by (negative because we want to rotate the image to align the points)
        transform.SetAngle(angle+np.pi)

        # Resample the image with the transformation
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)

        # Apply the rotation
        rotated_image = resampler.Execute(image)

        stack_corrected_f3 = sitk.GetArrayFromImage(rotated_image)
        image = sitk.GetImageFromArray(stack_a12)
        rotated_image = resampler.Execute(image)
        stack_corrected_a12 = sitk.GetArrayFromImage(rotated_image)

        pointsx = [i for i in range(stack_corrected_f3.shape[0]) for j in range(stack_corrected_f3.shape[1])]
        pointsy = [j for i in range(stack_corrected_f3.shape[0]) for j in range(stack_corrected_f3.shape[1])]
        weights = [stack_corrected_f3[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
        weights_f3 = [w if w>weight_th else 0 for w in weights]
        xs_f3_corr = np.average(np.array(pointsx), weights=weights_f3)
        ys_f3_corr = np.average(np.array(pointsy), weights=weights_f3)

        weights = [stack_corrected_a12[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
        weights_a12 = [w if w>weight_th else 0 for w in weights ]
        xs_a12_corr = np.average(np.array(pointsx), weights=weights_a12)
        ys_a12_corr = np.average(np.array(pointsy), weights=weights_a12)

        image = sitk.GetImageFromArray(stack_casp3)
        rotated_image = resampler.Execute(image)
        stack_corrected_casp3 = sitk.GetArrayFromImage(rotated_image)

        if ys_a12_corr > ys_f3_corr:
            angle = np.pi
            image = sitk.GetImageFromArray(stack_f3)
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
            image = sitk.GetImageFromArray(stack_corrected_f3)
            rotated_image = resampler.Execute(image)
            stack_corrected_f3 = sitk.GetArrayFromImage(rotated_image)
            
            image = sitk.GetImageFromArray(stack_corrected_a12)
            rotated_image = resampler.Execute(image)
            stack_corrected_a12 = sitk.GetArrayFromImage(rotated_image)
            
            image = sitk.GetImageFromArray(stack_corrected_casp3)
            rotated_image = resampler.Execute(image)
            stack_corrected_casp3 = sitk.GetArrayFromImage(rotated_image)


            pointsx = [i for i in range(stack_corrected_f3.shape[0]) for j in range(stack_corrected_f3.shape[1])]
            pointsy = [j for i in range(stack_corrected_f3.shape[0]) for j in range(stack_corrected_f3.shape[1])]
            weights = [stack_corrected_f3[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
            weights_f3 = [w if w>weight_th else 0 for w in weights]
            xs_f3_corr = np.average(np.array(pointsx), weights=weights_f3)
            ys_f3_corr = np.average(np.array(pointsy), weights=weights_f3)

            weights = [stack_corrected_a12[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
            weights_a12 = [w if w>weight_th else 0 for w in weights ]
            xs_a12_corr = np.average(np.array(pointsx), weights=weights_a12)
            ys_a12_corr = np.average(np.array(pointsy), weights=weights_a12)

        stack_corrected = construct_RGB(R=stack_corrected_a12, G=stack_corrected_f3)

        f3_proj = np.mean(stack_corrected_f3, axis=0)
        a12_proj = np.mean(stack_corrected_a12, axis=0)
        casp3_proj = np.mean(stack_corrected_casp3, axis=0)
        comb = gaussian_filter1d(f3_proj*a12_proj, sigma=sigma)

        F3_PROJ.append(f3_proj)
        A12_PROJ.append(a12_proj)
        CASP3_PROJ.append(casp3_proj)
        borders.append(np.argmax(comb))

    F3_SIGNALS = np.ones((len(borders), len(F3_PROJ[0])*2 + 1))*np.nan
    A12_SIGNALS = np.ones_like(F3_SIGNALS)*np.nan
    CASP3_SIGNALS = np.ones_like(F3_SIGNALS)*np.nan

    for b in range(len(borders)):
        f3_signal = np.zeros(len(F3_PROJ[0])*2 + 1)
        a12_signal = np.zeros_like(f3_signal)
        casp3_signal = np.zeros_like(f3_signal)
        center = len(F3_PROJ[0])
        border = borders[b]
        F3_SIGNALS[b, center-border:center] = F3_PROJ[b][:border]
        F3_SIGNALS[b, center:2*center-border] = F3_PROJ[b][border:]

        A12_SIGNALS[b, center-border:center] = A12_PROJ[b][:border]
        A12_SIGNALS[b, center:2*center-border] = A12_PROJ[b][border:]

        CASP3_SIGNALS[b, center-border:center] = CASP3_PROJ[b][:border]/(F3_PROJ[b][:border]+A12_PROJ[b][:border])
        CASP3_SIGNALS[b, center:2*center-border] = CASP3_PROJ[b][border:]/(F3_PROJ[b][border:]+A12_PROJ[b][border:])


    f3_signal = np.nanmean(F3_SIGNALS, axis=0)
    f3_signal_std = np.nanstd(F3_SIGNALS, axis=0)

    a12_signal = np.nanmean(A12_SIGNALS, axis=0)
    a12_signal_std = np.nanstd(A12_SIGNALS, axis=0)

    casp3_signal = np.nanmean(CASP3_SIGNALS, axis=0)
    casp3_signal_std = np.nanstd(CASP3_SIGNALS, axis=0)

    x = [i for i in range(len(f3_signal))]

    fig, ax = plt.subplots()
    axc = ax.twinx()
    ax.plot(x, f3_signal, color="green", lw=4)
    ax.fill_between(x, f3_signal - f3_signal_std, f3_signal + f3_signal_std, color="green", alpha=0.2)

    ax.plot(x, a12_signal, color="magenta", lw=4)
    ax.fill_between(x, a12_signal - a12_signal_std, a12_signal + a12_signal_std, color="magenta", alpha=0.2)

    axc.plot(x, casp3_signal, color=[238/255, 210/255, 0.0], lw=4)
    axc.fill_between(x, casp3_signal - casp3_signal_std, casp3_signal + casp3_signal_std, color=[238/255, 210/255, 0.0], alpha=0.2)
    ax.set_title(cname)
    plt.savefig(path_data+cname+".png")