from qlivecell import correct_drift, cellSegTrack, extract_fluoro, get_file_names, get_file_name, get_intenity_profile, construct_RGB
import os

path_data = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/merge/2024_07_22_DMSOmerged_Casp3/"
path_save = "/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/merge/stresults/"

channel_names = ["A12", "Casp3", "F3", "DAPI"]

try: 
    files = get_file_names(path_save)
except: 
    import os
    os.makedirs(path_save)


### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

### DEFINE ARGUMENTS ###
segmentation_args={
    'method': 'stardist2D', 
    'model': model, 
    'blur': None, 
    # 'n_tiles': (2,2),
}

concatenation3D_args = {
    'distance_th_z': 3.0, # microns
    'relative_overlap':False, 
    'use_full_matrix_to_compute_overlap':True, 
    'z_neighborhood':2, 
    'overlap_gradient_th':0.3, 
    'min_cell_planes': 1,
}

tracking_args = {
    'time_step': 5, # minutes
    'method': 'greedy', 
    'z_th':10, 
    'dist_th' : 10.0,
}

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    # 'plot_stack_dims': (512, 512), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[1]
}

error_correction_args = {
    'backup_steps': 10,
    'line_builder_mode': 'points',
}

files = get_file_names(path_data)
files = [file for file in files if ".tif" in file]
file = files[12]

path_data_file = path_data+file
file, embcode = get_file_name(path_data, file, allow_file_fragment=False, return_files=False, return_name=True)
path_save_file = path_save+embcode
            
ch_F3 = channel_names.index("F3")
batch_args = {
    'name_format':"ch"+str(ch_F3)+"_{}",
    'extension':".tif",
}

chans = [ch_F3]
for _ch in range(len(channel_names)):
    if _ch not in chans:
        chans.append(_ch)

cST_F3 = cellSegTrack(
    path_data_file,
    path_save_file,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    tracking_args=tracking_args,
    error_correction_args=error_correction_args,
    plot_args=plot_args,
    batch_args=batch_args,
    channels=chans
)

cST_F3.run()

ch_A12 = channel_names.index("A12")
batch_args = {
    'name_format':"ch"+str(ch_A12)+"_{}",
    'extension':".tif",
}

chans = [ch_A12]
for _ch in range(len(channel_names)):
    if _ch not in chans:
        chans.append(_ch)

cST_A12 = cellSegTrack(
    path_data_file,
    path_save_file,
    segmentation_args=segmentation_args,
    concatenation3D_args=concatenation3D_args,
    tracking_args=tracking_args,
    error_correction_args=error_correction_args,
    plot_args=plot_args,
    batch_args=batch_args,
    channels=chans
)

cST_A12.run()

import numpy as np 

results_F3 = extract_fluoro(cST_F3)
f3 = results_F3["channel_{}".format(ch_F3)]
correction_function_F3, intensity_profile_F3, z_positions_F3 = get_intenity_profile(cST_F3, ch_F3, cell_number_threshold=7.0, fit_everything=False)

stack = cST_F3.hyperstack[0,:,ch_F3].astype("float32")
for z in range(stack.shape[0]):
    stack[z] = stack[z] / correction_function_F3[z]
stack *= np.mean(intensity_profile_F3)
# cST_F3.hyperstack[0,:,ch_F3] = stack.astype("uint8")

results_A12 = extract_fluoro(cST_A12)
a12 = results_A12["channel_{}".format(ch_A12)]

correction_function_A12, intensity_profile_A12, z_positions_A12 = get_intenity_profile(cST_A12, ch_A12, cell_number_threshold=7.0, fit_everything=False)
stack = cST_A12.hyperstack[0,:,ch_A12].astype("float32")
for z in range(stack.shape[0]):
    stack[z] = stack[z] / correction_function_A12[z]
stack *= np.mean(intensity_profile_A12)
# cST_A12.hyperstack[0,:,ch_A12] = stack.astype("uint8")

ch_Casp3 = channel_names.index("Casp3")

plot_args = {
    'plot_layout': (1,1),
    'plot_overlap': 1,
    'masks_cmap': 'tab10',
    # 'plot_stack_dims': (512, 512), 
    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
    'channels':[ch_A12]
}
cST_A12.plot(plot_args=plot_args)

# NOW WE CAN COMPUTE THE CENTROID OF EACH CHANNEL


stack_f3 = np.max(cST_F3.hyperstack[0,:,ch_F3], axis=0)
stack_a12 = np.max(cST_A12.hyperstack[0,:,ch_A12], axis=0)
stack_casp3 = np.max(cST_A12.hyperstack[0,:,ch_Casp3], axis=0)

pointsx = [i for i in range(stack_f3.shape[0]) for j in range(stack_f3.shape[1])]
pointsy = [j for i in range(stack_f3.shape[0]) for j in range(stack_f3.shape[1])]
weights = [stack_f3[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
weights_f3 = [w if w>20 else 0 for w in weights]
xs_f3 = np.average(np.array(pointsx), weights=weights_f3)
ys_f3 = np.average(np.array(pointsy), weights=weights_f3)

weights = [stack_a12[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
weights_a12 = [w if w>20 else 0 for w in weights ]
xs_a12 = np.average(np.array(pointsx), weights=weights_a12)
ys_a12 = np.average(np.array(pointsy), weights=weights_a12)

stack = construct_RGB(R=stack_a12, G=stack_f3)
import matplotlib.pyplot as plt
plt.imshow(stack)
plt.scatter([ys_f3],[xs_f3], color="white", s=50)
plt.scatter([ys_a12],[xs_a12], color="white", s=50)

plt.show()


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
weights_f3 = [w if w>20 else 0 for w in weights]
xs_f3_corr = np.average(np.array(pointsx), weights=weights_f3)
ys_f3_corr = np.average(np.array(pointsy), weights=weights_f3)

weights = [stack_corrected_a12[pointsx[i],pointsy[i]] for i in range(len(pointsx))]
weights_a12 = [w if w>20 else 0 for w in weights ]
xs_a12_corr = np.average(np.array(pointsx), weights=weights_a12)
ys_a12_corr = np.average(np.array(pointsy), weights=weights_a12)

image = sitk.GetImageFromArray(stack_casp3)
rotated_image = resampler.Execute(image)
stack_corrected_casp3 = sitk.GetArrayFromImage(rotated_image)

stack_corrected = construct_RGB(R=stack_corrected_a12, G=stack_corrected_f3)

fig, ax = plt.subplots(2,2, figsize=(10,10))
ax[0,0].imshow(stack)
ax[0,0].scatter([ys_f3],[xs_f3], color="white", s=50)
ax[0,0].scatter([ys_a12],[xs_a12], color="white", s=50)
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
f3_proj = np.mean(stack_f3, axis=0)
ax[1,0].plot(f3_proj, c="green", lw=4)
a12_proj = np.mean(stack_a12, axis=0)
ax[1,0].plot(a12_proj, c="magenta", lw=4)
casp3_proj = np.mean(stack_casp3, axis=0)
ax[1,0].plot(casp3_proj, c="yellow", lw=4)

ax[0,1].imshow(stack_corrected)
ax[0,1].scatter([ys_f3_corr],[xs_f3_corr], color="white", s=50)
ax[0,1].scatter([ys_a12_corr],[xs_a12_corr], color="white", s=50)
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
f3_proj = np.mean(stack_corrected_f3, axis=0)
ax[1,1].plot(f3_proj, c="green", lw=4)
a12_proj = np.mean(stack_corrected_a12, axis=0)
ax[1,1].plot(a12_proj, c="magenta", lw=4)
casp3_proj = np.mean(stack_corrected_casp3, axis=0)
ax[1,1].plot(casp3_proj, c="yellow", lw=4)
plt.tight_layout()
plt.show()
