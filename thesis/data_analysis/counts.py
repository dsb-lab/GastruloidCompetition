### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, save_4Dstack, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
experiment = "2023_11_17_Casp3"
# experiment = "2024_03_Casp3"
TIMES = ["48hr", "72hr", "96hr"]
# TIMES = ["48hr", "60hr", "72hr", "96hr"]
CONDS = ["WT", "KO"]

fig, ax = plt.subplots(2,4, figsize=(3.5*4,6))

stages = ["early","mid", "late"]

F3_WT = []
A12_WT = []
Casp3_F3_WT =  []
Casp3_A12_WT =  []

F3_KO = []
A12_KO = []
Casp3_F3_KO = []
Casp3_A12_KO = []

for stage in stages:
    F3_WT.append([])
    A12_WT.append([])
    Casp3_F3_WT.append([])
    Casp3_A12_WT.append([])

    F3_KO.append([])
    A12_KO.append([])
    Casp3_F3_KO.append([])
    Casp3_A12_KO.append([])

    for TIME in TIMES:
        for COND in CONDS:
            path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/stacks/{}/{}/'.format(experiment, TIME, COND)
            path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/ctobjects/{}/{}/'.format(experiment, TIME, COND)

            try: 
                files = get_file_names(path_save_dir)
            except: 
                import os
                os.mkdir(path_save_dir)
                
            ### GET FULL FILE NAME AND FILE CODE ###
            files = get_file_names(path_data_dir)

            CENTERS = []
            FATES = []
            LABS = []

            # channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
            channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
            if "96hr" in path_data_dir:
                channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

            F3_counts = []
            Casp3_F3_counts = []
            A12_counts = []
            Casp3_A12_counts = []

            for f, file in enumerate(files):
                path_data = path_data_dir+file
                file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
                path_save = path_save_dir+embcode
                try: 
                    files = get_file_names(path_save)
                except: 
                    import os
                    os.mkdir(path_save)

                ### DEFINE ARGUMENTS ###
                segmentation_args={
                    'method': 'stardist2D', 
                    'model': model, 
                    'blur': [2,1], 
                    'min_outline_length':100,
                }

                concatenation3D_args = {
                    'distance_th_z': 3.0, # microns
                    'relative_overlap':False, 
                    'use_full_matrix_to_compute_overlap':True, 
                    'z_neighborhood':2, 
                    'overlap_gradient_th':0.3, 
                    'min_cell_planes': 2,
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
                    # 'plot_stack_dims': (256, 256), 
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
                    # 'plot_stack_dims': (256, 256), 
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

                ch = channel_names.index("Casp3")
                chans = [ch]
                for _ch in range(len(channel_names)):
                    if _ch not in chans:
                        chans.append(_ch)
                batch_args = {
                    'name_format':"ch"+str(ch)+"_{}_"+stage,
                    'extension':".tif",
                }
                plot_args = {
                    'plot_layout': (1,1),
                    'plot_overlap': 1,
                    'masks_cmap': 'tab10',
                    # 'plot_stack_dims': (256, 256), 
                    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
                    'channels':[ch],
                    'min_outline_length':75,
                }


                ### DEFINE ARGUMENTS ###
                segmentation_args={
                    'method': 'stardist2D', 
                    'model': model, 
                    'blur': [5,1], 
                    # 'n_tiles': (2,2),
                }
                CT_Casp3 = cellSegTrack(
                    path_data,
                    path_save,
                    segmentation_args=segmentation_args,
                    concatenation3D_args=concatenation3D_args,
                    error_correction_args=error_correction_args,
                    plot_args=plot_args,
                    batch_args=batch_args,
                    channels=chans
                )

                CT_Casp3.load()

                plot_args = {
                    'plot_layout': (1,1),
                    'plot_overlap': 1,
                    'masks_cmap': 'tab10',
                    'plot_stack_dims': (512, 512), 
                    'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
                    'channels':[1,0,3],
                    'min_outline_length':75,
                }

                # CT_Casp3.plot_tracking(plot_args=plot_args)

                
                F3_counts.append(len(CT_F3.jitcells))
                A12_counts.append(len(CT_A12.jitcells))

                import numpy as np

                ### CORRECT DISTRIBUTIONS ###

                centers = []
                areas = []

                F3_dist = []
                for cell in CT_F3.jitcells: 
                    for zid, z in enumerate(cell.zs[0]):
                        mask = cell.masks[0][zid]
                        img = CT_F3.hyperstack[0,z, channel_names.index("F3")]
                        F3_dist.append(np.mean(img[mask[:,1], mask[:,0]]))
                        areas.append(len(mask))
                    centers.append(cell.centers[0])
                        
                A12_dist = []
                for cell in CT_A12.jitcells: 
                    for zid, z in enumerate(cell.zs[0]):
                        mask = cell.masks[0][zid]
                        areas.append(len(mask))
                        img = CT_A12.hyperstack[0,z, channel_names.index("A12")]
                        A12_dist.append(np.mean(img[mask[:,1], mask[:,0]]))
                    centers.append(cell.centers[0])

                area = np.mean(areas)
                dim = 2*np.sqrt(area/np.pi)

                F3_dist = np.array(F3_dist)
                A12_dist = np.array(A12_dist)

                mdiff = np.mean(F3_dist) - np.mean(A12_dist)
                if mdiff > 0:
                    A12_dist += mdiff
                else: 
                    F3_dist -= mdiff 

                zres = CT_F3.metadata["Zresolution"]
                xyres = CT_F3.metadata["XYresolution"]

                fates = []
                centers = []
                labels = []
                for cell in CT_F3.jitcells:
                    fates.append(0)
                    centers.append(cell.centers[0]*[zres, xyres, xyres])
                    labels.append(cell.label)
                for cell in CT_A12.jitcells:
                    fates.append(1)
                    centers.append(cell.centers[0]*[zres, xyres, xyres])
                    labels.append(cell.label)

                Casp3_F3 = 0
                Casp3_A12 = 0
                len_pre_casp3 = len(centers)
                for cell in CT_Casp3.jitcells:
                    casp3 = []
                    a12 = []
                    f3 = []
                    for zid, z in enumerate(cell.zs[0]):
                        mask = cell.masks[0][zid]
                        img = CT_Casp3.hyperstack[0,z, channel_names.index("Casp3")]
                        casp3.append(np.mean(img[mask[:,1], mask[:,0]]))
                        
                        img = CT_Casp3.hyperstack[0,z, channel_names.index("A12")]
                        a12.append(np.mean(img[mask[:,1], mask[:,0]]))
                        
                        img = CT_Casp3.hyperstack[0,z, channel_names.index("F3")]
                        f3.append(np.mean(img[mask[:,1], mask[:,0]]))

                    centers.append(cell.centers[0]*[zres, xyres, xyres])
                    labels.append(cell.label)

                    zz = np.int64(cell.centers[0][0])
                    idx = cell.zs[0].index(zz)
                    if f3[idx] > a12[idx]:
                        fates.append(2)
                        Casp3_F3+=1
                    else:
                        fates.append(3)
                        Casp3_A12+=1
                
                Casp3_F3_counts.append(Casp3_F3)
                Casp3_A12_counts.append(Casp3_A12)

            if COND == "WT":
                F3_WT[-1].append(F3_counts)
                A12_WT[-1].append(A12_counts)
                Casp3_F3_WT[-1].append(Casp3_F3_counts)
                Casp3_A12_WT[-1].append(Casp3_A12_counts)

            elif COND == "KO":
                F3_KO[-1].append(F3_counts)
                A12_KO[-1].append(A12_counts)
                Casp3_F3_KO[-1].append(Casp3_F3_counts)
                Casp3_A12_KO[-1].append(Casp3_A12_counts)

path_figures = "/home/pablo/Desktop/PhD/projects/GastruloidCompetition/thesis/figures/counts/"

all_files = []
all_data = []
all_names = ["file_name", "F3", "A12", "apo_F3_early", "apo_A12_early", "apo_F3_mid", "apo_A12_mid", "apo_F3_late", "apo_A12_late"]
for T, TIME in enumerate(TIMES):
    for C,COND in enumerate(CONDS):
        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/stacks/{}/{}/'.format(experiment, TIME, COND)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/{}/ctobjects/{}/{}/'.format(experiment, TIME, COND)

        try: 
            files = get_file_names(path_save_dir)
        except: 
            import os
            os.mkdir(path_save_dir)
            
        ### GET FULL FILE NAME AND FILE CODE ###
        files = get_file_names(path_data_dir)
        print(files)
        all_files = [*all_files, *files]
        for f, file in enumerate(files):
            data = []
            if COND == "WT":
                data.append(F3_WT[0][T][f])
                data.append(A12_WT[0][T][f])
                data.append(Casp3_F3_WT[0][T][f])
                data.append(Casp3_A12_WT[0][T][f])
                data.append(Casp3_F3_WT[1][T][f])
                data.append(Casp3_A12_WT[1][T][f])
                data.append(Casp3_F3_WT[2][T][f])
                data.append(Casp3_A12_WT[2][T][f])
            elif COND == "KO":
                data.append(F3_KO[0][T][f])
                data.append(A12_KO[0][T][f])
                data.append(Casp3_F3_KO[0][T][f])
                data.append(Casp3_A12_KO[0][T][f])
                data.append(Casp3_F3_KO[1][T][f])
                data.append(Casp3_A12_KO[1][T][f])
                data.append(Casp3_F3_KO[2][T][f])
                data.append(Casp3_A12_KO[2][T][f])
            all_data.append(data)

import csv
# Output CSV file path
output_file = path_figures+"data_counts.csv"
# Write to CSV
with open(output_file, mode="w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(all_names)
    
    # Write the data rows
    for file, values in zip(all_files, all_data):
        csv_writer.writerow([file] + values)


import matplotlib as mpl
plt.rcParams.update({
    "text.usetex": True,
})
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath'
mpl.rc('font', size=16) 
mpl.rc('axes', labelsize=16) 
mpl.rc('xtick', labelsize=16) 
mpl.rc('ytick', labelsize=16) 
mpl.rc('legend', fontsize=16) 

fig, ax = plt.subplots(2,4, figsize=(3.5*4,6))

ax[0,0].set_title("total cells")
ax[0,1].set_title("early apoptotis")
ax[0,2].set_title("mid apoptotis")
ax[0,3].set_title("late apoptotis")


ax[0,0].set_ylabel(r"WT \\ cell number")
ax[1,0].set_ylabel(r"KO \\ cell number")

ax[0,1].set_ylabel("apoptosis ratio")
ax[1,1].set_ylabel("apoptosis ratio")

### WT ###

time=range(len(TIMES))

F3_WT_means = np.array([np.mean(data) for data in F3_WT[0]])
F3_WT_stds = np.array([np.std(data) for data in F3_WT[0]])

ax[0,0].plot(time, F3_WT_means, color=[0.0,0.8,0.0], lw=2, zorder=1)
ax[0,0].scatter(time, F3_WT_means, color=[0.0,0.6,0.0], edgecolor='k', s=50, zorder=2)
ax[0,0].fill_between(time, F3_WT_means-F3_WT_stds, F3_WT_means+F3_WT_stds, alpha=0.1, color=[0.0,0.8,0.0])

A12_WT_means = np.array([np.mean(data) for data in A12_WT[0]])
A12_WT_stds = np.array([np.std(data) for data in A12_WT[0]])

ax[0,0].plot(time, A12_WT_means, color=[0.8,0.0,0.8], lw=2, zorder=1)
ax[0,0].scatter(time, A12_WT_means, color=[0.6,0.0,0.6], edgecolor='k', s=50, zorder=2)
ax[0,0].fill_between(time, A12_WT_means-A12_WT_stds, A12_WT_means+A12_WT_stds, alpha=0.1, color=[0.8,0.0,0.8])


F3_KO_means = np.array([np.mean(data) for data in F3_KO[0]])
F3_KO_stds = np.array([np.std(data) for data in F3_KO[0]])

ax[1,0].plot(time, F3_KO_means, color=[0.0,0.8,0.0], lw=2, zorder=1)
ax[1,0].scatter(time, F3_KO_means, color=[0.0,0.6,0.0], edgecolor='k', s=50, zorder=2)
ax[1,0].fill_between(time, F3_KO_means-F3_KO_stds, F3_KO_means+F3_KO_stds, alpha=0.1, color=[0.8,0.0,0.8])

A12_KO_means = np.array([np.mean(data) for data in A12_KO[0]])
A12_KO_stds = np.array([np.std(data) for data in A12_KO[0]])

ax[1,0].plot(time, A12_KO_means, color=[0.8,0.0,0.8], lw=2, zorder=1)
ax[1,0].scatter(time, A12_KO_means, color=[0.6,0.0,0.6], edgecolor='k', s=50, zorder=2)
ax[1,0].fill_between(time, A12_KO_means-A12_KO_stds, A12_KO_means+A12_KO_stds, alpha=0.1, color=[0.8,0.0,0.8])
ax[1,0].sharey(ax[0,0])

ax[0, 0].set_xticks([i for i in time])
ax[0, 0].set_xticklabels([None]*len(time))
ax[1, 0].set_xticks([i for i in time])
ax[1, 0].set_xticklabels(TIMES)

ax[0, 0].spines[['right', 'top']].set_visible(False)
ax[1, 0].spines[['right', 'top']].set_visible(False)

for st, stage in enumerate(stages):

    DATA = [np.array(Casp3_F3_WT[st][n])/(np.array(F3_WT[st][n]) + np.array(Casp3_F3_WT[st][n])) for n in range(len(F3_WT[st]))]
    Casp3_F3_WT_means = np.array([np.mean(data) for data in DATA])
    Casp3_F3_WT_stds = np.array([np.std(data) for data in DATA])

    ax[0,st+1].plot(time, Casp3_F3_WT_means, color=[0.0,0.8,0.0], lw=2, zorder=1)
    ax[0,st+1].scatter(time, Casp3_F3_WT_means, color=[0.0,0.6,0.0], edgecolor='k', s=50, zorder=2)
    ax[0,st+1].fill_between(time, Casp3_F3_WT_means-Casp3_F3_WT_stds, Casp3_F3_WT_means+Casp3_F3_WT_stds, alpha=0.1, color=[0.0,0.8,0.0])

    DATA = [np.array(Casp3_F3_KO[st][n])/(np.array(F3_KO[st][n]) + np.array(Casp3_F3_KO[st][n])) for n in range(len(F3_KO[st]))]
    Casp3_F3_KO_means = np.array([np.mean(data) for data in DATA])
    Casp3_F3_KO_stds = np.array([np.std(data) for data in DATA])

    ax[1,st+1].plot(time, Casp3_F3_KO_means, color=[0.0,0.8,0.0], lw=2, zorder=1)
    ax[1,st+1].scatter(time, Casp3_F3_KO_means, color=[0.0,0.6,0.0], edgecolor='k', s=50, zorder=2)
    ax[1,st+1].fill_between(time, Casp3_F3_KO_means-Casp3_F3_KO_stds, Casp3_F3_KO_means+Casp3_F3_KO_stds, alpha=0.1, color=[0.0,0.8,0.0])
    ax[1, st+1].sharey(ax[0, st+1])

    DATA = [np.array(Casp3_A12_WT[st][n])/(np.array(A12_WT[st][n]) + np.array(Casp3_A12_WT[st][n])) for n in range(len(A12_WT[st]))]
    Casp3_A12_WT_means = np.array([np.mean(data) for data in DATA])
    Casp3_A12_WT_stds = np.array([np.std(data) for data in DATA])

    ax[0,st+1].plot(time, Casp3_A12_WT_means, color=[0.8,0.0,0.8], lw=2, zorder=1)
    ax[0,st+1].scatter(time, Casp3_A12_WT_means, color=[0.6,0.0,0.6], edgecolor='k', s=50, zorder=2)
    ax[0,st+1].fill_between(time, Casp3_A12_WT_means-Casp3_A12_WT_stds, Casp3_A12_WT_means+Casp3_A12_WT_stds, alpha=0.1, color=[0.8,0.0,0.8])

    ax[0, st+1].set_xticks([i for i in time])
    ax[0, st+1].set_xticklabels([None]*len(time))

    DATA = [np.array(Casp3_A12_KO[st][n])/(np.array(A12_KO[st][n]) + np.array(Casp3_A12_KO[st][n])) for n in range(len(A12_KO[st]))]
    Casp3_A12_KO_means = np.array([np.mean(data) for data in DATA])
    Casp3_A12_KO_stds = np.array([np.std(data) for data in DATA])

    ax[1,st+1].plot(time, Casp3_A12_KO_means, color=[0.8,0.0,0.8], lw=2, zorder=1)
    ax[1,st+1].scatter(time, Casp3_A12_KO_means, color=[0.6,0.0,0.6], edgecolor='k', s=50, zorder=2)
    ax[1,st+1].fill_between(time, Casp3_A12_KO_means-Casp3_A12_KO_stds, Casp3_A12_KO_means+Casp3_A12_KO_stds, alpha=0.1, color=[0.8,0.0,0.8])
    ax[1, st+1].sharey(ax[0, st+1])
    
    ax[0, st+1].spines[['right', 'top']].set_visible(False)
    ax[1, st+1].spines[['right', 'top']].set_visible(False)
    ax[1, st+1].set_xticks([i for i in time])
    ax[1, st+1].set_xticklabels(TIMES)

plt.tight_layout()
plt.savefig(path_figures+"counts_apo.svg")
plt.savefig(path_figures+"counts_apo.pdf")
plt.show()

barwidth = 0.5

fig, ax = plt.subplots(1, 4, figsize=(14, 3.5))

ax[0].set_title("total cells")
ax[1].set_title("early apoptotis")
ax[2].set_title("mid apoptotis")
ax[3].set_title("late apoptotis")
ax[0].set_ylabel("cell number")
ax[1].set_ylabel("apoptosis ratio")

from scipy.stats import ttest_ind

pvalues = []
for cc, condition in enumerate(F3_WT[0]):
    tt = ttest_ind(np.array(F3_WT[0][cc]), np.array(F3_KO[0][cc]))
    pvalues.append(tt.pvalue)
    print(tt.pvalue)

data = [np.array([np.mean(data) for data in F3_WT[0]]), np.array([np.mean(data) for data in F3_KO[0]])]
data_std = [np.array([np.std(data) for data in F3_WT[0]]), np.array([np.std(data) for data in F3_KO[0]])]
maxz = np.max(np.array(data) + np.array(data_std))
x = np.arange(len(TIMES)) * 1.5 + 1
for i in range(2):
    if i==0:
        offset=-barwidth/2
        color = [0.0,0.3,0.0]
    elif i == 1:
        offset = +barwidth/2
        color = [0.0,0.8,0.0]

    rects = ax[0].bar(x + offset, np.array(data[i]), barwidth, label=CONDS[i], color=color, edgecolor="k")
    ax[0].errorbar(x+offset,  np.array(data[i]), yerr=np.array(data_std[i]), capsize=5, fmt="o", color="k")
    
    if i==1:
        for p, pval in enumerate(pvalues):
            x1 = x[p]-offset
            x2 = x[p]+offset
            y = maxz*1.1
            h = maxz*0.01
            ax[0].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c="k")
            if pval > 0.05:
                ax[0].text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color="k", fontsize=12)
            elif pval > 0.01:
                ax[0].text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color="k", fontsize=12)
            elif pval > 0.001:
                ax[0].text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color="k", fontsize=12)
            else:
                ax[0].text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color="k", fontsize=12)
                    
ax[0].set_xticks(x, TIMES)
ax[0].spines[['right', 'top']].set_visible(False)

DATA1 = [np.array(Casp3_F3_WT[0][n])/(np.array(F3_WT[0][n]) + np.array(Casp3_F3_WT[0][n])) for n in range(len(F3_WT[0]))]
DATA2 = [np.array(Casp3_F3_KO[0][n])/(np.array(F3_KO[0][n]) + np.array(Casp3_F3_KO[0][n])) for n in range(len(F3_KO[0]))]

pvalues = []
for cc, condition in enumerate(DATA1):
    tt = ttest_ind(np.array(DATA1[cc]), np.array(DATA2[cc]))
    pvalues.append(tt.pvalue)
    print(tt.pvalue)

data = [np.array([np.mean(data) for data in DATA1]), np.array([np.mean(data) for data in DATA2])]
data_std = [np.array([np.std(data) for data in DATA1]), np.array([np.std(data) for data in DATA2])]
maxz = np.max(np.array(data) + np.array(data_std))
x = np.arange(len(TIMES)) * 1.5 + 1
for i in range(2):
    if i==0:
        offset=-barwidth/2
        color = [0.0,0.3,0.0]
    elif i == 1:
        offset = +barwidth/2
        color = [0.0,0.8,0.0]

    rects = ax[1].bar(x + offset, np.array(data[i]), barwidth, label=CONDS[i], color=color, edgecolor="k")
    ax[1].errorbar(x+offset,  np.array(data[i]), yerr=np.array(data_std[i]), capsize=5, fmt="o", color="k")
    
    if i==1:
        for p, pval in enumerate(pvalues):
            x1 = x[p]-offset
            x2 = x[p]+offset
            y = maxz*1.1
            h = maxz*0.01
            ax[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c="k")
            if pval > 0.05:
                ax[1].text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color="k", fontsize=12)
            elif pval > 0.01:
                ax[1].text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color="k", fontsize=12)
            elif pval > 0.001:
                ax[1].text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color="k", fontsize=12)
            else:
                ax[1].text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color="k", fontsize=12)
                    
ax[1].set_xticks(x, TIMES)
ax[1].spines[['right', 'top']].set_visible(False)

DATA1 = [np.array(Casp3_F3_WT[1][n])/(np.array(F3_WT[1][n]) + np.array(Casp3_F3_WT[1][n])) for n in range(len(F3_WT[1]))]
DATA2 = [np.array(Casp3_F3_KO[1][n])/(np.array(F3_KO[1][n]) + np.array(Casp3_F3_KO[1][n])) for n in range(len(F3_KO[1]))]

pvalues = []
for cc, condition in enumerate(DATA1):
    tt = ttest_ind(np.array(DATA1[cc]), np.array(DATA2[cc]))
    pvalues.append(tt.pvalue)
    print(tt.pvalue)

data = [np.array([np.mean(data) for data in DATA1]), np.array([np.mean(data) for data in DATA2])]
data_std = [np.array([np.std(data) for data in DATA1]), np.array([np.std(data) for data in DATA2])]
maxz = np.max(np.array(data) + np.array(data_std))
x = np.arange(len(TIMES)) * 1.5 + 1
for i in range(2):
    if i==0:
        offset=-barwidth/2
        color = [0.0,0.3,0.0]
    elif i == 1:
        offset = +barwidth/2
        color = [0.0,0.8,0.0]

    rects = ax[2].bar(x + offset, np.array(data[i]), barwidth, label=CONDS[i], color=color, edgecolor="k")
    ax[2].errorbar(x+offset,  np.array(data[i]), yerr=np.array(data_std[i]), capsize=5, fmt="o", color="k")
    
    if i==1:
        for p, pval in enumerate(pvalues):
            x1 = x[p]-offset
            x2 = x[p]+offset
            y = maxz*1.1
            h = maxz*0.01
            ax[2].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c="k")
            if pval > 0.05:
                ax[2].text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color="k", fontsize=12)
            elif pval > 0.01:
                ax[2].text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color="k", fontsize=12)
            elif pval > 0.001:
                ax[2].text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color="k", fontsize=12)
            else:
                ax[2].text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color="k", fontsize=12)
                    
ax[2].set_xticks(x, TIMES)
ax[2].spines[['right', 'top']].set_visible(False)

DATA1 = [np.array(Casp3_F3_WT[2][n])/(np.array(F3_WT[2][n]) + np.array(Casp3_F3_WT[2][n])) for n in range(len(F3_WT[2]))]
DATA2 = [np.array(Casp3_F3_KO[2][n])/(np.array(F3_KO[2][n]) + np.array(Casp3_F3_KO[2][n])) for n in range(len(F3_KO[2]))]

pvalues = []
for cc, condition in enumerate(DATA1):
    tt = ttest_ind(np.array(DATA1[cc]), np.array(DATA2[cc]))
    pvalues.append(tt.pvalue)
    print(tt.pvalue)

data = [np.array([np.mean(data) for data in DATA1]), np.array([np.mean(data) for data in DATA2])]
data_std = [np.array([np.std(data) for data in DATA1]), np.array([np.std(data) for data in DATA2])]
maxz = np.max(np.array(data) + np.array(data_std))
x = np.arange(len(TIMES)) * 1.5 + 1
for i in range(2):
    if i==0:
        offset=-barwidth/2
        color = [0.0,0.3,0.0]
    elif i == 1:
        offset = +barwidth/2
        color = [0.0,0.8,0.0]

    rects = ax[3].bar(x + offset, np.array(data[i]), barwidth, label=CONDS[i], color=color, edgecolor="k")
    ax[3].errorbar(x+offset,  np.array(data[i]), yerr=np.array(data_std[i]), capsize=5, fmt="o", color="k")
    
    if i==1:
        for p, pval in enumerate(pvalues):
            x1 = x[p]-offset
            x2 = x[p]+offset
            y = maxz*1.1
            h = maxz*0.01
            ax[3].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c="k")
            if pval > 0.05:
                ax[3].text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color="k", fontsize=12)
            elif pval > 0.01:
                ax[3].text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color="k", fontsize=12)
            elif pval > 0.001:
                ax[3].text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color="k", fontsize=12)
            else:
                ax[3].text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color="k", fontsize=12)
                    
ax[3].set_xticks(x, TIMES)
ax[3].spines[['right', 'top']].set_visible(False)
ax[3].legend()
plt.tight_layout()
plt.savefig(path_figures+"counts_comparison.svg")
plt.savefig(path_figures+"counts_comparison.pdf")
plt.show()



