### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, check_or_create_dir, get_file_names

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
mpl.rc('font', size=22) 
mpl.rc('axes', labelsize=22) 
mpl.rc('xtick', labelsize=22) 
mpl.rc('ytick', labelsize=22) 
mpl.rc('legend', fontsize=20) 

F3_all = [[] for z in range(10)]
F3_F3 = [[] for z in range(10)]
F3_A12 = [[] for z in range(10)]
F3_DAPI = [[] for z in range(10)]

A12_all = [[] for z in range(10)]
A12_F3 = [[] for z in range(10)]
A12_A12 = [[] for z in range(10)]
A12_DAPI = [[] for z in range(10)]

DAPI_all = [[] for z in range(10)]

colors = [[] for z in range(10)]
fates = [[] for z in range(10)]

zs = []
all_files = []

CONDS = ["auxin_48-72_48", "auxin_48-72_72" , "auxin_48-72_96", "auxin_72-96_72", "auxin_72-96_96", "noauxin_72", "noauxin_96"]

calibF3 = np.load("/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/secondaryonly/F3(150)+OsTIR9-40(25)_48h_emiRFP-2ndaryA488-mCh-DAPI_(40xSil)_Stack1/calibration_F3_to_p53.npz")
p53_F3_s_global = float(calibF3["s"])
p53_F3_0z = calibF3["b0z"]

calibA12 = np.load("/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/secondaryonly/F3(150)+OsTIR9-40(25)_48h_emiRFP-2ndaryA488-mCh-DAPI_(40xSil)_Stack1/calibration_A12_to_p53.npz")
p53_A12_s_global = float(calibA12["s"])
p53_A12_0z = calibA12["b0z"]

for C, COND in enumerate(CONDS):
    path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
    path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/test/{}/".format(COND)
        
    check_or_create_dir(path_save_dir)

    files = get_file_names(path_data_dir)

    channel_names = ["A12", "p53", "F3", "DAPI"]

    for f, file in enumerate(files):
            
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

        CT_F3.run()

        ch = channel_names.index("A12")
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

        CT_A12.run()
        zs.append(CT_A12.hyperstack.shape[1])
        
        ch_F3 = channel_names.index("F3")
        ch_A12 = channel_names.index("A12")
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

            colors[z].append([0.0,0.8,0.0, 0.3])
            fates[z].append("F3")
            
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
            colors[z].append([0.8,0.0,0.8, 0.3]) 
            fates[z].append("A12")

for z in range(zs[-1]):
    fig, ax = plt.subplots()
    ax.set_xlabel("mean H2B-mCherry [a.u.]")
    ax.set_ylabel("mean H2B-emiRFP [a.u.]")            
    ax.scatter(F3_all[z], A12_all[z],c=colors[z], edgecolors='none')
    # ax.set_xlim(-5, 20000)
    # ax.set_ylim(-5, 20000)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.spines[['right', 'top']].set_visible(False)
    # plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/clustering/corrected_z{}.svg".format(z))
    plt.tight_layout()
    plt.show()

F3_means = np.array([np.mean(f3) for f3 in F3_F3])
F3_stds = np.array([np.std(f3) for f3 in F3_F3])

A12_means = np.array([np.mean(a12) for a12 in A12_A12])
A12_stds = np.array([np.std(a12) for a12 in A12_A12])

DAPI_means = np.array([np.mean(dapi) for dapi in DAPI_all])
DAPI_stds = np.array([np.std(dapi) for dapi in DAPI_all])

fig, ax = plt.subplots()

ax.plot(range(10), F3_means, color=[0.0,0.8,0.0, 1.0], label="H2B-mCherry on F3")
ax.fill_between(range(10), F3_means - F3_stds, F3_means + F3_stds, color=[0.0,0.8,0.0], alpha=0.2)

ax.plot(range(10), A12_means, color=[0.8,0.0,0.8, 1.0], label="H2B-emiRFP on A12")
ax.fill_between(range(10), A12_means - A12_stds, A12_means + A12_stds, color=[0.8,0.0,0.8], alpha=0.2)

ax.plot(range(10), DAPI_means, color="cyan", label="DAPI on all cells")
ax.fill_between(range(10), DAPI_means - DAPI_stds, DAPI_means + DAPI_stds, color="cyan", alpha=0.2)

ax.set_xlabel("z")
ax.set_ylabel("fluoro [a.u.]")
ax.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.cluster import SpectralClustering


# Instantiate KMeans with the number of clusters
remove_cell = [[] for z in range(10)]

from sklearn.cluster import DBSCAN
import numpy as np

for z in range(10):
    data1 = F3_all[z]
    data2 = A12_all[z]
    checkvar = fates[z]
    X = np.transpose(np.asarray([np.log(data1), np.log(data2)]))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    cluster_centers = kmeans.cluster_centers_
    clustered_fates = np.argsort(cluster_centers[:,0])
    labels = kmeans.labels_

    # clustering = DBSCAN(eps=0.5, min_samples=2).fit(X)
    # labels = clustering.labels_
    # clustered_fates = np.unique(labels)

    # clustering = SpectralClustering(n_clusters=2,
    #                             assign_labels='discretize',
    #                             random_state=0).fit(X)
    # labels = clustering.labels_
    # clustered_fates = np.unique(labels)
    
    colors_clustering = []

    A12th = np.percentile(data2,20)
    F3th = np.percentile(data1,20)

    for i, lab in enumerate(labels):
        if lab==clustered_fates[0]:
            if checkvar[i]!="A12":
                colors_clustering.append([0.2, 0.2, 0.2, 1.0])
                remove_cell[z].append(True)
            else:
                colors_clustering.append([0.8, 0.0, 0.8, 0.7])
                remove_cell[z].append(False)
        elif lab==clustered_fates[-1]:
            if checkvar[i]!="F3":
                colors_clustering.append([0.2, 0.2, 0.2, 1.0])
                remove_cell[z].append(True)
            else:
                remove_cell[z].append(False)
                colors_clustering.append([0.0, 0.8, 0.0, 0.7])
        else:
            remove_cell[z].append(False)
            colors_clustering.append([0.0, 0.0, 0.0, 0.7])

        if checkvar[i]=="A12":
            if data2[i] < A12th:
                remove_cell[z][-1]=True
                colors_clustering[-1] = [0.2, 0.2, 0.2, 1.0]
        else:
            if data1[i] < F3th:
                remove_cell[z][-1]=True
                colors_clustering[-1] = [0.2, 0.2, 0.2, 1.0]
            
    # Plot the original data points and cluster centers
    from matplotlib.lines import Line2D
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=colors_clustering, edgecolors='k')
    # plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', s=200, linewidths=3, color='red', label='Cluster Centers')
    plt.xlabel('log(H2B-emiRFG)')
    plt.ylabel('log(H2B-mCherry)')
    plt.tight_layout()

    # Custom legend handles
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="A12",
            markerfacecolor=(0.8, 0.0, 0.8, 0.7), markersize=10),
        Line2D([0], [0], marker="o", color="w", label="F3",
            markerfacecolor=(0.0, 0.8, 0.0, 0.7), markersize=10),
        Line2D([0], [0], marker="o", color="w", label="discarded cells",
            markerfacecolor=(0.2, 0.2, 0.2, 1.0), markersize=10),
    ]
    plt.legend(handles=legend_elements, loc="best", frameon=True)
    plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/clustering/clusteringloglog_z{}.svg".format(z))
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(data1, data2, c=colors_clustering, edgecolors='k')
    plt.xlabel('H2B-emiRFP')
    plt.ylabel('H2B-mCherry')
    plt.tight_layout()

    # Custom legend handles
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="A12",
            markerfacecolor=(0.8, 0.0, 0.8, 0.7), markersize=10),
        Line2D([0], [0], marker="o", color="w", label="F3",
            markerfacecolor=(0.0, 0.8, 0.0, 0.7), markersize=10),
        Line2D([0], [0], marker="o", color="w", label="discarded cells",
            markerfacecolor=(0.2, 0.2, 0.2, 1.0), markersize=10),
    ]
    plt.legend(handles=legend_elements, loc="best", frameon=True)
    plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53_degrons/clustering/clustering_z{}.svg".format(z))
    plt.show()

# # Noy is time for the removal
# current_zid = [0 for z in range(10)]
# for C, COND in enumerate(CONDS):
#     path_data_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
#     path_save_dir="/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/p53_analysis/segobjects/2025_09_09_OsTIRMosaic_p53Timecourse/{}/".format(COND)
        
#     check_or_create_dir(path_save_dir)

#     files = get_file_names(path_data_dir)

#     channel_names = ["A12", "p53", "F3", "DAPI"]

#     for f, file in enumerate(files):
                
#         all_files.append(file)
#         path_data = path_data_dir+file
#         file, embcode = get_file_name(path_data_dir, file, allow_file_fragment=False, return_files=False, return_name=True)
#         path_save = path_save_dir+embcode
            
#         check_or_create_dir(path_save)

#         ### DEFINE ARGUMENTS ###
#         segmentation_args={
#             'method': 'stardist2D', 
#             'model': model, 
#             'blur': [2,1], 
#             'min_outline_length':100,
#         }

#         concatenation3D_args = {
#             'do_3Dconcatenation': False
#         }

#         error_correction_args = {
#             'backup_steps': 10,
#             'line_builder_mode': 'points',
#         }

#         ch = channel_names.index("F3")

#         batch_args = {
#             'name_format':"ch"+str(ch)+"_{}",
#             'extension':".tif",
#         } 
#         plot_args = {
#             'plot_layout': (1,1),
#             'plot_overlap': 1,
#             'masks_cmap': 'tab10',
#             'plot_stack_dims': (256, 256), 
#             'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
#             'channels':[ch],
#             'min_outline_length':75,
#         }

#         chans = [ch]
#         for _ch in range(len(channel_names)):
#             if _ch not in chans:
#                 chans.append(_ch)
                
#         CT_F3 = cellSegTrack(
#             path_data,
#             path_save,
#             segmentation_args=segmentation_args,
#             concatenation3D_args=concatenation3D_args,
#             error_correction_args=error_correction_args,
#             plot_args=plot_args,
#             batch_args=batch_args,
#             channels=chans
#         )

#         CT_F3.load()

#         ch = channel_names.index("A12")
#         batch_args = {
#             'name_format':"ch"+str(ch)+"_{}",
#             'extension':".tif",
#         }

#         plot_args = {
#             'plot_layout': (1,1),
#             'plot_overlap': 1,
#             'masks_cmap': 'tab10',
#             'plot_stack_dims': (256, 256), 
#             'plot_centers':[False, False], # [Plot center as a dot, plot label on 3D center]
#             'channels':[ch],
#             'min_outline_length':75,
#         }
        
#         chans = [ch]
#         for _ch in range(len(channel_names)):
#             if _ch not in chans:
#                 chans.append(_ch)

#         CT_A12 = cellSegTrack(
#             path_data,
#             path_save,
#             segmentation_args=segmentation_args,
#             concatenation3D_args=concatenation3D_args,
#             error_correction_args=error_correction_args,
#             plot_args=plot_args,
#             batch_args=batch_args,
#             channels=chans
#         )

#         CT_A12.load()
        
#         labs_to_rem = []
#         for cell in CT_F3.jitcells:
#             center = cell.centers[0]
#             z = int(center[0])
#             if remove_cell[z][current_zid[z]]:
#                 labs_to_rem.append(cell.label)
#             current_zid[z]+=1
            
#         for lab in labs_to_rem:
#             CT_F3._del_cell(lab)  
                
#         CT_F3.update_labels() 
        
#         labs_to_rem = []
#         for cell in CT_A12.jitcells:
#             center = cell.centers[0]
#             z = int(center[0])
#             if remove_cell[z][current_zid[z]]:
#                 labs_to_rem.append(cell.label)
#             current_zid[z]+=1
            
#         for lab in labs_to_rem:
#             CT_A12._del_cell(lab)  
                
#         CT_A12.update_labels() 
