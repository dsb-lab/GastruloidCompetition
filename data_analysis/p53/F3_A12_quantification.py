### LOAD PACKAGE ###
from qlivecell import get_file_name, cellSegTrack, check_or_create_dir, get_file_names

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

import numpy as np
import matplotlib.pyplot as plt

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

files_to_exclude = [
    "n2_F3(150)+WT(150)_72h_emiRFP-p53-mCh-DAPI_(40xSil)_Stack1.tif",
    "n2_F3(150)+WT(150)_72h_emiRFP-p53-mCh-DAPI_(40xSil)_Stack2.tif"
]

CONDS = ["WT", "KO"]

repeats = ["n2", "n3", "n4"]
repeats = ["n4"]

zs = []

all_files = []

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
                
for z in range(zs[0]):
    fig, ax = plt.subplots()
    ax.set_xlabel("mean H2B-mCherry [a.u.]")
    ax.set_ylabel("mean H2B-emiRFP [a.u.]")            
    ax.scatter(F3_all[z], A12_all[z],c=colors[z], edgecolors='none')
    # ax.set_xlim(-5, 20000)
    # ax.set_ylim(-5, 20000)
    # ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.spines[['right', 'top']].set_visible(False)
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
ax.title("Mean with std ribbon")
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
    plt.xlabel('log(emiRFG)')
    plt.ylabel('log(mCherry)')
    plt.tight_layout()

    # Custom legend handles
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="A12",
            markerfacecolor=(0.8, 0.0, 0.8, 0.7), markersize=10),
        Line2D([0], [0], marker="o", color="w", label="F3",
            markerfacecolor=(0.0, 0.8, 0.0, 0.7), markersize=10),
        Line2D([0], [0], marker="o", color="w", label="spillover cells",
            markerfacecolor=(0.2, 0.2, 0.2, 1.0), markersize=10),
    ]
    plt.legend(handles=legend_elements, loc="best", frameon=True)
    plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53/clustering/clusteringloglog_z{}.svg".format(z))
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(data1, data2, c=colors_clustering, edgecolors='k')
    plt.xlabel('emiRFP')
    plt.ylabel('mCherry')
    plt.tight_layout()

    # Custom legend handles
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="A12",
            markerfacecolor=(0.8, 0.0, 0.8, 0.7), markersize=10),
        Line2D([0], [0], marker="o", color="w", label="F3",
            markerfacecolor=(0.0, 0.8, 0.0, 0.7), markersize=10),
        Line2D([0], [0], marker="o", color="w", label="spillover cells",
            markerfacecolor=(0.2, 0.2, 0.2, 1.0), markersize=10),
    ]
    plt.legend(handles=legend_elements, loc="best", frameon=True)
    plt.savefig("/home/pablo/Desktop/PhD/projects/GastruloidCompetition/results/p53/clustering/clustering_z{}.svg".format(z))
    plt.show()

# Now is time for the removal
current_zid = [0 for z in range(10)]
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
            
            labs_to_rem = []
            for cell in CT_F3.jitcells:
                center = cell.centers[0]
                z = int(center[0])
                if remove_cell[z][current_zid[z]]:
                    labs_to_rem.append(cell.label)
                current_zid[z]+=1
            
            for lab in labs_to_rem:
                CT_F3._del_cell(lab)  
                
            CT_F3.update_labels() 
        
            labs_to_rem = []
            for cell in CT_A12.jitcells:
                center = cell.centers[0]
                z = int(center[0])
                if remove_cell[z][current_zid[z]]:
                    labs_to_rem.append(cell.label)
                current_zid[z]+=1
            
            for lab in labs_to_rem:
                CT_A12._del_cell(lab)  
                
            CT_A12.update_labels() 
