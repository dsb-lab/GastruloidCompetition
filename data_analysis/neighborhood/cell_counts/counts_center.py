### LOAD PACKAGE ###
from embdevtools import get_file_name, CellTracking, save_4Dstack, save_4Dstack_labels, norm_stack_per_z, compute_labels_stack, get_file_names, construct_RGB, extract_fluoro, correct_drift
import numpy as np
import matplotlib.pyplot as plt

### LOAD STARDIST MODEL ###
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# ### PATH TO YOU DATA FOLDER AND TO YOUR SAVING FOLDER ###
TIMES = ["48hr", "72hr", "96hr"]
CONDS = ["WT", "KO"]
for TIME in TIMES:
    for COND in CONDS:

        path_data_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/stacks/{}/{}/'.format(TIME, COND)
        path_save_dir='/home/pablo/Desktop/PhD/projects/Data/gastruloids/joshi/competition/2023_11_17_Casp3/ctobjects/{}/{}/'.format(TIME, COND)

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

        channel_names = ["F3", "A12", "DAPI", "Casp3", "BF"]
        if "96hr" in path_data_dir:
            channel_names = ["A12", "F3", "Casp3", "BF", "DAPI"]

        F3_counts = []
        A12_counts = []
        Casp3_counts = []

        for f, file in enumerate(files):
            F3_counts.append([])
            A12_counts.append([])
            Casp3_counts.append([])

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
                'blur': None, 
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
            CT_F3 = CellTracking(
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

            CT_A12 = CellTracking(
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

            ### DEFINE ARGUMENTS ###
            segmentation_args={
                'method': 'stardist2D', 
                'model': model, 
                'blur': [5,1], 
                # 'n_tiles': (2,2),
            }
            CT_Casp3 = CellTracking(
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
                
            # CT_F3.plot_tracking()
            # CT_A12.plot_tracking()
            CT_Casp3.plot_tracking()
            
            
            F3_counts[-1].append(len(CT_F3.jitcells))
            A12_counts[-1].append(len(CT_A12.jitcells))
            Casp3_counts[-1].append(len(CT_Casp3.jitcells))
            
            
            import numpy as np

            ### CORRECT DISTRIBUTIONS ###

            F3_dist = []
            for cell in CT_F3.jitcells: 
                for zid, z in enumerate(cell.zs[0]):
                    mask = cell.masks[0][zid]
                    img = CT_F3.hyperstack[0,z, channel_names.index("F3")]
                    F3_dist.append(np.mean(img[mask[:,1], mask[:,0]]))

            A12_dist = []
            for cell in CT_A12.jitcells: 
                for zid, z in enumerate(cell.zs[0]):
                    mask = cell.masks[0][zid]
                    img = CT_A12.hyperstack[0,z, channel_names.index("A12")]
                    A12_dist.append(np.mean(img[mask[:,1], mask[:,0]]))
            
            F3_dist = np.array(F3_dist)
            A12_dist = np.array(A12_dist)
            
            mdiff = np.mean(F3_dist) - np.mean(A12_dist)
            if mdiff > 0:
                A12_dist += mdiff
            else: 
                F3_dist -= mdiff 
                
            # fig, ax = plt.subplots()
            # _ = ax.hist(F3_dist, color=[0.9,0,0.9,0.5], bins=40, density=True, label="F3")
            # _ = ax.hist(A12_dist, color=[0,0.8,0,0.5], bins=40, density=True, label="A12")
            # plt.legend()
            # plt.show()
            
            from scipy.spatial import ConvexHull



            Casp3_F3 = 0
            Casp3_A12 = 0
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

                # idx = np.argmax(casp3)
                zz = np.int64(cell.centers[0][0])
                idx = cell.zs[0].index(zz)
                if f3[idx] > a12[idx]:
                    Casp3_F3+=1
                else:
                    Casp3_A12+=1


            print()
            print(TIME)
            print(COND)
            print("F3", len(CT_F3.jitcells)+Casp3_F3)
            print("A12", len(CT_A12.jitcells)+Casp3_A12)
            
            print("F3 casp3", Casp3_F3)
            print("A12 casp3", Casp3_A12)
            
            print("F3 casp3 %", 100 * (Casp3_F3 / (len(CT_F3.jitcells)+Casp3_F3)))
            print("A12 casp3 %", 100 * (Casp3_A12 / (len(CT_A12.jitcells)+Casp3_A12)))
            print()
            