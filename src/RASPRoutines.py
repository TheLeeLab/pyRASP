# -*- coding: utf-8 -*-
"""
This class contains functions that collect analysis routines for RASP
jsb92, 2024/02/08
"""
import os
import fnmatch
import numpy as np
import pandas as pd
import sys

module_dir = os.path.dirname(__file__)
sys.path.append(module_dir)
import IOFunctions
IO = IOFunctions.IO_Functions()
import AnalysisFunctions
A_F = AnalysisFunctions.Analysis_Functions()


class RASP_Routines():
    def __init__(self, defaultarea=True, defaultd=True, defaultrad=True, defaultflat=True, defaultdfocus=True, defaultintfocus=True, defaultcellparams=True, defaultcameraparams=True):
        """
        Initialises class.
    
        Args:
        - defaultarea (boolean). If True, uses area default for analysis later
        - defaultd (boolean). If True, uses default pixel radius for analysis later
        - defaultrad (boolean). If True, uses radiality default for analysis later
        - defaultflat (boolean). If True, uses flatness default for analysis later
        - defaultdfocus (boolean). If True, uses differential infocus default for analysis later
        - defaultintfocus (boolean). If True, uses integral infocus default for analysis later
        - defaultcameraparams (boolean). If True, uses camera parameters in folder for analysis later
        """
        self = self
        self.defaultfolder = os.path.join(os.path.split(module_dir)[0],
                            'default_analysis_parameters')
        if defaultarea == True:
            if os.path.isfile(os.path.join(self.defaultfolder, 'areathres.json')):
                data = IO.load_json(os.path.join(self.defaultfolder, 'areathres.json'))
                self.areathres = float(data['areathres'])
            else:
                self.areathres = 30.
            
        if defaultd == True:
            if os.path.isfile(os.path.join(self.defaultfolder, 'areathres.json')):
                data = IO.load_json(os.path.join(self.defaultfolder, 'areathres.json'))
                self.d = int(data['d'])
            else:
                self.d = int(2.)
        
        if defaultrad == True:
            if os.path.isfile(os.path.join(self.defaultfolder, 'rad_neg.json')):
                data = IO.load_json(os.path.join(self.defaultfolder, 'rad_neg.json'))
                self.integratedGrad = float(data['integratedGrad'])
            else:
                self.integratedGrad = 0.
                
        if defaultflat == True:
            if os.path.isfile(os.path.join(self.defaultfolder, 'rad_neg.json')):
                data = IO.load_json(os.path.join(self.defaultfolder, 'rad_neg.json'))
                self.flatness = float(data['flatness'])
            else:
                self.flatness = 1.
                
        if defaultdfocus == True:
            if os.path.isfile(os.path.join(self.defaultfolder, 'infocus.json')):
                data = IO.load_json(os.path.join(self.defaultfolder, 'infocus.json'))
                self.focus_score_diff = float(data['focus_score_diff'])
            else:
                self.focus_score_diff = 0.2
                
        if defaultintfocus == True:
            if os.path.isfile(os.path.join(self.defaultfolder, 'infocus.json')):
                data = IO.load_json(os.path.join(self.defaultfolder, 'infocus.json'))
                self.focus_score_int = float(data['focus_score_int'])
            else:
                self.focus_score_int = 390.
                
        if defaultcellparams == True:
            if os.path.isfile(os.path.join(self.defaultfolder, 'default_cell_params.json')):
                data = IO.load_json(os.path.join(self.defaultfolder, 'default_cell_params.json'))
                self.cell_sigma1 = float(data['sigma1'])
                self.cell_sigma2 = float(data['sigma2'])
                self.cell_threshold1 = float(data['threshold1'])
                self.cell_threshold2 = float(data['threshold1'])
            else:
                self.cell_sigma1 = 2.
                self.cell_sigma2 = 40.
                self.cell_threshold1 = 200.
                self.cell_threshold2 = 200.
                
        if defaultcameraparams == True:
            if os.path.isfile(os.path.join(self.defaultfolder, 'camera_params.json')):
                data = IO.load_json(os.path.join(self.defaultfolder, 'camera_params.json'))
                self.QE = float(data['QE'])
            else:
                self.QE = 0.95
                
            if os.path.isfile(os.path.join(self.defaultfolder, 'gain_map.tif')):
                self.gain_map = IO.read_tiff(os.path.join(self.defaultfolder, 'gain_map.tif'))
            else:
                self.gain_map = 1.
                
            if os.path.isfile(os.path.join(self.defaultfolder, 'offset_map.tif')):
                self.offset_map = IO.read_tiff(os.path.join(self.defaultfolder, 'offset_map.tif'))
            else:
                self.offset_map = 0.
            
            if type(self.gain_map) is not float and type(self.offset_map) is not float:
                if self.gain_map.shape != self.offset_map.shape:
                    print("Gain and Offset maps are not the same shapes. Defaulting to default gain (1) and offset (0) parameters.")
                    self.gain_map = 1.
                    self.offset_map = 0.
        return
    
    def get_infocus_planes(self, image, kernel):
        """
        Gets z planes that area in focus from an image stack
    
        Args:
        - image (array). image as numpy array
        - kernel (array). gaussian blur kernel   
        """

        na, na, na, focus_score, na = A_F.calculate_gradient_field(image, kernel)
        z_planes = A_F.infocus_indices(focus_score, self.focus_score_diff)
        return z_planes
        
    def count_spots(self, database, z_planes):
        """
        Counts spots per z plane
    
        Args:
        - database is pandas array of spots
        - z_planes is range of zplanes
        """
        columns = ['z', 'n_spots']
        
        spots_per_plane = np.zeros_like(z_planes)
        for z in enumerate(z_planes):
            spots_per_plane[z[0]] = len(database.z[database.z == z[1]+1])
        n_spots = pd.DataFrame(data=np.vstack([z_planes+1, spots_per_plane]).T, 
                               columns=columns)
        return n_spots
        
    def calibrate_radiality(self, folder, imtype='.tif', gsigma=1.4, rwave=2., accepted_ratio=1):
        """
        Calibrates radility parameters. Given a folder of negative controls,
        analyses them and saves the radiality parameter to the .json file, as
        well as writing it to the current class radiality and flatness values
    
        Args:
        - folder (string). Folder containing negative control tifs
        - imtype (string). Type of images being analysed, default tif
        - gsigma (float). gaussian blurring parameter (default 1.4)
        - rwave (float). Ricker wavelent sigma (default 2.)
        - accepted_ratio (float). Percentage accepted of false positives
        """
        files_list = os.listdir(folder)
        files = np.sort([e for e in files_list if imtype in e])
        
        k1, k2 = A_F.create_kernel(gsigma, rwave) # create image processing kernels
        rdl = [0., 0., 0.]
        thres = 0.05
        for i in np.arange(len(files)):
            file_path = os.path.join(folder, files[i])
            image = IO.read_tiff(file_path)
            if len(image.shape) < 3:
                dl_mask, centroids, radiality = A_F.compute_image_props(image, k1, k2, thres, 10000., self.areathres, rdl, self.d, calib=True)
                if i == 0:
                    r1_neg = radiality[:,0]
                    r2_neg = radiality[:,1]
                else:
                    r1_neg = np.hstack([r1_neg, radiality[:,0]])
                    r2_neg = np.hstack([r2_neg, radiality[:,1]])
            else:
                z_planes = self.get_infocus_planes(image, k1)
                z_planes = np.arange(z_planes[0], z_planes[-1])
                if len(z_planes) != 0: # if there are images we want to analyse
                    dl_mask, centroids, radiality = A_F.compute_image_props(image, k1, k2, thres, 10000., self.areathres, rdl, self.d, z_planes=z_planes, calib=True)
                    for z in enumerate(z_planes):
                        if (i == 0) and (z[0] == 0):
                            r1_neg = radiality[z[1]][:,0]
                            r2_neg = radiality[z[1]][:,1]
                        else:
                            r1_neg = np.hstack([r1_neg, radiality[z[1]][:,0]])
                            r2_neg = np.hstack([r2_neg, radiality[z[1]][:,1]])

        rad_1 = np.percentile(r1_neg, accepted_ratio)
        rad_2 = np.percentile(r2_neg, 100.-accepted_ratio)
        
        def bincalculator(data):
            """ bincalculator function
            # reads in data and generates bins according to Freedman-Diaconis rule
            # ================INPUTS============= 
            # data is data to be histogrammed
            # ================OUTPUT============= 
            # bins """
            N = len(data)
            sigma = np.std(data)
        
            binwidth = np.multiply(np.multiply(np.power(N, np.divide(-1,3)), sigma), 3.5)
            bins = np.linspace(np.min(data), np.max(data), int((np.max(data) - np.min(data))/binwidth)+1)
            return bins
        
        to_save = {'flatness' : rad_1, 'integratedGrad' : rad_2}
        
        IO.make_directory(self.defaultfolder)
        IO.save_as_json(to_save, os.path.join(self.defaultfolder, 'rad_neg.json'))
        self.flatness = rad_1
        self.integratedGrad = rad_2
        print("Radiality calibrated using negative control"+
              " images in "+str(folder)+". New flatness is "+
              str(np.around(rad_1, 2))+" and new integrated gradient is "
              +str(np.around(rad_2, 2))+". Parameters saved in "
              +str(self.defaultfolder)+".")
        
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2)
        axs[0].hist(r1_neg, bins=bincalculator(r1_neg), color='#808080', density=True);
        ylim0, ylim1 = axs[0].get_ylim()[0], axs[0].get_ylim()[1]
        axs[0].vlines(rad_1, ylim0, ylim1, color='k', label='threshold', ls='--')
        axs[0].set_ylim([ylim0, ylim1])
        axs[0].set_xlabel('flatness metric')
        axs[0].set_ylabel('probability density') 
        axs[0].grid(True,which="both",ls="--",c='gray', lw=0.25, alpha=0.25)  
        axs[0].legend(loc='best', frameon=False)
               
        axs[1].hist(r2_neg, bins=bincalculator(r2_neg), color='#808080', density=True);
        ylim0, ylim1 = axs[1].get_ylim()[0], axs[1].get_ylim()[1]
        axs[1].vlines(rad_2, ylim0, ylim1, color='k', label='threshold', ls='--')
        axs[1].set_xlabel('integrated gradient metric')
        axs[1].set_xlim([0, np.max(r2_neg)])
        axs[1].set_ylim([ylim0, ylim1])
        axs[1].legend(loc='best', frameon=False)
        axs[1].grid(True,which="both",ls="--",c='gray', lw=0.25, alpha=0.25)  
        plt.tight_layout()
        plt.show(block=False)

        return
    
    def calibrate_area(self, folder, imtype='.tif', gsigma=1.4, rwave=2., large_thres=10000.):
        """
        Calibrates area threshold. Given a folder of bead images,
        analyses them and saves the radiality parameter to the .json file, as
        well as writing it to the current class radiality and flatness values
    
        Args:
        - folder (string). Folder containing bead (bright) control tifs
        - imtype (string). Type of images being analysed, default tif
        - gisgma (float). gaussian blurring parameter (default 1.4)
        - rwave (float). Ricker wavelent sigma (default 2.)
        """
        files_list = os.listdir(folder)
        files = np.sort([e for e in files_list if imtype in e])
        
        k1, k2 = A_F.create_kernel(gsigma, rwave) # create image processing kernels
        accepted_ratio = 95.; # perc. of CDF we'll use
        areathres = 1000. # arbitrarily high area threshold for this calibration
        thres = 0.05 # threshold is 0.05
        rdl = [self.flatness, self.integratedGrad, 0.]
        for i in np.arange(len(files)):
            file_path = os.path.join(folder, files[i])
            image = IO.read_tiff(file_path)
            if len(image.shape) < 3:
                dl_mask, centroids, radiality = A_F.compute_image_props(image, 
                            k1, k2, thres, large_thres, areathres, rdl, self.d)
                pixel_index_list, areas, centroids = A_F.calculate_region_properties(dl_mask)
                if i == 0:
                    a_neg = areas
                else:
                    a_neg = np.hstack([a_neg, areas])
            else:
                dl_mask, centroids, radiality = A_F.compute_image_props(image, 
                    k1, k2, thres, large_thres, areathres, rdl, self.d, z_planes=np.arange(image.shape[2]))
                for j in np.arange(image.shape[2]):
                    pixel_index_list, areas, centroids = A_F.calculate_region_properties(dl_mask[:,:,j])
                    if (i == 0) and (j == 0):
                        a_neg = areas
                    else:
                        a_neg = np.hstack([a_neg, areas])

        area_thresh = int(np.ceil(np.percentile(a_neg, accepted_ratio)))
                
        to_save = {'areathres' : area_thresh, 'd': self.d}
        
        IO.make_directory(self.defaultfolder)
        IO.save_as_json(to_save, os.path.join(self.defaultfolder,
                                              'areathres.json'))
        self.areathres = area_thresh
        self.d = self.d
        
        print("Area threshold using beads"+
              " images in "+str(folder)+". New area threshold is "+
              str(np.around(area_thresh, 2))+
              " new radiality radius calibrated and is "+
              str(int(self.d))+
              ". Parameters saved in "
              +str(self.defaultfolder)+".")
        
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1)
        axs.ecdf(a_neg, color='k');
        xlim0, xlim1 = axs.get_xlim()[0], axs.get_xlim()[1]
        axs.hlines(accepted_ratio/100., xlim0, xlim1, color='k', label='threshold', ls='--')
        axs.set_ylim([0, 1])
        axs.set_xlim([xlim0, xlim1])
        axs.set_xlabel('puncta area (pixel)')
        axs.set_ylabel('probability ') 
        axs.grid(True,which="both",ls="--",c='gray', lw=0.25, alpha=0.25)  
        axs.legend(loc='lower right', frameon=False)
        plt.tight_layout()
        plt.show(block=False)

        return
    
    def analyse_images(self, folder, imtype='.tif', thres=0.05, 
                       large_thres=450., gsigma=1.4, rwave=2.,
                       oligomer_string='C1', cell_string='C0',
                       if_filter=True, im_start=1, cell_analysis=True, 
                       one_savefile=False, disp=True):
        """
        analyses data from images in a specified folder,
        saves spots, locations, intensities and backgrounds in a folder created
        next to the folder analysed with _analysis string attached
        also writes a folder with _analysisparameters and saves analysis parameters
        used for particular experiment
    
        Args:
        - folder (string). Folder containing images
        - imtype (string). Type of images being analysed, default tif
        - gisgma (float). gaussian blurring parameter (default 1.4)
        - rwave (float). Ricker wavelent sigma (default 2.)
        - oligomer_string (string). string for oligomer-containing data (default C1)
        - cell string (string). string for cell-containing data (default C0)
        - if_filter (boolean). Filter images for focus (default True)
        - im_start (integer). Images to start from (default 1)
        - cell_analysis (boolean). Parameter where script also analyses cell
        images and computes colocalisation likelihood ratios.
        - one_savefile (boolean). Parameter that, if true, doesn't save a file
        per image but amalgamates them into one file
        - disp (boolean). If true, prints when analysed an image stack.

        """
        files = self.file_search(folder, oligomer_string, imtype)
        if cell_analysis == True:
            cell_files = self.file_search(folder, cell_string, imtype)

        k1, k2 = A_F.create_kernel(gsigma, rwave) # create image processing kernels
        rdl = [self.flatness, self.integratedGrad, 0.]
        
        # create analysis and analysis parameter directories
        analysis_directory = os.path.abspath(folder)+'_analysis'
        IO.make_directory(analysis_directory)
        analysis_p_directory = os.path.abspath(folder)+'_analysisparameters'

        to_save = {'areathres': self.areathres, 'flatness': self.flatness, 'd': self.d,
                   'integratedGrad': self.integratedGrad, 'gaussian_sigma':
                       gsigma, 'ricker_sigma': rwave, 'thres': thres,
                       'large_thres': large_thres, 
                       'focus_score_diff': self.focus_score_diff,
                       'cell_sigma1': self.cell_sigma1,
                       'cell_sigma2': self.cell_sigma2,
                       'cell_threshold1': self.cell_threshold1,
                       'cell_threshold2': self.cell_threshold2,
                       'QE': self.QE}
        IO.save_analysis_params(analysis_p_directory, 
                to_save, gain_map=self.gain_map, offset_map=self.offset_map)
        
        for i in np.arange(len(files)):
            img = IO.read_tiff_tophotons(os.path.join(folder, files[i]), 
            QE=self.QE, gain_map=self.gain_map, offset_map=self.offset_map)
            if cell_analysis == True:
                img_cell = IO.read_tiff_tophotons(os.path.join(folder, cell_files[i]), 
                QE=self.QE, gain_map=self.gain_map, offset_map=self.offset_map)
            if im_start > 1:
                img = img[:, :, im_start:]
            if len(img.shape) > 2: # if a z-stack
                z_planes = self.get_infocus_planes(img, k1)
                
                if cell_analysis == False:
                    to_save = A_F.compute_spot_props(img, 
                    k1, k2, thres=thres, large_thres=large_thres, 
                    areathres=self.areathres, rdl=rdl, z=z_planes, d=self.d)
                else:
                    to_save, to_save_cell, cell_mask = A_F.compute_spot_and_cell_props(img, img_cell, k1, k2,
                                        prot_thres=thres, large_prot_thres=large_thres, 
                                        areathres=self.areathres, rdl=rdl, z=z_planes, 
                                        cell_threshold1=self.cell_threshold1, 
                                        cell_threshold2=self.cell_threshold1, 
                                        cell_sigma1=self.cell_sigma1,
                                        cell_sigma2=self.cell_sigma2,
                                        d=self.d)
                
                if one_savefile == False:
                    savename = os.path.join(analysis_directory, 
                    files[i].split(imtype)[0]+'.csv')
                    to_save.to_csv(savename, index=False)
                    if cell_analysis == True:
                        to_save_cell.to_csv(os.path.join(analysis_directory, 
                        files[i].split(imtype)[0]+'_cell_analysis.csv'), index=False)
                        IO.write_tiff(cell_mask, os.path.join(analysis_directory, 
                        files[i].split(imtype)[0]+'_cellMask.tiff'), bit=np.uint8)
                else:
                    to_save['image_filename'] = np.full_like(to_save.z.values, files[i], dtype='object')
                    savename = os.path.join(analysis_directory, 'spot_analysis.csv')
                    savename_spot = os.path.join(analysis_directory, 'spot_numbers.csv')
                    n_spots = self.count_spots(to_save, np.arange(z_planes[0], z_planes[1]))
                    n_spots['image_filename'] = np.full_like(n_spots.z.values, files[i], dtype='object')

                    if cell_analysis == True:
                        to_save_cell['image_filename'] = np.full_like(to_save_cell.z.values, files[i], dtype='object')
                        savename_cell = os.path.join(analysis_directory, 'cell_colocalisation_analysis.csv')
                        IO.write_tiff(cell_mask, os.path.join(analysis_directory, 
                        files[i].split(imtype)[0]+'_cellMask.tiff'), bit=np.uint8)

                    if i != 0:
                        to_save.to_csv(savename, mode='a', header=False, index=False)
                        n_spots.to_csv(savename_spot, mode='a', header=False, index=False)
                        if cell_analysis == True:
                            to_save_cell.to_csv(savename_cell, mode='a', header=False, index=False)
                    else:
                        to_save.to_csv(savename, index=False)
                        n_spots.to_csv(savename_spot, index=False)
                        if cell_analysis == True:
                            to_save_cell.to_csv(savename_cell, index=False)
            else: # if not a z-stack
                to_save = A_F.compute_spot_props(img, k1, k2, thres=thres,
                large_thres=large_thres, areathres=self.areathres, rdl=rdl, d=self.d)
                
                if one_savefile == False:
                    savename = os.path.join(analysis_directory, 
                        files[i].split(imtype)[0]+'.csv')
                    to_save.to_csv(savename, index=False)
                else:
                    to_save['image_filename'] = np.full_like(to_save.z.values, files[i], dtype='object')
                    savename = os.path.join(analysis_directory, 'spot_analysis.csv')
                    if i != 0:
                        to_save.to_csv(savename, mode='a', header=False, index=False)
                    else:
                        to_save.to_csv(savename, index=False)
            if disp == True:
                print("Analysed image", files[i], "data saved in", savename)
        return
    
    def save_analysis_results(self, directory, file, to_save, 
                    rsid, cell_analysis=False, to_save_cell=0, cell_mask=0):
        """
        Saves analysis results to the specified directory.
        
        Args:
        - directory (str): The directory where the results will be saved.
        - file_path (str): The file path of the original data file.
        - data_to_save (pandas.DataFrame): The data to be saved.
        - rsid (float): The rsid value associated with the data.
        - cell_analysis (bool): Indicates whether cell analysis was performed (default False).
        - cell_data_to_save (pandas.DataFrame): The cell data to be saved (default None).
        - cell_mask (numpy.ndarray): The cell mask to be saved as a TIFF file (default None).
        """
        IO.make_directory(directory)
        savefile = os.path.split(file)[-1]
        to_save['rsid'] = np.full_like(to_save.z.values, rsid)
        to_save.to_csv(os.path.join(directory, 
        savefile+'.csv'), index=False)
        if cell_analysis == True:
            to_save_cell['rsid'] = np.full_like(to_save_cell.z.values, rsid)
            to_save_cell.to_csv(os.path.join(directory, 
            savefile+'_cell_analysis.csv'), index=False)
            IO.write_tiff(cell_mask, os.path.join(directory, 
            savefile+'_cellMask.tiff'), bit=np.uint8)
        return
    
    def save_analysis_results_onesavefile(self, analysis_directory, file, 
                to_save, rsid, z_planes, i, cell_analysis=False, cell_file=0, to_save_cell=0, cell_mask=0):
        """
        Saves analysis results to the specified directory.
        
        Args:
        - directory (str): The directory where the results will be saved.
        - file_path (str): The file path of the original data file.
        - data_to_save (pandas.DataFrame): The data to be saved.
        - rsid (float): The rsid value associated with the data.
        - cell_analysis (bool): Indicates whether cell analysis was performed (default False).
        - cell_data_to_save (pandas.DataFrame): The cell data to be saved (default None).
        - cell_mask (numpy.ndarray): The cell mask to be saved as a TIFF file (default None).
        """
        to_save['rsid'] = np.full_like(to_save.z.values, rsid)
        to_save['image_filename'] = np.full_like(to_save.z.values, os.path.split(file)[-1], dtype='object')
        savename = os.path.join(analysis_directory, 'spot_analysis.csv')
        savename_spot = os.path.join(analysis_directory, 'spot_numbers.csv')
        if cell_analysis == True:
            savename_cell = os.path.join(analysis_directory, 'cell_colocalisation_analysis.csv')
            to_save_cell['rsid'] = np.full_like(to_save_cell.z.values, rsid)
            to_save_cell['image_filename'] = np.full_like(to_save_cell.z.values, os.path.split(file)[-1], dtype='object')
            savefile = os.path.split(cell_file)[-1]
            directory = os.path.split(os.path.split(file)[0])[0]+'_cellMasks'
            IO.make_directory(directory)
            IO.write_tiff(cell_mask, os.path.join(directory, 
                                                  savefile+'_cellMask.tiff'), bit=np.uint8)
            
        n_spots = self.count_spots(to_save, np.arange(z_planes[0], z_planes[1]))
        n_spots['rsid'] = np.full_like(n_spots.z.values, rsid)
        n_spots['image_filename'] = np.full_like(n_spots.z.values, os.path.split(file)[-1], dtype='object')

        if i != 0:
            to_save.to_csv(savename, mode='a', header=False, index=False)
            n_spots.to_csv(savename_spot, mode='a', header=False, index=False)
            if cell_analysis == True:
                to_save_cell.to_csv(savename_cell, mode='a', header=False, index=False)
        else:
            to_save.to_csv(savename, index=False)
            n_spots.to_csv(savename_spot, index=False)
            if cell_analysis == True:
                to_save_cell.to_csv(savename_cell, index=False)
        return
    
    def file_search(self, folder, string1, string2):
        """
        Search for files containing 'string1' in their names within 'folder',
        and then filter the results to include only those containing 'string2'.

        Args:
        - folder (str): The directory to search for files.
        - string1 (str): The first string to search for in the filenames.
        - string2 (str): The second string to filter the filenames containing string1.

        Returns:
        - file_list (list): A sorted list of file paths matching the search criteria.
        """
        # Get a list of all files containing 'string1' in their names within 'folder'
        file_list = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(folder)
            for f in fnmatch.filter(files, '*'+string1+'*')]
        file_list = np.sort([e for e in file_list if string2 in e])
        return file_list
    
    def analyse_round_subfolder(self, folder, k1, k2, rdl, imtype='.tif', thres=0.05, 
                             large_thres=450., gsigma=1.4, rwave=2., 
                             oligomer_string='C1', cell_string='C0',
                             if_filter=True, im_start=1, cell_analysis=False, 
                             one_savefile=True, disp=True):
        """
        analyses data in a folder specified,
        folder has either "Round" in the title
        or multiple rounds below; 
        structure as in Lee Lab Cambridge Experiment
        saves spots, locations, intensities and backgrounds in a folder created
        next to the folder analysed with _analysis string attached
        also writes a folder with _analysisparameters and saves analysis parameters
        used for particular experiment
    
        Args:
        - folder (string). Folder containing images
        - k1 (matrix). convolution kernel 1
        - k2 (matrix). convolution kernel 2
        - rdl (vector). radiality filter
        - imtype (string). Type of images being analysed, default tif
        - gisgma (float). gaussian blurring parameter (default 1.4)
        - rwave (float). Ricker wavelent sigma (default 2.)
        - oligomer_string (string). string for oligomer-containing data (default C1)
        - cell string (string). string for cell-containing data (default C0)
        - if_filter (boolean). Filter images for focus (default True)
        - im_start (integer). Images to start from (default 1)
        - one_savefile (boolean). Parameter that, if true, doesn't save a file
        per image but amalgamates them into one file
        - cell_analysis (boolean). Parameter where script also analyses cell
        images and computes colocalisation likelihood ratios.
        - disp (boolean). If True, outputs a message saying analysed image.
        """

        r = float(os.path.split(folder)[1].split('Round')[1]) # get round for rsid later
        
        # create analysis parameter directory
        analysis_p_directory = os.path.abspath(folder)+'_analysisparameters'

        to_save = {'areathres': self.areathres, 'flatness': self.flatness,
                   'd': self.d, 'integratedGrad': self.integratedGrad,
                   'gaussian_sigma': gsigma, 'ricker_sigma': rwave, 
                   'thres': thres, 'large_thres': large_thres, 
                       'focus_score_diff': self.focus_score_diff,
                       'cell_sigma1': self.cell_sigma1,
                       'cell_sigma2': self.cell_sigma2,
                       'cell_threshold1': self.cell_threshold1,
                       'cell_threshold2': self.cell_threshold2,
                       'QE': self.QE}
        IO.save_analysis_params(analysis_p_directory, 
                to_save, gain_map=self.gain_map, offset_map=self.offset_map)
       
        oligomer_files = self.file_search(folder, oligomer_string, imtype)
        if cell_analysis == True:
            cell_files = self.file_search(folder, cell_string, imtype)
    
        if one_savefile == True:
            analysis_directory = os.path.abspath(folder)+'_analysis'
            IO.make_directory(analysis_directory)
        
        for i in np.arange(len(oligomer_files)):
            if 'Sample' in oligomer_files[i]:
                s = float(os.path.split(os.path.split(os.path.split(oligomer_files[i])[0])[0])[1].split('Sample')[1])/100
            else:
                s = float(os.path.split(os.path.split(os.path.split(oligomer_files[i])[0])[0])[1].split('S')[1])/100
            rsid = r + s
            img = IO.read_tiff_tophotons(oligomer_files[i], 
            QE=self.QE, gain_map=self.gain_map, offset_map=self.offset_map)
            if cell_analysis == True:
                img_cell = IO.read_tiff_tophotons(cell_files[i], 
                QE=self.QE, gain_map=self.gain_map, offset_map=self.offset_map)

            if len(img.shape) > 2: # if a z-stack
                z_planes = self.get_infocus_planes(img, k1)
                
                if cell_analysis == False:
                    to_save = A_F.compute_spot_props(img, 
                    k1, k2, thres=thres, large_thres=large_thres, 
                    areathres=self.areathres, rdl=rdl, z=z_planes, d=self.d)
                else:
                    to_save, to_save_cell, cell_mask = A_F.compute_spot_and_cell_props(img, img_cell, k1, k2,
                                        prot_thres=thres, large_prot_thres=large_thres, 
                                        areathres=self.areathres, rdl=rdl, z=z_planes, 
                                        cell_threshold1=self.cell_threshold1, 
                                        cell_threshold2=self.cell_threshold1, 
                                        cell_sigma1=self.cell_sigma1,
                                        cell_sigma2=self.cell_sigma2,
                                        d=self.d)
                
                file = oligomer_files[i]
                directory = os.path.split(os.path.split(file)[0])[0]+'_analysis'
                

                if one_savefile == False:
                    if cell_analysis == True:
                        self.save_analysis_results(self, directory, 
                        file, to_save, rsid, cell_analysis, to_save_cell, cell_mask)
                    else:
                        self.save_analysis_results(self, directory, 
                        file, to_save, rsid, cell_analysis)                    
                else:
                    if cell_analysis == True:
                        self.save_analysis_results_onesavefile(analysis_directory, file, 
                            to_save, rsid, z_planes, i, cell_analysis, cell_files[i], to_save_cell, cell_mask)
                    else:
                        self.save_analysis_results_onesavefile(analysis_directory, file, 
                            to_save, rsid, z_planes, i, False)
            else: # if not a z-stack
                to_save = A_F.compute_spot_props(img, k1, k2, thres=thres,
                large_thres=large_thres, areathres=self.areathres, rdl=rdl,
                d=self.d)
                
                if one_savefile == False:
                    directory = os.path.split(os.path.split(oligomer_files[i])[0])+'_analysis'
                    IO.make_directory(directory)
                    to_save['rsid'] = np.full_like(to_save.z.values, rsid)
                    to_save['image_filename'] = np.full_like(to_save.z.values, os.path.split(oligomer_files[i])[-1], dtype='object')
                    savefile = os.path.split(oligomer_files[i])[-1]
                    savename = os.path.join(directory, 
                    savefile+'.csv')
                    to_save.to_csv(savename, index=False)
                else:
                    to_save['rsid'] = np.full_like(to_save.z.values, rsid)
                    savename = os.path.join(analysis_directory, 'spot_analysis.csv')
                    if i != 0:
                        to_save.to_csv(savename, mode='a', header=False, index=False)
                    else:
                        to_save.to_csv(savename, index=False)
            if disp == True:
                print("Analysed image", oligomer_files[i], "data saved in", analysis_directory)

        return
    
    def analyse_round_images(self, folder, imtype='.tif', thres=0.05, 
                             large_thres=450., gsigma=1.4, rwave=2., 
                             oligomer_string='C1', cell_string='C0',
                             if_filter=True, im_start=1, cell_analysis=False, one_savefile=True):
        """
        analyses data in a folder specified,
        folder has either "Round" in the title
        or multiple rounds below; 
        structure as in Lee Lab Cambridge Experiment
        saves spots, locations, intensities and backgrounds in a folder created
        next to the folder analysed with _analysis string attached
        also writes a folder with _analysisparameters and saves analysis parameters
        used for particular experiment
    
        Args:
        - folder (string). Folder containing images
        - imtype (string). Type of images being analysed, default tif
        - gisgma (float). gaussian blurring parameter (default 1.4)
        - rwave (float). Ricker wavelent sigma (default 2.)
        - oligomer_string (string). string for oligomer-containing data (default C1)
        - cell string (string). string for cell-containing data (default C0)
        - if_filter (boolean). Filter images for focus (default True)
        - im_start (integer). Images to start from (default 1)
        - one_savefile (boolean). Parameter that, if true, doesn't save a file
        per image but amalgamates them into one file
        - cell_analysis (boolean). Parameter where script also analyses cell
        images and computes colocalisation likelihood ratios.
        """
        k1, k2 = A_F.create_kernel(gsigma, rwave) # create image processing kernels
        rdl = [self.flatness, self.integratedGrad, 0.]

        if 'Round' in folder:
            self.analyse_round_subfolder(folder, k1, k2, rdl, imtype, thres, 
                                     large_thres, gsigma, rwave, 
                                     oligomer_string, cell_string,
                                     if_filter, im_start, cell_analysis, one_savefile)
        else:
            folders = np.sort([e for e in os.listdir(folder) if 'Round' in e])
            folders = np.sort([e for e in folders if 2 > len(e.split('_'))]) # remove any folders that aren't just roundN
            if len(folders) > 0:
                for f in folders:
                    self.analyse_round_subfolder(os.path.join(folder, f), k1, k2, rdl, imtype, thres, 
                                             large_thres, gsigma, rwave, 
                                             oligomer_string, cell_string,
                                             if_filter, im_start, 
                                             cell_analysis, one_savefile)
        return
