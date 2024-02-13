# -*- coding: utf-8 -*-
"""
This class contains functions that collect analysis routines for RASP
jsb92, 2024/02/08
"""
from . import IOFunctions
IO = IOFunctions.IO_Functions()
from . import AnalysisFunctions
A_F = AnalysisFunctions.Analysis_Functions()
import os
import fnmatch
import numpy as np
import pandas as pd

class RASP_Routines():
    def __init__(self, defaultarea=True, defaultrad=True, defaultsteep=True, defaultdfocus=True, defaultintfocus=True, defaultcameraparams=True):
        """
        Initialises class.
    
        Args:
        - defaultarea (boolean). If True, uses area default for analysis later
        - defaultrad (boolean). If True, uses radiality default for analysis later
        - defaultsteep (boolean). If True, uses steepness default for analysis later
        - defaultdfocus (boolean). If True, uses differential infocus default for analysis later
        - defaultintfocus (boolean). If True, uses integral infocus default for analysis later
        - defaultcameraparams (boolean). If True, uses camera parameters in folder for analysis later
        """

        self = self
        self.defaultfolder = 'default_analysis_parameters'
        if defaultarea == True:
            if os.path.isfile(os.path.join(self.defaultfolder, 'areathres.json')):
                data = IO.load_json(os.path.join(self.defaultfolder, 'areathres.json'))
                self.areathres = float(data['areathres'])
            else:
                self.areathres = 30.
        
        if defaultrad == True:
            if os.path.isfile(os.path.join(self.defaultfolder, 'rad_neg.json')):
                data = IO.load_json(os.path.join(self.defaultfolder, 'rad_neg.json'))
                self.integratedGrad = float(data['integratedGrad'])
            else:
                self.integratedGrad = 0.
                
        if defaultsteep == True:
            if os.path.isfile(os.path.join(self.defaultfolder, 'rad_neg.json')):
                data = IO.load_json(os.path.join(self.defaultfolder, 'rad_neg.json'))
                self.steepness = float(data['steepness'])
            else:
                self.steepness = 1.
                
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
    
    def compute_image_props(self, image, k1, k2, thres=0.05, large_thres=450., areathres=30., rdl=[50., 0., 0.]):
        """
        Gets basic image properties (dl_mask, centroids, radiality)
        from a single image
    
        Args:
        - image (array). image as numpy array
        - k1 (array). gaussian blur kernel
        - k2 (array). ricker wavelet kernel
        - thres (float). percentage threshold
        - areathres (float). area threshold
        - rdl (array). radiality thresholds
   
        """
        large_mask = A_F.detect_large_features(image, large_thres)
        img2, Gx, Gy, focusScore, cfactor = A_F.calculate_gradient_field(image, k1)
        dl_mask, centroids, radiality, idxs = A_F.small_feature_kernel(image, 
        large_mask, img2, Gx, Gy,
        k2, thres, areathres, rdl)
        
        return dl_mask, centroids, radiality
    
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
        
    def compute_spot_props(self, image, k1, k2, thres=0.05, large_thres=450., 
                           areathres=30., rdl=[50., 0., 0.], z=[0]):
        """
        Gets basic image properties (dl_mask, centroids, radiality)
        from a single image
    
        Args:
        - image (array). image as numpy array
        - k1 (array). gaussian blur kernel
        - k2 (array). ricker wavelet kernel
        - thres (float). percentage threshold
        - areathres (float). area threshold
        - rdl (array). radiality thresholds
        - z (array). z planes to image, default 0
        """
        
        columns = ['x', 'y', 'z', 'sum_intensity_in_photons', 'bg', 'zi', 'zf']
        if len(z) > 1:
            z_planes = np.arange(z[0], z[1])
            for zp in z_planes:
                img_z = image[:, :, zp]
                large_mask = A_F.detect_large_features(img_z, large_thres)
                img2, Gx, Gy, focusScore, cfactor = A_F.calculate_gradient_field(img_z, k1)
                dl_mask, centroids, radiality, idxs = A_F.small_feature_kernel(img_z, 
                large_mask, img2, Gx, Gy,
                k2, thres, areathres, rdl)
                estimated_intensity, estimated_background = A_F.estimate_intensity(img_z, centroids)
                to_keep = ~np.isnan(estimated_intensity)
                estimated_intensity = estimated_intensity[to_keep]
                estimated_background = estimated_background[to_keep]
                centroids = centroids[to_keep, :]    
                dataarray = np.vstack([centroids[:, 0], centroids[:, 1], 
                np.full_like(centroids[:, 0], zp+1), estimated_intensity, 
                estimated_background, np.full_like(centroids[:, 0], 1+z_planes[0]),
                np.full_like(centroids[:, 0], 1+z_planes[-1])])                
                if zp == z_planes[0]:
                    to_save = pd.DataFrame(data=dataarray.T, columns=columns)
                else:
                    to_save = pd.concat([to_save, pd.DataFrame(data=dataarray.T, columns=columns)])
            to_save = to_save.reset_index(drop=True)
        else:
            large_mask = A_F.detect_large_features(image, large_thres)
            img2, Gx, Gy, focusScore, cfactor = A_F.calculate_gradient_field(image, k1)
            dl_mask, centroids, radiality, idxs = A_F.small_feature_kernel(image, 
            large_mask, img2, Gx, Gy,
            k2, thres, areathres, rdl)
            estimated_intensity, estimated_background = A_F.estimate_intensity(image, centroids)
            to_keep = ~np.isnan(estimated_intensity)
            estimated_intensity = estimated_intensity[to_keep]
            estimated_background = estimated_background[to_keep]
            centroids = centroids[to_keep, :]
            dataarray = np.vstack([centroids[:, 0], centroids[:, 1], 
            np.full_like(centroids[:, 0], 1), estimated_intensity, 
            estimated_background, np.full_like(centroids[:, 0], 1),
            np.full_like(centroids[:, 0], 1)])
            to_save = pd.DataFrame(data=dataarray.T, columns=columns)
            
        return to_save

    def calibrate_radiality(self, folder, imtype='.tif', gsigma=1.4, rwave=2., large_thres=10000.):
        """
        Calibrates radility parameters. Given a folder of negative controls,
        analyses them and saves the radiality parameter to the .json file, as
        well as writing it to the current class radiality and steepness values
    
        Args:
        - folder (string). Folder containing negative control tifs
        - imtype (string). Type of images being analysed, default tif
        - gsigma (float). gaussian blurring parameter (default 1.4)
        - rwave (float). Ricker wavelent sigma (default 2.)
        - large_thres (float). threshold for large objects.
        """
        files_list = os.listdir(folder)
        files = np.sort([e for e in files_list if imtype in e])
        
        k1, k2 = A_F.create_kernel(gsigma, rwave) # create image processing kernels
        accepted_ratio = 0.2; # perc. false positives accepted per dimension
        rdl = [1000., 0., 0.]
        thres = 0.05
        for i in np.arange(len(files)):
            file_path = os.path.join(folder, files[i])
            image = IO.read_tiff(file_path)
            if len(image.shape) < 3:
                dl_mask, centroids, radiality = self.compute_image_props(image, k1, k2, thres, large_thres, self.areathres, rdl)
                if i == 0:
                    r1_neg = radiality[:,0]
                    r2_neg = radiality[:,1]
                else:
                    r1_neg = np.hstack([r1_neg, radiality[:,0]])
                    r2_neg = np.hstack([r1_neg, radiality[:,1]])
            else:
                for j in np.arange(image.shape[2]):
                    dl_mask, centroids, radiality = self.compute_image_props(image[:,:,j], k1, k2, thres, large_thres, self.areathres, rdl)
                    if (i == 0) and (j == 0):
                        r1_neg = radiality[:,0]
                        r2_neg = radiality[:,1]
                    else:
                        r1_neg = np.hstack([r1_neg, radiality[:,0]])
                        r2_neg = np.hstack([r1_neg, radiality[:,1]])

        rad_1 = np.percentile(r1_neg, accepted_ratio)
        rad_2 = np.percentile(r2_neg, 100.-accepted_ratio)
        
        to_save = {'steepness' : rad_1, 'integratedGrad' : rad_2}
        
        IO.make_directory(self.defaultfolder)
        IO.save_as_json(to_save, os.path.join(self.defaultfolder, 'rad_neg.json'))
        self.steepness = rad_1
        self.integratedGrad = rad_2
        return
    
    def calibrate_area(self, folder, imtype='.tif', gsigma=1.4, rwave=2., large_thres=10000.):
        """
        Calibrates area threshold. Given a folder of bead images,
        analyses them and saves the radiality parameter to the .json file, as
        well as writing it to the current class radiality and steepness values
    
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
        rdl = [self.steepness, self.integratedGrad, 0.]
        for i in np.arange(len(files)):
            file_path = os.path.join(folder, files[i])
            image = IO.read_tiff(file_path)
            if len(image.shape) < 3:
                dl_mask, centroids, radiality = self.compute_image_props(image, 
                            k1, k2, thres, large_thres, areathres, rdl)
                pixel_index_list, areas, centroids = A_F.calculate_region_properties(dl_mask)
                if i == 0:
                    a_neg = areas
                else:
                    a_neg = np.hstack([a_neg, areas])
            else:
                for j in np.arange(image.shape[2]):
                    dl_mask, centroids, radiality = self.compute_image_props(image, 
                            k1, k2, thres, large_thres, areathres, rdl)
                    pixel_index_list, areas, centroids = A_F.calculate_region_properties(dl_mask)
                    if (i == 0) and (j == 0):
                        a_neg = areas
                    else:
                        a_neg = np.hstack([a_neg, areas])

        area_thresh = int(np.ceil(np.percentile(a_neg, accepted_ratio)))
        
        to_save = {'areathres' : area_thresh}
        
        IO.make_directory(self.defaultfolder)
        IO.save_as_json(to_save, os.path.join(self.defaultfolder,
                                              'areathres.json'))
        self.areathres = area_thresh
        return
    
    def analyse_images(self, folder, imtype='.tif', thres=0.05, 
                       large_thres=450., gsigma=1.4, rwave=2.,
                       if_filter=True, im_start=1, one_savefile=False):
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
        - if_filter (boolean). Filter images for focus (default True)
        - im_start (integer). Images to start from (default 1)
        - one_savefile (boolean). Parameter that, if true, doesn't save a file
        per image but amalgamates them into one file

        """
        files_list = os.listdir(folder)
        files = np.sort([e for e in files_list if imtype in e])
        
        k1, k2 = A_F.create_kernel(gsigma, rwave) # create image processing kernels
        rdl = [self.steepness, self.integratedGrad, 0.]
        
        # create analysis and analysis parameter directories
        analysis_directory = os.path.abspath(folder)+'_analysis'
        IO.make_directory(analysis_directory)
        analysis_p_directory = os.path.abspath(folder)+'_analysisparameters'

        to_save = {'areathres': self.areathres, 'steepness': self.steepness, 
                   'integratedGrad': self.integratedGrad, 'gaussian_sigma':
                       gsigma, 'ricker_sigma': rwave, 'thres': thres,
                       'large_thres': large_thres, 
                       'focus_score_diff': self.focus_score_diff,
                       'QE': self.QE}
        IO.save_analysis_params(analysis_p_directory, 
                to_save, gain_map=self.gain_map, offset_map=self.offset_map)
        
        for i in np.arange(len(files)):
            img = IO.read_tiff_tophotons(os.path.join(folder, files[i]), 
            QE=self.QE, gain_map=self.gain_map, offset_map=self.offset_map)
            if im_start > 1:
                img = img[:, :, im_start:]
            if len(img.shape) > 2: # if a z-stack
                z_planes = self.get_infocus_planes(img, k1)
                
                to_save = self.compute_spot_props(img, 
                k1, k2, thres=thres, large_thres=large_thres, 
                areathres=self.areathres, rdl=rdl, z=z_planes)
                
                if one_savefile == False:
                    to_save.to_csv(os.path.join(analysis_directory, 
                    files[i].split(imtype)[0]+'.csv'), index=False)
                else:
                    to_save['image_filename'] = np.full_like(to_save.z.values, files[i], dtype='object')
                    savename = os.path.join(analysis_directory, 'spot_analysis.csv')
                    if i != 0:
                        to_save.to_csv(savename, mode='a', header=False, index=False)
                    else:
                        to_save.to_csv(savename, index=False)
            else: # if not a z-stack
                to_save = self.compute_spot_props(img, k1, k2, thres=thres,
                large_thres=large_thres, areathres=self.areathres, rdl=rdl)
                
                if one_savefile == False:
                    to_save.to_csv(os.path.join(analysis_directory, 
                        files[i].split(imtype)[0]+'.csv'), index=False)
                else:
                    to_save['image_filename'] = np.full_like(to_save.z.values, files[i], dtype='object')
                    savename = os.path.join(analysis_directory, 'spot_analysis.csv')
                    if i != 0:
                        to_save.to_csv(savename, mode='a', header=False, index=False)
                    else:
                        to_save.to_csv(savename, index=False)
        return
    
    def analyse_round_images(self, folder, imtype='.tif', thres=0.05, 
                             large_thres=450., gsigma=1.4, rwave=2., 
                             oligomer_string='C1', cell_string='C0',
                             if_filter=True, im_start=1, one_savefile=True):
        """
        analyses data in a folder specified,
        folder has "RoundN/SN" structure as in Lee Lab Cambridge Experiment
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

        """
        r = float(os.path.split(folder)[1].split('Round')[1]) # get round for rsid later
        
        k1, k2 = A_F.create_kernel(gsigma, rwave) # create image processing kernels
        rdl = [self.steepness, self.integratedGrad, 0.]
        
        # create analysis parameter directory
        analysis_p_directory = os.path.abspath(folder)+'_analysisparameters'

        to_save = {'areathres': self.areathres, 'steepness': self.steepness, 
                   'integratedGrad': self.integratedGrad, 'gaussian_sigma':
                       gsigma, 'ricker_sigma': rwave, 'thres': thres,
                       'large_thres': large_thres, 
                       'focus_score_diff': self.focus_score_diff,
                       'QE': self.QE}
        IO.save_analysis_params(analysis_p_directory, 
                to_save, gain_map=self.gain_map, offset_map=self.offset_map)
       
        oligomer_files = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(folder)
            for f in fnmatch.filter(files, '*'+oligomer_string+'*')]
        oligomer_files = np.sort([e for e in oligomer_files if imtype in e])

        
        analysis_directory = os.path.abspath(folder)+'_analysis'
        IO.make_directory(analysis_directory)
        
        for i in np.arange(len(oligomer_files)):
            s = float(os.path.split(os.path.split(os.path.split(oligomer_files[i])[0])[0])[1].split('S')[1])/100
            rsid = r + s
            img = IO.read_tiff_tophotons(oligomer_files[i], 
            QE=self.QE, gain_map=self.gain_map, offset_map=self.offset_map)
            if len(img.shape) > 2: # if a z-stack
                z_planes = self.get_infocus_planes(img, k1)
                
                to_save = self.compute_spot_props(img, 
                k1, k2, thres=thres, large_thres=large_thres, 
                areathres=self.areathres, rdl=rdl, z=z_planes)
                
                if one_savefile == False:
                    directory = os.path.split(os.path.split(oligomer_files[i])[0])+'_analysis'
                    IO.make_directory(directory)
                    savefile = os.path.split(oligomer_files[i])[-1]
                    to_save['rsid'] = np.full_like(to_save.z.values, rsid)
                    to_save.to_csv(os.path.join(directory, 
                    savefile+'.csv'), index=False)
                else:
                    to_save['rsid'] = np.full_like(to_save.z.values, rsid)
                    to_save['image_filename'] = np.full_like(to_save.z.values, os.path.split(oligomer_files[i])[-1], dtype='object')
                    savename = os.path.join(analysis_directory, 'spot_analysis.csv')
                    savename_spot = os.path.join(analysis_directory, 'spot_numbers.csv')
                    
                    n_spots = self.count_spots(to_save, np.arange(z_planes[0], z_planes[1]))
                    n_spots['rsid'] = np.full_like(n_spots.z.values, rsid)
                    n_spots['image_filename'] = np.full_like(n_spots.z.values, os.path.split(oligomer_files[i])[-1], dtype='object')

                    if i != 0:
                        to_save.to_csv(savename, mode='a', header=False, index=False)
                        n_spots.to_csv(savename_spot, mode='a', header=False, index=False)
                    else:
                        to_save.to_csv(savename, index=False)
                        n_spots.to_csv(savename_spot, index=False)
            else: # if not a z-stack
                to_save = self.compute_spot_props(img, k1, k2, thres=thres,
                large_thres=large_thres, areathres=self.areathres, rdl=rdl)
                
                if one_savefile == False:
                    directory = os.path.split(os.path.split(oligomer_files[i])[0])+'_analysis'
                    IO.make_directory(directory)
                    to_save['rsid'] = np.full_like(to_save.z.values, rsid)
                    to_save['image_filename'] = np.full_like(to_save.z.values, os.path.split(oligomer_files[i])[-1], dtype='object')
                    savefile = os.path.split(oligomer_files[i])[-1]
                    to_save.to_csv(os.path.join(directory, 
                    savefile+'.csv'), index=False)
                else:
                    to_save['rsid'] = np.full_like(to_save.z.values, rsid)
                    savename = os.path.join(analysis_directory, 'spot_analysis.csv')
                    if i != 0:
                        to_save.to_csv(savename, mode='a', header=False, index=False)
                    else:
                        to_save.to_csv(savename, index=False)
        return