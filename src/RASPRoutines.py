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
import numpy as np
import pandas as pd

class RASP_Routines():
    def __init__(self, defaultarea=True, defaultrad=True, defaultsteep=True):
        """
        Initialises class.
    
        Args:
        - defaultarea (boolean). If True, uses area default for analysis later
        - defaultrad (boolean). If True, uses radiality default for analysis later
        - defaultsteep (boolean). If True, uses steepness default for analysis later
    
        """

        self = self
        if defaultarea == True:
            if os.path.isfile(os.path.join('analysis_parameters', 'areathres.json')):
                data = IO.load_json(os.path.join('analysis_parameters', 'areathres.json'))
                self.areathres = float(data['areathres'])
            else:
                self.areathres = 30.
        
        if defaultrad == True:
            if os.path.isfile(os.path.join('analysis_parameters', 'rad_neg.json')):
                data = IO.load_json(os.path.join('analysis_parameters', 'rad_neg.json'))
                self.integratedGrad = float(data['integratedGrad'])
            else:
                self.integratedGrad = 0.
                
        if defaultsteep == True:
            if os.path.isfile(os.path.join('analysis_parameters', 'rad_neg.json')):
                data = IO.load_json(os.path.join('analysis_parameters', 'rad_neg.json'))
                self.steepness = float(data['steepness'])
            else:
                self.steepness = 1.
            
        return
    
    def compute_image_props(self, image, k1, k2, thres=0.05, large_thres=600., areathres=30., rdl=[50., 0., 0.]):
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
    
    def compute_spot_props(self, image, k1, k2, thres=0.05, large_thres=600., areathres=30., rdl=[50., 0., 0.]):
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
        estimated_intensity, estimated_background = A_F.estimate_intensity(image, centroids)
        
        return centroids, estimated_intensity, estimated_background

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
        
        IO.make_directory('analysis_parameters')
        IO.save_as_json(to_save, os.path.join('analysis_parameters', 'rad_neg.json'))
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
        areathres = 1000.
        thres = 0.05
        rdl = [self.steepness, self.integratedGrad, 0.]
        for i in np.arange(len(files)):
            file_path = os.path.join(folder, files[i])
            image = IO.read_tiff(file_path)
            if len(image.shape) < 3:
                dl_mask, centroids, radiality = self.compute_image_props(image, k1, k2, thres, large_thres, areathres, rdl)
                pixel_index_list, areas, centroids = A_F.calculate_region_properties(dl_mask)
                if i == 0:
                    a_neg = areas
                else:
                    a_neg = np.hstack([a_neg, areas])
            else:
                for j in np.arange(image.shape[2]):
                    dl_mask, centroids, radiality = self.compute_image_props(image, k1, k2, thres, large_thres, areathres, rdl)
                    pixel_index_list, areas, centroids = A_F.calculate_region_properties(dl_mask)
                    if (i == 0) and (j == 0):
                        a_neg = areas
                    else:
                        a_neg = np.hstack([a_neg, areas])

        area_thresh = int(np.ceil(np.percentile(a_neg, accepted_ratio)))
        
        to_save = {'areathres' : area_thresh}
        
        IO.make_directory('analysis_parameters')
        IO.save_as_json(to_save, os.path.join('analysis_parameters', 'areathres.json'))
        self.areathres = area_thresh
        return
    
    def analyse_images(self, folder, imtype='.tif', thres=0.05, large_thres=600., gsigma=1.4, rwave=2., im_start=1):
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
        """
        files_list = os.listdir(folder)
        files = np.sort([e for e in files_list if imtype in e])
        
        k1, k2 = A_F.create_kernel(gsigma, rwave) # create image processing kernels
        rdl = [self.steepness, self.integratedGrad, 0.]
        
        # create analysis and analysis parameter directories
        analysis_directory = os.path.abspath(folder)+'_analysis'
        IO.make_directory(analysis_directory)
        analysis_p_directory = os.path.abspath(folder)+'_analysisparameters'
        IO.make_directory(analysis_p_directory)

        to_save = {'areathres': self.areathres, 'steepness': self.steepness, 
                   'integratedGrad': self.integratedGrad, 'gaussian_sigma':
                       gsigma, 'ricker_sigma': rwave, 'thres':thres,
                       'large_thres':large_thres}
        IO.save_as_json(to_save, os.path.join(analysis_p_directory, 'analysis_params.json'))
        
        for i in np.arange(len(files)):
            img = IO.read_tiff(os.path.join(folder, files[i]))
            if im_start > 1:
                img = img[:, :, im_start:]
            columns = ['x', 'y', 'z', 'sum_intensity', 'bg']
            if len(img.shape) > 2: # if a z-stack
                z_planes = img.shape[2]
                for zp in np.arange(z_planes):
                    img_z = img[:, :, zp]
                    centroids, estimated_intensity, estimated_background = self.compute_spot_props(img_z, 
                    k1, k2, thres=thres, large_thres=large_thres, 
                    areathres=self.areathres, rdl=rdl)
                    dataarray = np.vstack([centroids[:, 0], centroids[:, 1], 
                    np.full_like(centroids[:, 0], zp+1), estimated_intensity, estimated_background])
                    if zp == 0:
                        to_save = pd.DataFrame(data=dataarray.T, columns=columns)
                    else:
                        to_save = pd.concat([to_save, pd.DataFrame(data=dataarray.T, columns=columns)])
                to_save = to_save.reset_index(drop=True)
                to_save.to_csv(os.path.join(analysis_directory, files[i].split(imtype)[0]+'.csv'), index=False)
            else: # if not a z-stack
                centroids, estimated_intensity, estimated_background = self.compute_spot_props(img, 
                k1, k2, thres=thres, large_thres=large_thres, 
                areathres=self.areathres, rdl=rdl)
                dataarray = np.vstack([centroids[:, 0], centroids[:, 1], 
                np.full_like(centroids[:, 0], zp+1), estimated_intensity, estimated_background])
                to_save = pd.DataFrame(data=dataarray, columns=columns)
                to_save.to_csv(os.path.join(analysis_directory, files[i].split(imtype)[0]+'.csv'), index=False)
        return
    
    def analyse_round_images(self, folder, imtype='.tif', thres=0.05, large_thres=600., gsigma=1.4, rwave=2.):
        """
        analyses data in a folder specified,
        folder has "round/sampleN" structure as in Lee Lab Cambridge Experiment
        saves spots, locations, intensities and backgrounds in a folder created
        next to the folder analysed with _analysis string attached
        also writes a folder with _analysisparameters and saves analysis parameters
        used for particular experiment
    
        Args:
        - folder (string). Folder containing images
        - imtype (string). Type of images being analysed, default tif
        - gisgma (float). gaussian blurring parameter (default 1.4)
        - rwave (float). Ricker wavelent sigma (default 2.)
        """
        files_list = os.listdir(folder)
        files = np.sort([e for e in files_list if imtype in e])
        
        k1, k2 = A_F.create_kernel(gsigma, rwave) # create image processing kernels
        rdl = [self.steepness, self.integratedGrad, 0.]
        
        analysis_directory = os.path.abspath(folder)+'_analysis'
        IO.make_directory(analysis_directory)
        analysis_p_directory = os.path.abspath(folder)+'_analysisparameters'
        IO.make_directory(analysis_p_directory)

        to_save = {'areathres': self.areathres, 'steepness': self.steepness, 
                   'integratedGrad': self.integratedGrad, 'gaussian_sigma':
                       gsigma, 'ricker_sigma': rwave, 'thres':thres,
                       'large_thres':large_thres}
        IO.save_as_json(to_save, os.path.join(analysis_p_directory, 'analysis_params.json'))
       


        return