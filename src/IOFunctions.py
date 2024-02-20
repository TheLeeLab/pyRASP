# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to IO of files based for the
RASP code
jsb92, 2024/01/02
"""
import json
import os
from skimage import io
import numpy as np

class IO_Functions():
    def __init__(self):
        self = self
        return
    
    def save_analysis_params(self, analysis_p_directory, to_save, gain_map=0, offset_map=0):
        """
        saves analysis parameters.
    
        Args:
        - analysis_p_directory (str): The folder to save to.
        - to_save (dict): dict to save of analysis parameters.
        - gain_map (array): gain_map to save
        - offset_map (array): offset_map to save
    
        """
        self.make_directory(analysis_p_directory)
        self.save_as_json(to_save, os.path.join(analysis_p_directory, 'analysis_params.json'))
        if type(gain_map) != float:
            self.write_tiff(gain_map, os.path.join(analysis_p_directory, 'gain_map.tif'), np.uint32)
        if type(offset_map) != float:
            self.write_tiff(offset_map, os.path.join(analysis_p_directory, 'offset_map.tif'), np.uint32)
        return
    
    def load_json(self, filename):
        """
        Loads data from a JSON file.
    
        Args:
        - filename (str): The name of the JSON file to load.
    
        Returns:
        - data (dict): The loaded JSON data.
        """
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    
    def make_directory(self, directory_path):
        """
        Creates a directory if it doesn't exist.

        Args:
        - directory_path (str): The path of the directory to be created.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def save_as_json(self, data, file_name):
        """
        Saves data to a JSON file.
    
        Args:
        - data (dict): The data to be saved in JSON format.
        - file_name (str): The name of the JSON file.
        """
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def read_tiff(self, file_path):
        """
        Read a TIFF file using the skimage library.
    
        Args:
        - file_path (str): The path to the TIFF file to be read.
    
        Returns:
        - image (numpy.ndarray): The image data from the TIFF file.
        """
        # Use skimage's imread function to read the TIFF file
        # specifying the 'tifffile' plugin explicitly
        image = io.imread(file_path, plugin='tifffile')
        if len(image.shape) > 2: # if image a stack
            if image.shape[0] < image.shape[-1]: # if stack is wrong way round
                image = image.T
        return np.asarray(np.swapaxes(image,0,1), dtype='double')
    
    def read_tiff_tophotons(self, file_path, QE=0.95, gain_map=1., offset_map=0.):
        """
        Read a TIFF file using the skimage library.
        Use camera parameters to convert output to photons
    
        Args:
        - file_path (str): The path to the TIFF file to be read.
        - QR (float): QE of camera
        - gain_map (matrix, or float): gain map. Assumes units of ADU/photoelectrons
        - offset_map (matrix, or float): offset map. Assumes units of ADU
    
        Returns:
        - image (numpy.ndarray): The image data from the TIFF file.
        """
        # Use skimage's imread function to read the TIFF file
        # specifying the 'tifffile' plugin explicitly
        image = io.imread(file_path, plugin='tifffile')
        if len(image.shape) > 2: # if image a stack
            if image.shape[0] < image.shape[-1]: # if stack is wrong way round
                image = image.T
        data = np.asarray(np.swapaxes(image,0,1), dtype='double')
        if type(gain_map) is not float:
            if data.shape[:2] != gain_map.shape:
                print("Gain and offset map not compatible with image dimensions. Defaulting to gain of 1 and offset of 0.")
                gain_map = 1.
                offset_map = 0.
        
        if type(gain_map) is not float:
            data = np.divide(np.divide(np.subtract(data, offset_map[:, :, np.newaxis]), gain_map[:, :, np.newaxis]), QE)
        else:
            data = np.divide(np.divide(np.subtract(data, offset_map), gain_map), QE)
        return data
    
    def write_tiff(self, volume, file_path, bit=np.uint16):
        """
        Write a TIFF file using the skimage library.
    
        Args:
        - volume (numpy.ndarray): The volume data to be saved as a TIFF file.
        - file_path (str): The path where the TIFF file will be saved.
        - bit (int): Bit-depth for the saved TIFF file (default is 16).
    
        Notes:
        - The function uses skimage's imsave to save the volume as a TIFF file.
        - The plugin is set to 'tifffile' and photometric to 'minisblack'.
        - Additional metadata specifying the software as 'Python' is included.
        """
        io.imsave(file_path, np.asarray(volume, dtype=bit), plugin='tifffile', bigtiff=True, photometric='minisblack', metadata={'Software': 'Python'}, check_contrast=False)
