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
    
    def write_tiff(self, volume, file_path, bit=16):
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
        io.imsave(file_path, volume, plugin='tifffile', photometric='minisblack', metadata={'Software': 'Python'})
