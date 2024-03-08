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
            
    def read_multichannel_tiff_tophotons(self, file_path, order='tzc', n_t=1, n_z=1, n_c=1):
        """
        Read a multichannel TIFF file using the skimage library, 
        averaging/splitting as appropriate. Converts to photons.
    
        Args:
        - file_path (str): The path to the TIFF file to be read.
        - order (str): default 'tzc', order images are saved in (this defines the averaging).
            Options are:
                'tzc' (first does all time steps, then does these at different zs, then returns and does other colour channels)
                'tcz' (first does all time steps, then does different colour channels, then moves in z)
                'ztc' (first does all z stacks, then repeats at different time steps, then does other colour channels)
                'zct' (first does all z stacks, then does at different colour channels, then does time steps)
                'ctz' (first does different colours, then repeats at different time steps, then moves in z)
                'czt' (first does different colours, then does different z stacks, then does all time steps)
        - n_t (int): number of time steps per condition. Default 1
        - n_z (int): number of z stacks per condition. Default 1
        - n_c (int): number of colours per condition. Default 1
        
        Returns:
        - image_colourchannels (dict of numpy.ndarrays): Dict of the image data
        from the TIFF file. Dict will contain N objects that are N colour channels.
        Final dimension of the NDarrays is the z coordiante.
        """
        image_unaveraged = self.read_tiff_tophotons(file_path)
        image_colourchannels = {}
        if 't' == order[0]: # if a time average first
            tlocs = np.arange(0, image_unaveraged.shape[-1]+n_t, n_t)
            image = np.zeros([image_unaveraged.shape[0], image_unaveraged.shape[0], n_z*n_c])
            for t in np.arange(len(tlocs)-1):
                image[:, :, t] = np.mean(image_unaveraged[:, :, tlocs[t]:tlocs[t+1]], axis=-1)
                
            if 'z' == order[1]: # if then does z-steps
                dividing_locations = np.arange(0, image.shape[-1]+n_z, n_z)
                for i in np.arange(len(dividing_locations)-1):
                    minloc = dividing_locations[i]
                    maxloc = dividing_locations[i+1]
                    image_colourchannels[i] = image[:, :, minloc:maxloc]
            else:
                for colour in np.arange(n_c):
                    colour_locs = np.arange(colour, image.shape[-1], n_c, dtype=int)
                    image_colourchannels[colour] = image[:, :, colour_locs]
        return image_colourchannels

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
        if len(volume.shape) > 2: # if image a stack
            volume = volume.T
            volume = np.asarray(np.swapaxes(volume,1,2), dtype='double')
        io.imsave(file_path, np.asarray(volume, dtype=bit), plugin='tifffile', bigtiff=True, photometric='minisblack', metadata={'Software': 'Python'}, check_contrast=False)
