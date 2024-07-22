# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to IO of files based for the RASP code.
jsb92, 2024/01/02
"""
import json
import os
from skimage import io
import numpy as np
import polars as pl

import sys

module_dir = os.path.dirname(__file__)
sys.path.append(module_dir)
import AnalysisFunctions

A_F = AnalysisFunctions.Analysis_Functions()


class IO_Functions:
    def __init__(self):
        self = self
        return

    def save_analysis(
        self,
        to_save,
        to_save_largeobjects,
        analysis_directory,
        imtype,
        protein_string,
        cell_string,
        files,
        i=0,
        z_planes=[0, 0],
        cell_analysis=False,
        cell_mask=False,
        to_save_cell=False,
        one_savefile=True,
    ):
        """
        saves analysis.

        Args:
            to_save (pl.DataFrame): polars dataframe.
            to_save_largeobjects (pd.DataFrame): pandas dataframe of large objects.
            analysis_directory (string): analysis directory to save in
            imtype (string): string of image type
            protein_string (np.1darray): strings for protein-stained data (default C1)
            cell_string (np.1darray): strings for cell-containing data (default C0)
            i (int): location in files where we're analysing
            z_planes (np.1darray): array of z locations to count spots
            cell_analysis (boolean): if doing cell analysis saving
            cell_mask (np.ndarray): cell mask if cell analysis saving
            to_save_cell (pl.DataFrame): polars dataframe
            one_savefile (boolean): if True, saving all analysis in one csv
        """

        if one_savefile == False:
            savename = os.path.join(
                analysis_directory,
                os.path.split(files[i])[-1].split(imtype)[0] + ".csv",
            )
            savename_lo = os.path.join(
                analysis_directory,
                os.path.split(files[i])[-1].split(imtype)[0] + "_largeobjects.csv",
            )
            to_save.write_csv(savename)
            to_save_largeobjects.write_csv(savename_lo)
            if cell_analysis == True:
                to_save_cell.write_csv(
                    os.path.join(
                        analysis_directory,
                        os.path.split(files[i])[-1]
                        .split(imtype)[0]
                        .split(protein_string)[0]
                        + str(cell_string)
                        + "_cell_analysis.csv",
                    ),
                )
                self.write_tiff(
                    cell_mask,
                    os.path.join(
                        analysis_directory,
                        os.path.split(files[i])[-1]
                        .split(imtype)[0]
                        .split(protein_string)[0]
                        + str(cell_string)
                        + "_cellMask.tiff",
                    ),
                    bit=np.uint8,
                )
        else:
            to_save = to_save.with_columns(
                image_filename=np.full_like(
                    to_save["z"].to_numpy(), files[i], dtype="object"
                )
            )
            to_save_largeobjects = to_save_largeobjects.with_columns(
                image_filename=np.full_like(
                    to_save_largeobjects["z"].to_numpy(), files[i], dtype="object"
                )
            )

            savename = os.path.join(analysis_directory, "spot_analysis.csv")
            savename_lo = os.path.join(analysis_directory, "largeobject_analysis.csv")
            savename_spot = os.path.join(analysis_directory, "spot_numbers.csv")
            savename_nlargeobjects = os.path.join(
                analysis_directory, "largeobject_numbers.csv"
            )

            n_spots = A_F.count_spots(to_save, np.arange(z_planes[0], z_planes[1]))
            n_spots = n_spots.with_columns(
                image_filename=np.full_like(
                    n_spots["z"].to_numpy(), files[i], dtype="object"
                )
            )

            n_largeobjects = A_F.count_spots(
                to_save_largeobjects, np.arange(z_planes[0], z_planes[1])
            )
            n_largeobjects = n_largeobjects.with_columns(
                image_filename=np.full_like(
                    n_largeobjects["z"].to_numpy(), files[i], dtype="object"
                )
            )

            if cell_analysis == True:
                to_save_cell = to_save_cell.with_columns(
                    image_filename=np.full_like(
                        to_save_cell["z"].to_numpy(), files[i], dtype="object"
                    )
                )
                savename_cell = os.path.join(
                    analysis_directory, "cell_colocalisation_analysis.csv"
                )
                self.write_tiff(
                    cell_mask,
                    os.path.join(
                        analysis_directory,
                        os.path.split(files[i])[-1]
                        .split(imtype)[0]
                        .split(protein_string)[0]
                        + str(cell_string)
                        + "_cellMask.tiff",
                    ),
                    bit=np.uint8,
                )

            if i != 0:
                with open(savename, mode="ab") as f:
                    to_save.write_csv(f, include_header=False)
                with open(savename_lo, mode="ab") as f:
                    to_save_largeobjects.write_csv(f, include_header=False)
                with open(savename_spot, mode="ab") as f:
                    n_spots.write_csv(f, include_header=False)
                with open(savename_nlargeobjects, mode="ab") as f:
                    n_largeobjects.write_csv(f, include_header=False)
                if cell_analysis == True:
                    with open(savename_cell, mode="ab") as f:
                        to_save_cell.write_csv(f, include_header=False)
            else:
                to_save.write_csv(savename)
                to_save_largeobjects.write_csv(savename_lo)
                n_spots.write_csv(savename_spot)
                n_largeobjects.write_csv(savename_nlargeobjects)
                if cell_analysis == True:
                    to_save_cell.write_csv(savename_cell)
        return

    def save_analysis_params(
        self, analysis_p_directory, to_save, gain_map=0, offset_map=0
    ):
        """
        saves analysis parameters.

        Args:
            analysis_p_directory (str): The folder to save to.
            to_save (dict): dict to save of analysis parameters.
            gain_map (array): gain_map to save
            offset_map (array): offset_map to save

        """
        self.make_directory(analysis_p_directory)
        self.save_as_json(
            to_save, os.path.join(analysis_p_directory, "analysis_params.json")
        )
        if type(gain_map) != float:
            self.write_tiff(
                gain_map, os.path.join(analysis_p_directory, "gain_map.tif"), np.uint32
            )
        if type(offset_map) != float:
            self.write_tiff(
                offset_map,
                os.path.join(analysis_p_directory, "offset_map.tif"),
                np.uint32,
            )
        return

    def load_json(self, filename):
        """
        Loads data from a JSON file.

        Args:
            filename (str): The name of the JSON file to load.

        Returns:
            data (dict): The loaded JSON data.
        """
        with open(filename, "r") as file:
            data = json.load(file)
        return data

    def make_directory(self, directory_path):
        """
        Creates a directory if it doesn't exist.

        Args:
            directory_path (str): The path of the directory to be created.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def save_as_json(self, data, file_name):
        """
        Saves data to a JSON file.

        Args:
            data (dict): The data to be saved in JSON format.
            file_name (str): The name of the JSON file.
        """
        with open(file_name, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def read_tiff(self, file_path):
        """
        Read a TIFF file using the skimage library.

        Args:
            file_path (str): The path to the TIFF file to be read.

        Returns:
            image (numpy.ndarray): The image data from the TIFF file.
        """
        # Use skimage's imread function to read the TIFF file
        # specifying the 'tifffile' plugin explicitly
        image = io.imread(file_path, plugin="tifffile")
        if len(image.shape) > 2:  # if image a stack
            image = image.T
        return np.asarray(np.swapaxes(image, 0, 1), dtype="double")

    def read_tiff_tophotons(self, file_path, QE=0.95, gain_map=1.0, offset_map=0.0):
        """
        Read a TIFF file using the skimage library.
        Use camera parameters to convert output to photons

        Args:
            file_path (str): The path to the TIFF file to be read.
            QR (float): QE of camera
            gain_map (matrix, or float): gain map. Assumes units of ADU/photoelectrons
            offset_map (matrix, or float): offset map. Assumes units of ADU

        Returns:
            image (numpy.ndarray): The image data from the TIFF file.
        """
        # Use skimage's imread function to read the TIFF file
        # specifying the 'tifffile' plugin explicitly
        image = io.imread(file_path, plugin="tifffile")
        if len(image.shape) > 2:  # if image a stack
            image = image.T
        data = np.asarray(np.swapaxes(image, 0, 1), dtype="double")
        if type(gain_map) is not float:
            if data.shape[:2] != gain_map.shape:
                print(
                    "Gain and offset map not compatible with image dimensions. Defaulting to gain of 1 and offset of 0."
                )
                gain_map = 1.0
                offset_map = 0.0

        if type(gain_map) is not float:
            data = np.divide(
                np.divide(
                    np.subtract(data, offset_map[:, :, np.newaxis]),
                    gain_map[:, :, np.newaxis],
                ),
                QE,
            )
        else:
            data = np.divide(np.divide(np.subtract(data, offset_map), gain_map), QE)
        return data

    def write_tiff(self, volume, file_path, bit=np.uint16):
        """
        Write a TIFF file using the skimage library.

        Args:
            volume (numpy.ndarray): The volume data to be saved as a TIFF file.
            file_path (str): The path where the TIFF file will be saved.
            bit (int): Bit-depth for the saved TIFF file (default is 16).

        Notes:
            The function uses skimage's imsave to save the volume as a TIFF file.
            The plugin is set to 'tifffile' and photometric to 'minisblack'.
            Additional metadata specifying the software as 'Python' is included.
        """
        if len(volume.shape) > 2:  # if image a stack
            volume = volume.T
            volume = np.asarray(np.swapaxes(volume, 1, 2), dtype="double")
        io.imsave(
            file_path,
            np.asarray(volume, dtype=bit),
            plugin="tifffile",
            bigtiff=True,
            photometric="minisblack",
            metadata={"Software": "Python"},
            check_contrast=False,
        )
