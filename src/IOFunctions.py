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
import denoisetools as ncs
import warnings

warnings.filterwarnings("ignore")


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
        lo_mask=[0],
        cell_mask=None,
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
            lo_mask (np.ndarray): mask of large objects
            cell_mask (np.ndarray): cell mask if cell mask saving
            one_savefile (boolean): if True, saving all analysis in one csv
        """
        module_dir = os.path.dirname(__file__)
        sys.path.append(module_dir)
        import AnalysisFunctions

        A_F = AnalysisFunctions.Analysis_Functions()

        def _get_base_filename(file):
            return os.path.split(file)[-1].split(imtype)[0]

        def _write_dataframe(df, filepath, append=False):
            if df.shape[0] > 0:
                if append and os.path.isfile(filepath):
                    with open(filepath, mode="ab") as f:
                        df.write_csv(f, include_header=False)
                else:
                    df.write_csv(filepath)

        # Handle separate file saving
        if not one_savefile:
            base_filename = _get_base_filename(files[i])
            savename = os.path.join(analysis_directory, f"{base_filename}.csv")
            savename_lo = os.path.join(
                analysis_directory, f"{base_filename}_largeobjects.csv"
            )

            to_save.write_csv(savename)
            to_save_largeobjects.write_csv(savename_lo)

            self.write_tiff(
                lo_mask,
                os.path.join(analysis_directory, f"{base_filename}_loMask.tiff"),
                bit=np.uint8,
            )

            if cell_mask is not None:
                self.write_tiff(
                    cell_mask,
                    os.path.join(
                        analysis_directory,
                        f"{base_filename}.split(protein_string)[0]{cell_string}_cellMask.tiff",
                    ),
                    bit=np.uint8,
                )
            return

        # Handling single file saving
        if to_save is not None:
            to_save = to_save.with_columns(
                image_filename=np.full_like(
                    to_save["z"].to_numpy(), files[i], dtype="object"
                )
            )
            n_spots = A_F.count_spots(to_save, np.arange(z_planes[0], z_planes[1]))
            n_spots = n_spots.with_columns(
                image_filename=np.full_like(
                    n_spots["z"].to_numpy(), files[i], dtype="object"
                )
            )
        else:
            n_spots = None

        if to_save_largeobjects is not None:
            to_save_largeobjects = to_save_largeobjects.with_columns(
                image_filename=np.full_like(
                    to_save_largeobjects["z"].to_numpy(), files[i], dtype="object"
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
        else:
            n_largeobjects = None

        # Write large object mask
        self.write_tiff(
            lo_mask,
            os.path.join(
                analysis_directory, f"{_get_base_filename(files[i])}_loMask.tiff"
            ),
            bit=np.uint8,
        )

        # Prepare save paths
        save_paths = [
            os.path.join(analysis_directory, "spot_analysis.csv"),
            os.path.join(analysis_directory, "largeobject_analysis.csv"),
            os.path.join(analysis_directory, "spot_numbers.csv"),
            os.path.join(analysis_directory, "largeobject_numbers.csv"),
        ]

        # Count spots and large objects

        # Optional cell mask
        if cell_mask is not None:
            self.write_tiff(
                cell_mask,
                os.path.join(
                    analysis_directory,
                    f"{_get_base_filename(files[i]).split(protein_string)[0]}{cell_string}_cellMask.tiff",
                ),
                bit=np.uint8,
            )

        # Save or append data
        dfs_to_save = [to_save, to_save_largeobjects, n_spots, n_largeobjects]
        for df, path in zip(dfs_to_save, save_paths):
            if df is not None:
                _write_dataframe(df, path, append=(i != 0))
        return

    def save_analysis_params(
        self,
        analysis_p_directory,
        to_save,
        gain_map=0,
        offset_map=0,
        variance_map=0,
    ):
        """
        saves analysis parameters.

        Args:
            analysis_p_directory (str): The folder to save to.
            to_save (dict): dict to save of analysis parameters.
            gain_map (array): gain_map to save
            offset_map (array): offset_map to save
            variance_map (array): variance_map to save

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
        if type(variance_map) != float:
            self.write_tiff(
                variance_map,
                os.path.join(analysis_p_directory, "variance_map.tif"),
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

    def read_tiff(self, file_path, frame=None):
        """
        Read a TIFF file using the skimage library.

        Args:
            file_path (str): The path to the TIFF file to be read.
            frame (int): if not None, loads a single frame

        Returns:
            image (numpy.ndarray): The image data from the TIFF file.
        """
        # Use skimage's imread function to read the TIFF file
        # specifying the 'tifffile' plugin explicitly
        if isinstance(frame, type(None)):
            image = io.imread(file_path, plugin="tifffile")
        else:
            image = io.imread(file_path, plugin="tifffile", key=frame)
        if len(image.shape) > 2:  # if image a stack
            image = np.swapaxes(np.swapaxes(image, 0, -1), 0, 1)
        return np.asarray(image, dtype="double")

    def read_tiff_tophotons(
        self,
        file_path,
        QE=0.95,
        gain_map=1.0,
        offset_map=0.0,
        variance_map=1.0,
        frame=None,
        error_correction=False,
        NA=1.45,
        wavelength=0.6,
        pixelsize=0.11,
        alpha=0.2,
        R=40,
    ):
        """
        Read a TIFF file using the skimage library.
        Use camera parameters to convert output to photons

        Args:
            file_path (str): The path to the TIFF file to be read.
            QR (float): QE of camera
            gain_map (matrix, or float): gain map. Assumes units of ADU/photoelectrons
            offset_map (matrix, or float): offset map. Assumes units of ADU
            frame (int, optional): if not None, loads a single frame
            error_correction (boolean): if True, uses Huang's error correction
            NA (float): if using Huang's error correction, needed
            wavelength (float): if using Huang's error correction, needed (same units as pixel size)
            pixelsize (float): pixel size if error correcting
            alpha (float): weighting parameter for error correction (typically 0.2)
            iterationN (int): number of iterations for the error correction


        Returns:
            image (numpy.ndarray): The image data from the TIFF file.
        """
        # Use skimage's imread function to read the TIFF file
        # specifying the 'tifffile' plugin explicitly
        if isinstance(frame, type(None)):
            data = io.imread(file_path, plugin="tifffile")
        else:
            data = io.imread(file_path, plugin="tifffile", key=frame)
        if len(data.shape) > 2:  # if image a stack
            data = np.swapaxes(np.swapaxes(data, 0, -1), 0, 1)

        if type(gain_map) is not float:
            if data.shape[:2] != gain_map.shape:
                print(
                    "Gain and offset map not compatible with image dimensions. Defaulting to gain of 1 and offset of 0."
                )
                gain_map = 1.0
                offset_map = 0.0

        if error_correction == True:
            for z in np.arange(data.shape[-1]):
                data[:, :, z] = np.squeeze(
                    ncs.reducenoise(
                        R,
                        np.expand_dims(data[:, :, z], 0),
                        variance_map,
                        gain_map,
                        int(data.shape[0]),
                        pixelsize,
                        NA,
                        wavelength,
                        alpha,
                        15,
                        Type="OTFweighted",
                    )
                )
        else:
            if type(gain_map) is not float:
                if len(data.shape) > 2:
                    data = np.divide(
                        np.divide(
                            np.subtract(data, offset_map[:, :, np.newaxis]),
                            gain_map[:, :, np.newaxis],
                        ),
                        QE,
                    )
                else:
                    data = np.divide(
                        np.divide(np.subtract(data, offset_map), gain_map), QE
                    )
            else:
                data = np.divide(np.divide(np.subtract(data, offset_map), gain_map), QE)
        return data

    def write_tiff(self, volume, file_path, bit=np.uint16, pixel_size=0.11):
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
        xamount = str(volume.shape[0])
        yamount = str(volume.shape[1])
        if len(volume.shape) > 2:  # if image a stack
            volume = volume.T
            volume = np.asarray(np.swapaxes(volume, 1, 2))

        description = "ImageJ=1.54f\nunit=micron\nmin=" + xamount + "\nmax=" + yamount

        pixel_unit = int(1e6 / pixel_size)

        extra_tags = [
            ("ImageDescription", "s", 1, description, True),
            ("XResolution", "i", 2, (pixel_unit, 1000000), True),
            ("YResolution", "i", 2, (pixel_unit, 1000000), True),
            ("ResolutionUnit", "i", 1, True),
        ]

        io.imsave(
            file_path,
            np.asarray(volume, dtype=bit),
            plugin="tifffile",
            extratags=extra_tags,
            check_contrast=False,
        )

    def save_abovebelowthresholdcoloc(
        self,
        plane_1_analysis_AT,
        plane_2_analysis_AT,
        spot_1_analysis_AT,
        spot_2_analysis_AT,
        plane_1_analysis_UT,
        plane_2_analysis_UT,
        spot_1_analysis_UT,
        spot_2_analysis_UT,
        analysis_file_1,
        analysis_file_2,
        spot_1_string,
        spot_2_string,
        threshold1_str,
        threshold2_str,
    ):
        """
        saves analysis of above and below threshold.

        Args:
            plane_1_analysis_AT (pl.DataFrame): polars dataframe.
            plane_2_analysis_AT (pl.DataFrame): polars dataframe.
            spot_1_analysis_AT (pl.DataFrame): polars dataframe.
            spot_2_analysis_AT (pl.DataFrame): polars dataframe.
            plane_1_analysis_UT (pl.DataFrame): polars dataframe.
            plane_2_analysis_UT (pl.DataFrame): polars dataframe.
            spot_1_analysis_UT (pl.DataFrame): polars dataframe.
            spot_2_analysis_UT (pl.DataFrame): polars dataframe.
            analysis_file_1 (str): string of analysis file 1.
            analysis_file_2 (str): string of analysis file 1.
            spot_1_string (str): string of spot 1.
            spot_2_string (str): string of spot 2.
            threshold1_str (str): string of threshold 1.
            threshold2_str (str): string of threshold 2.
        """

        def _generate_filename(
            base_file,
            spot_str,
            threshold1,
            spot1_str,
            threshold2,
            spot2_str,
            prefix="",
            suffix="",
        ):
            return f"{base_file.split('.')[0]}{prefix}_colocalisationwith_{spot_str}_{threshold1}_{spot1_str}_photonthreshold_{threshold2}_{spot2_str}_photonthreshold{suffix}.csv"

        def _save_dataframe(dataframe, filename):
            if isinstance(dataframe, pl.DataFrame):
                dataframe.write_csv(filename)

        _save_dataframe(
            plane_1_analysis_AT,
            _generate_filename(
                analysis_file_1,
                spot_2_string,
                threshold1_str,
                spot_1_string,
                threshold2_str,
                spot_2_string,
                suffix="_abovethreshold",
            ),
        )
        _save_dataframe(
            plane_1_analysis_UT,
            _generate_filename(
                analysis_file_1,
                spot_2_string,
                threshold1_str,
                spot_1_string,
                threshold2_str,
                spot_2_string,
                suffix="_belowthreshold",
            ),
        )

        _save_dataframe(
            plane_2_analysis_AT,
            _generate_filename(
                analysis_file_2,
                spot_1_string,
                threshold2_str,
                spot_2_string,
                threshold1_str,
                spot_1_string,
                suffix="_abovethreshold",
            ),
        )
        _save_dataframe(
            plane_1_analysis_UT,
            _generate_filename(
                analysis_file_2,
                spot_1_string,
                threshold2_str,
                spot_2_string,
                threshold1_str,
                spot_1_string,
                suffix="_belowthreshold",
            ),
        )

        # Save spot analyses
        _save_dataframes = [
            (
                spot_1_analysis_AT,
                analysis_file_1,
                spot_2_string,
                threshold1_str,
                spot_1_string,
                threshold2_str,
                "abovethreshold",
            ),
            (
                spot_1_analysis_UT,
                analysis_file_1,
                spot_2_string,
                threshold1_str,
                spot_1_string,
                threshold2_str,
                "belowthreshold",
            ),
            (
                spot_2_analysis_AT,
                analysis_file_2,
                spot_1_string,
                threshold2_str,
                spot_2_string,
                threshold1_str,
                "abovethreshold",
            ),
            (
                spot_2_analysis_UT,
                analysis_file_2,
                spot_1_string,
                threshold2_str,
                spot_2_string,
                threshold1_str,
                "belowthreshold",
            ),
        ]

        for (
            dataframe,
            file,
            spot_str1,
            threshold1,
            spot_str2,
            threshold2,
            threshold_type,
        ) in _save_dataframes:
            _save_dataframe(
                dataframe,
                _generate_filename(
                    file,
                    spot_str1,
                    threshold1,
                    spot_str2,
                    threshold2,
                    spot_str1,
                    prefix="_rawcolocalisationwith",
                    suffix=f"_{threshold_type}",
                ),
            )
        return
