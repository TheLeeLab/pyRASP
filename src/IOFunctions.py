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


class IO_Functions:
    def __init__(self):
        self = self
        return

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
        if isinstance(plane_1_analysis_AT, pl.DataFrame) and isinstance(
            plane_1_analysis_UT, pl.DataFrame
        ):
            above_str = "coincidence_above_" + threshold1_str
            above_cc_str = "chance_coincidence_above_" + threshold1_str
            below_str = "coincidence_below_" + threshold1_str
            below_cc_str = "chance_coincidence_below_" + threshold1_str

            plane_1_analysis = plane_1_analysis_AT
            plane_1_analysis = plane_1_analysis.rename({"coincidence": above_str})
            plane_1_analysis = plane_1_analysis.rename(
                {"chance_coincidence": above_cc_str}
            )

            plane_1_analysis = plane_1_analysis.with_columns(
                channelcol=plane_1_analysis_UT["coincidence"]
            ).rename({"channelcol": below_str})
            plane_1_analysis = plane_1_analysis.with_columns(
                channelcol=plane_1_analysis_UT["chance_coincidence"]
            ).rename({"channelcol": below_cc_str})

            plane_1_analysis = plane_1_analysis[
                above_str,
                above_cc_str,
                below_str,
                below_cc_str,
                "z",
                "image_filename",
            ]

            plane_1_analysis.write_csv(
                analysis_file_1.split(".")[0]
                + "_colocalisationwith_"
                + spot_2_string
                + "_"
                + threshold1_str
                + "_"
                + spot_1_string
                + "_photonthreshold_"
                + threshold2_str
                + "_"
                + spot_2_string
                + "_photonthreshold.csv"
            )
        else:
            if isinstance(plane_1_analysis_AT, pl.DataFrame):
                plane_1_analysis = plane_1_analysis_AT
                plane_1_analysis_AT.write_csv(
                    analysis_file_1.split(".")[0]
                    + "_colocalisationwith_"
                    + spot_2_string
                    + "_"
                    + threshold1_str
                    + "_"
                    + spot_1_string
                    + "_photonthreshold_"
                    + threshold2_str
                    + "_"
                    + spot_2_string
                    + "_photonthreshold_abovethreshold.csv"
                )
            if isinstance(plane_1_analysis_UT, pl.DataFrame):
                plane_1_analysis = plane_1_analysis_UT
                plane_1_analysis_UT.write_csv(
                    analysis_file_1.split(".")[0]
                    + "_colocalisationwith_"
                    + spot_2_string
                    + "_"
                    + threshold1_str
                    + "_"
                    + spot_1_string
                    + "_photonthreshold_"
                    + threshold2_str
                    + "_"
                    + spot_2_string
                    + "_photonthreshold_belowthreshold.csv"
                )

        if isinstance(plane_2_analysis_AT, pl.DataFrame) and isinstance(
            plane_2_analysis_UT, pl.DataFrame
        ):
            above_str = "coincidence_above_" + threshold2_str
            above_cc_str = "chance_coincidence_above_" + threshold2_str
            below_str = "coincidence_below_" + threshold2_str
            below_cc_str = "chance_coincidence_below_" + threshold2_str

            plane_2_analysis = plane_2_analysis_AT
            plane_2_analysis = plane_2_analysis.rename({"coincidence": above_str})
            plane_2_analysis = plane_2_analysis.rename(
                {"chance_coincidence": above_cc_str}
            )

            plane_2_analysis = plane_2_analysis.with_columns(
                channelcol=plane_2_analysis_UT["coincidence"]
            ).rename({"channelcol": below_str})
            plane_2_analysis = plane_2_analysis.with_columns(
                channelcol=plane_2_analysis_UT["chance_coincidence"]
            ).rename({"channelcol": below_cc_str})

            plane_2_analysis = plane_2_analysis[
                above_str,
                above_cc_str,
                below_str,
                below_cc_str,
                "z",
                "image_filename",
            ]

            plane_2_analysis.write_csv(
                analysis_file_2.split(".")[0]
                + "_colocalisationwith_"
                + spot_1_string
                + "_"
                + threshold2_str
                + "_"
                + spot_2_string
                + "_photonthreshold_"
                + threshold1_str
                + "_"
                + spot_1_string
                + "_photonthreshold.csv"
            )
        else:
            if isinstance(plane_2_analysis_AT, pl.DataFrame):
                plane_2_analysis = plane_2_analysis_AT
                plane_2_analysis_AT.write_csv(
                    analysis_file_2.split(".")[0]
                    + "_colocalisationwith_"
                    + spot_1_string
                    + "_"
                    + threshold2_str
                    + "_"
                    + spot_2_string
                    + "_photonthreshold_"
                    + threshold1_str
                    + "_"
                    + spot_1_string
                    + "_photonthreshold_abovethreshold.csv"
                )
            if isinstance(plane_2_analysis_UT, pl.DataFrame):
                plane_2_analysis = plane_2_analysis_UT
                plane_2_analysis_UT.write_csv(
                    analysis_file_2.split(".")[0]
                    + "_colocalisationwith_"
                    + spot_1_string
                    + "_"
                    + threshold2_str
                    + "_"
                    + spot_2_string
                    + "_photonthreshold_"
                    + threshold1_str
                    + "_"
                    + spot_1_string
                    + "_photonthreshold_belowthreshold.csv"
                )

        if isinstance(spot_1_analysis_AT, pl.DataFrame):
            spot_1_analysis_AT.write_csv(
                analysis_file_1.split(".")[0]
                + "_rawcolocalisationwith_"
                + spot_2_string
                + "_"
                + threshold1_str
                + "_"
                + spot_1_string
                + "_photonthreshold_"
                + threshold2_str
                + "_"
                + spot_2_string
                + "_photonthreshold_abovethreshold.csv"
            )
        if isinstance(spot_1_analysis_UT, pl.DataFrame):
            spot_1_analysis_UT.write_csv(
                analysis_file_1.split(".")[0]
                + "_rawcolocalisationwith_"
                + spot_2_string
                + "_"
                + threshold1_str
                + "_"
                + spot_1_string
                + "_photonthreshold_"
                + threshold2_str
                + "_"
                + spot_2_string
                + "_photonthreshold_belowthreshold.csv"
            )
        if isinstance(spot_2_analysis_AT, pl.DataFrame):
            spot_2_analysis_AT.write_csv(
                analysis_file_2.split(".")[0]
                + "_rawcolocalisationwith_"
                + spot_1_string
                + "_"
                + threshold2_str
                + "_"
                + spot_2_string
                + "_photonthreshold_"
                + threshold1_str
                + "_"
                + spot_1_string
                + "_photonthreshold_abovethreshold.csv"
            )
        if isinstance(spot_2_analysis_UT, pl.DataFrame):
            spot_2_analysis_UT.write_csv(
                analysis_file_2.split(".")[0]
                + "_rawcolocalisationwith_"
                + spot_1_string
                + "_"
                + threshold2_str
                + "_"
                + spot_2_string
                + "_photonthreshold_"
                + threshold1_str
                + "_"
                + spot_1_string
                + "_photonthreshold_belowthreshold.csv"
            )
        return plane_1_analysis, plane_2_analysis

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
            lo_mask (np.ndarray): mask of large objects
            cell_analysis (boolean): if doing cell analysis saving
            cell_mask (np.ndarray): cell mask if cell analysis saving
            to_save_cell (pl.DataFrame): polars dataframe
            one_savefile (boolean): if True, saving all analysis in one csv
        """
        module_dir = os.path.dirname(__file__)
        sys.path.append(module_dir)
        import AnalysisFunctions

        A_F = AnalysisFunctions.Analysis_Functions()

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

            self.write_tiff(
                lo_mask,
                os.path.join(
                    analysis_directory,
                    os.path.split(files[i])[-1].split(imtype)[0] + "_loMask.tiff",
                ),
                bit=np.uint8,
            )

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

            self.write_tiff(
                lo_mask,
                os.path.join(
                    analysis_directory,
                    os.path.split(files[i])[-1].split(imtype)[0] + "_loMask.tiff",
                ),
                bit=np.uint8,
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
                if to_save.shape[0] > 0:
                    if os.path.isfile(savename):
                        with open(savename, mode="ab") as f:
                            to_save.write_csv(f, include_header=False)
                    else:
                        to_save.write_csv(savename)
                if to_save_largeobjects.shape[0] > 0:
                    if os.path.isfile(savename_lo):
                        with open(savename_lo, mode="ab") as f:
                            to_save_largeobjects.write_csv(f, include_header=False)
                    else:
                        to_save_largeobjects.write_csv(savename_lo)
                if n_spots.shape[0] > 0:
                    if os.path.isfile(savename_spot):
                        with open(savename_spot, mode="ab") as f:
                            n_spots.write_csv(f, include_header=False)
                    else:
                        n_spots.write_csv(savename_spot)
                if n_largeobjects.shape[0] > 0:
                    if os.path.isfile(savename_nlargeobjects):
                        with open(savename_nlargeobjects, mode="ab") as f:
                            n_largeobjects.write_csv(f, include_header=False)
                    else:
                        n_largeobjects.write_csv(savename_nlargeobjects)
                if cell_analysis == True:
                    if to_save_cell.shape[0] > 0:
                        if os.path.isfile(savename_cell):
                            with open(savename_cell, mode="ab") as f:
                                to_save_cell.write_csv(f, include_header=False)
                        else:
                            to_save_cell.write_csv(savename_cell)
            else:
                if to_save.shape[0] > 0:
                    to_save.write_csv(savename)
                if to_save_largeobjects.shape[0] > 0:
                    to_save_largeobjects.write_csv(savename_lo)
                if n_spots.shape[0] > 0:
                    n_spots.write_csv(savename_spot)
                if n_largeobjects.shape[0] > 0:
                    n_largeobjects.write_csv(savename_nlargeobjects)
                if cell_analysis == True:
                    if to_save_cell.shape[0] > 0:
                        to_save_cell.write_csv(savename_cell)
        return

    def save_analysis_params(
        self, analysis_p_directory, to_save, gain_map=0, offset_map=0, variance_map=0,
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
        R=8,
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
                data[:,:,z] = np.squeeze(ncs.reducenoise(
                    R,
                    np.expand_dims(data[:,:,z], 0),
                    variance_map,
                    gain_map,
                    int(data.shape[0]),
                    pixelsize,
                    NA,
                    wavelength,
                    alpha,
                    15,
                    Type="OTFweighted",
                ))
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
                    data = np.divide(np.divide(np.subtract(data, offset_map), gain_map), QE)
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
