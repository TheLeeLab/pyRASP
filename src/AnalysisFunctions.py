# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to analysis of images based on their
radiality, relating to the RASP concept.
jsb92, 2024/01/02
"""
import numpy as np
import polars as pl
import pathos
from pathos.pools import ThreadPool as Pool
from rdfpy import rdf
import time
from copy import copy

import os
import sys

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
import IOFunctions
import MultiD_RD_functions
import CoincidenceFunctions
import HelperFunctions
import Image_Analysis_Functions

C_F = CoincidenceFunctions.Coincidence_Functions()
H_F = HelperFunctions.Helper_Functions()
IA_F = Image_Analysis_Functions.ImageAnalysis_Functions()

IO = IOFunctions.IO_Functions()


class Analysis_Functions:
    def __init__(self, cpu_load=0.8):
        self.cpu_number = int(pathos.helpers.cpu_count() * cpu_load)
        return

    def count_spots(self, database, threshold=None):
        """
        Counts spots per z plane, optionally with a threshold.

        Args:
            database (polars DataFrame): DataFrame of spots.
            threshold (float, optional): Intensity threshold.

        Returns:
            polars.DataFrame: DataFrame with number of spots per z-plane.
        """
        results = None
        if threshold is None:
            columns = ["n_spots", "z", "image_filename"]
        else:
            columns = ["n_spots_above", "n_spots_below", "z", "image_filename"]
        for image_filename in np.sort(np.unique(database["image_filename"])):
            dataslice = database.filter(pl.col("image_filename") == image_filename)
            z_planes = np.sort(np.unique(dataslice["z"].to_numpy()))
            if threshold is None:
                spots_per_plane = [
                    len(dataslice.filter(pl.col("z") == z)["sum_intensity_in_photons"])
                    for z in z_planes
                ]
            else:
                spots_above = [
                    np.sum(
                        dataslice.filter(pl.col("z") == z)[
                            "sum_intensity_in_photons"
                        ].to_numpy()
                        > threshold
                    )
                    for z in z_planes
                ]
                spots_below = [
                    np.sum(
                        dataslice.filter(pl.col("z") == z)[
                            "sum_intensity_in_photons"
                        ].to_numpy()
                        <= threshold
                    )
                    for z in z_planes
                ]
                spots_per_plane = np.vstack([spots_above, spots_below])
            if results is not None:
                results = np.hstack(
                    [
                        results,
                        np.vstack(
                            [
                                spots_per_plane,
                                z_planes,
                                np.full(len(z_planes), image_filename),
                            ]
                        ),
                    ]
                )
            else:
                results = np.vstack(
                    [spots_per_plane, z_planes, np.full(len(z_planes), image_filename)]
                )
            if results is not None:
                spot_save = {}
                for i, column in enumerate(columns):
                    spot_save[column] = results[i, :]
                data = H_F.clean_database(pl.DataFrame(data=spot_save), columns)
            else:
                data = None
        return data

    def generate_indices(self, data, image_size, is_mask=False, is_lo=False):
        """
        Generate indices from coordinates.

        Args:
            data (np.ndarray): Array of coordinates or mask.
            image_size (tuple): Size of the image.
            is_mask (bool): Indicates if the data is a mask (default: False).

        Returns:
            np.ndarray: Indices of the data.
        """
        if is_mask:
            if is_lo:
                pil, _, _, _, _ = IA_F.calculate_region_properties(data)
                for i in np.arange(len(pil)):
                    pil[i] = np.ravel_multi_index(
                        [pil[i][:, 0], pil[i][:, 1]], image_size, order="F"
                    )
                return pil, len(pil)
            else:
                coords = np.column_stack(np.nonzero(data))
        else:
            coords = data
        for i in np.arange(len(image_size)):
            indices = np.where((coords[:, i] < 0) | (coords[:, i] >= image_size[i]))[0]
            coords[np.unique(indices), :] = 0
        return np.ravel_multi_index(coords.T, image_size, order="F")

    # TODO: correct this, it's crap
    def Gauss2DFitting(self, image, pixel_index_list, expanded_area=5):
        """
        Gets HWHM of PSFs from fitting Gaussians to spots (from pixel_index_list)
        Pixels expanded borders by n pixels (default 5)

        Args:
            image (array): image as numpy array
            pixel_index_list (list): list of pixel arrays
            expended area (int): range to expand the pixel mask to

        Returns:
            HWHMarray (array): array of half-width-half maxima from the fits
        """
        # this code from https://scipy-cookbook.readthedocs.io/items/FittingData.html
        from scipy import optimize

        def gauss_fit(height, center_x, center_y, width_x, width_y, bg):
            """Returns a gaussian function with the given parameters"""
            width_x = float(width_x)
            width_y = float(width_y)
            return (
                lambda x, y: height
                * np.exp(
                    -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2)
                    / 2
                )
                + bg
            )

        def moments(data):
            """Returns (height, x, y, width_x, width_y)
            the gaussian parameters of a 2D distribution by calculating its
            moments"""
            total = data.sum()
            X, Y = np.indices(data.shape)
            x = (X * data).sum() / total
            y = (Y * data).sum() / total
            col = data[:, int(y)]
            width_x = np.sqrt(
                np.abs((np.arange(col.size) - x) ** 2 * col).sum() / col.sum()
            )
            row = data[int(x), :]
            width_y = np.sqrt(
                np.abs((np.arange(row.size) - y) ** 2 * row).sum() / row.sum()
            )
            height = data.max()
            bg = data.min()
            return height, x, y, width_x, width_y, bg

        def fitgaussian(data):
            """Returns (height, x, y, width_x, width_y, bg)
            the gaussian parameters of a 2D distribution found by a fit"""
            params = moments(data)
            errorfunction = lambda p: np.ravel(
                (gauss_fit(*p)(*np.indices(data.shape)) - data)
            )
            p, success = optimize.leastsq(errorfunction, params)
            return np.abs(p), success

        HWHMarray = np.zeros(len(pixel_index_list))
        for p in np.arange(len(pixel_index_list)):
            xp = np.arange(
                np.min(np.unique(pixel_index_list[p][:, 0])) - expanded_area,
                np.max(np.unique(pixel_index_list[p][:, 0])) + expanded_area,
            )
            yp = np.arange(
                np.min(np.unique(pixel_index_list[p][:, 1])) - expanded_area,
                np.max(np.unique(pixel_index_list[p][:, 1])) + expanded_area,
            )
            x, y = np.meshgrid(xp, yp)
            params, success = fitgaussian(image[x, y])
            if success == True:
                (height, cx, cy, width_x, width_y, bg) = params
                HWHMarray[p] = np.sqrt(2 * np.log(2)) * np.mean([width_y, width_x])
            else:
                HWHMarray[p] = np.NAN
        HWHMarray = HWHMarray[~np.isnan(HWHMarray)]
        return HWHMarray

    def reject_outliers(
        self,
        data,
        k=4,
        return_indices=False,
        q1=None,
        q2=None,
        IQR=None,
        column_name="sum_intensity_in_photons",
    ):
        """
        Rejects outliers from data using the IQR method. Can return filtered data or indices.

        Args:
            data (np.ndarray or pl.DataFrame): Data array or DataFrame.
            k (float): Multiplier for the IQR to determine outliers.
            return_indices (bool): If True, return indices of outliers instead of filtered data.
            q1 (float, optional): Precomputed first quartile. If None, it will be computed.
            q2 (float, optional): Precomputed third quartile. If None, it will be computed.
            IQR (float, optional): Precomputed interquartile range. If None, it will be computed.
            column_name (str): Column name to use if data is a DataFrame.

        Returns:
            np.ndarray or pl.DataFrame: Filtered data or indices of outliers.
        """
        if isinstance(data, pl.DataFrame):
            data_array = data[column_name].to_numpy()
        else:
            data_array = data

        if q1 is None or q2 is None or IQR is None:
            q1, q2 = np.percentile(data_array, [25, 75])
            IQR = q2 - q1

        lower_limit = q1 - k * IQR
        upper_limit = q2 + k * IQR

        if return_indices:
            indices = np.arange(len(data_array))
            outlier_indices = np.hstack(
                [indices[data_array >= upper_limit], indices[data_array <= lower_limit]]
            )
            return outlier_indices
        else:
            if isinstance(data, pl.DataFrame):
                filtered_data = data.filter(
                    (pl.col(column_name) >= lower_limit)
                    & (pl.col(column_name) <= upper_limit)
                )
            else:
                filtered_data = data_array[
                    (data_array >= lower_limit) & (data_array <= upper_limit)
                ]
            return filtered_data, q1, q2, IQR

    def calculate_rdf_with_thresholds(
        self,
        analysis_data_1,
        analysis_data_2=None,
        mask_files=None,
        threshold_1=0,
        threshold_2=0,
        pixel_size=0.11,
        dr=1.0,
        analysis_type="single_channel",
        image_size=(1200, 1200),
    ):
        """
        Calculate RDF (Radial Distribution Function) with thresholds for different analysis types.

        Args:
            analysis_data_1 (pl.DataFrame): The analysis data of puncta set 1.
            analysis_data_2 (pl.DataFrame, optional): The analysis data of puncta set 2. Default is None.
            mask_files (array, optional): The mask file locations. Default is None. If analysis_type is "spot_mask", has to be same shape as analysis data filenames.
            threshold_1 (float): The photon threshold for puncta set 1.
            threshold_2 (float, optional): The photon threshold for puncta set 2. Default is None.
            pixel_size (float): Size of pixels.
            dr (float): Step for radial distribution function.
            aboveT (int): Do the calculation above or below threshold.
            analysis_type (str): Type of analysis to perform. Options are "two_channels", "single_channel", "spot_mask".
            image_size (tuple): Size of the image.

        Returns:
            rdf_analysis (pl.DataFrame): Polars DataFrame of the RDF analysis.
        """
        if analysis_type == "spot_mask":
            files = np.unique(analysis_data_1["image_filename"].to_numpy())
            if len(files) != len(mask_files):
                print("Number of image files must correspond to number of mask files.")
                return

        def filter_data(data, threshold):
            return data.filter(pl.col("sum_intensity_in_photons") > threshold)

        def harmonise_rdfs(g_r, radii):
            minmax = np.zeros([2, len(g_r.keys())])
            for i, key in enumerate(radii):
                minmax[:, i] = [np.min(radii[key]), np.max(radii[key])]
            overall_min = np.max(minmax[0, :])
            overall_max = np.min(minmax[1, :])
            for i, key in enumerate(radii):
                r_from = np.argmin(np.abs(radii[key] - overall_min))
                r_to = np.argmin(np.abs(radii[key] - overall_max))
                g_r[key] = g_r[key][r_from:r_to]

            radii = radii[key][r_from:r_to]
            g_r_overall = np.zeros([len(radii), len(g_r.keys())])
            for i, uid in enumerate(g_r.keys()):
                g_r_overall[:, i] = g_r[uid]

            g_r_overall[g_r_overall == 0] = np.NAN
            g_r_mean = np.nanmean(g_r_overall, axis=1)
            g_r_std = np.nanstd(g_r_overall, axis=1)

            return radii, g_r_mean, g_r_std

        def calculate_rdf_for_file_rdf(coordinates, pixel_size, dr):
            # Radial Distribution Function Calculation for spots
            g_r, radii = rdf(np.multiply(coordinates, pixel_size), dr=dr)
            return g_r, radii

        def calculate_rdf_for_file_spot_mask(
            coordinates_spot, coordinates_mask, pixel_size, dr, image_size
        ):
            # Radial Distribution Function Calculation for spots and mask
            r_max = np.divide(np.multiply(np.max(image_size), pixel_size), 4.0)
            g_r, radii = MultiD_RD_functions.multid_rdf(
                coordinates_spot * pixel_size,
                coordinates_mask * pixel_size,
                r_max,
                dr,
                boxdims=(
                    [
                        [0.0, 0.0],
                        [image_size[0] * pixel_size, image_size[1] * pixel_size],
                    ]
                ),
                parallel=True,
            )
            return g_r, radii

        start = time.time()

        analysis_data_1 = filter_data(analysis_data_1, threshold_1)
        if analysis_data_2 is not None:
            analysis_data_2 = filter_data(analysis_data_2, threshold_2)

        if len(analysis_data_1) == 0 or (
            analysis_data_2 is not None and len(analysis_data_2) == 0
        ):
            return np.NAN

        files_1 = np.unique(analysis_data_1["image_filename"].to_numpy())
        files_2 = (
            np.unique(analysis_data_2["image_filename"].to_numpy())
            if analysis_data_2 is not None
            else []
        )
        files = np.unique(np.hstack([files_1, files_2]))

        g_r = {}
        radii = {}

        for i, file in enumerate(files):
            if analysis_type == "two_channels":
                subset_1 = analysis_data_1.filter(pl.col("image_filename") == file)
                subset_2 = analysis_data_2.filter(pl.col("image_filename") == file)
                for z in np.unique(
                    np.hstack([subset_1["z"].to_numpy(), subset_2["z"].to_numpy()])
                ):
                    uid = str(file) + "___" + str(z)
                    filtered_subset_1 = subset_1.filter(pl.col("z") == z)
                    filtered_subset_2 = subset_2.filter(pl.col("z") == z)
                    coordinates_1 = np.vstack(
                        [
                            filtered_subset_1["x"].to_numpy(),
                            filtered_subset_1["y"].to_numpy(),
                        ]
                    ).T
                    coordinates_2 = np.vstack(
                        [
                            filtered_subset_2["x"].to_numpy(),
                            filtered_subset_2["y"].to_numpy(),
                        ]
                    ).T
                    if len(coordinates_1) > 0 and len(coordinates_2) > 0:
                        g_r[uid], radii[uid] = calculate_rdf_for_file_spot_mask(
                            coordinates_1, coordinates_2, pixel_size, dr, image_size
                        )
            elif analysis_type == "single_channel":
                subset = analysis_data_1.filter(pl.col("image_filename") == file)
                for z in np.unique(subset["z"].to_numpy()):
                    uid = str(file) + "___" + str(z)
                    subset_filter = subset.filter(pl.col("z") == z)
                    coordinates = np.vstack(
                        [subset_filter["x"].to_numpy(), subset_filter["y"].to_numpy()]
                    ).T
                    g_r[uid], radii[uid] = calculate_rdf_for_file_rdf(
                        coordinates, pixel_size, dr
                    )
            elif analysis_type == "spot_mask":
                cell_mask = IO.read_tiff(
                    os.path.join(os.path.dirname(file), mask_files[i])
                )
                subset = analysis_data_1.filter(pl.col("image_filename") == file)
                for z in np.unique(subset["z"].to_numpy()):
                    uid = str(file) + "___" + str(z)
                    filtered_subset = subset.filter(pl.col("z") == z)
                    x = filtered_subset["x"].to_numpy()
                    y = filtered_subset["y"].to_numpy()
                    coordinates_spot = np.vstack([x, y]).T
                    xm, ym = np.where(cell_mask[:, :, int(z) - 1])
                    coordinates_mask = np.vstack([xm, ym]).T
                    if len(coordinates_mask) > 0:
                        g_r[uid], radii[uid] = calculate_rdf_for_file_spot_mask(
                            coordinates_spot,
                            coordinates_mask,
                            pixel_size,
                            dr,
                            image_size,
                        )
            print(
                f"Computing RDF {analysis_type}     File {i + 1}/{len(files)}    Time elapsed: {time.time() - start:.3f} s",
                end="\r",
                flush=True,
            )

        radii, g_r_mean, g_r_std = harmonise_rdfs(g_r, radii)
        data = {"radii": radii, "g_r_mean": g_r_mean, "g_r_std": g_r_std}

        return pl.DataFrame(data)

    def filter_lo_intensity(self, lo_data, lo_mask, z_planes):
        """
        Filters lo mask, and polars dataframe, based on intensity.

        Args:
            lo_data (pl.DataFrame): The analysis file of large objects. Already thresholded (code assumes)
            lo_mask (np.ndarray): large object mask
        Returns:
            lo_data (pl.DataFrame): polars dataframe of the lo analysis
            lo_mask (np.ndarray): numpy array of lo mask
        """
        lo_mask_new = copy(lo_mask)
        for z_plane in np.asarray(z_planes, dtype=int):
            lo_tocheck = lo_data.filter(pl.col("z") == z_plane)
            pil, areas, centroids, _, _ = IA_F.calculate_region_properties(
                lo_mask[z_plane - 1, :, :]
            )
            for lo in np.arange(len(pil)):
                area = areas[lo]
                x = centroids[lo, 0]
                y = centroids[lo, 1]
                if (
                    ~np.any(np.isclose(area, lo_tocheck["area"].to_numpy(), atol=0.1))
                    and ~np.any(np.isclose(x, lo_tocheck["x"].to_numpy(), atol=0.1))
                    and ~np.any(np.isclose(y, lo_tocheck["y"].to_numpy(), atol=0.1))
                ):
                    lo_mask_new[z_plane - 1, pil[lo][:, 0], pil[lo][:, 1]] = 0
        return lo_mask_new

    def filter_cellmask_intensity(self, cell_mask, cell_image, intensity_threshold):
        """
        Filters cell mask, based on intensity.

        Args:
            cell_mask (np.ndarray): The cell mask.
            cell_image (np.ndarray): Raw cell image
        Returns:
            new_cell_mask (np.ndarray): numpy array of cell mask
        """
        new_cell_mask = copy(cell_mask)
        z_planes = cell_mask.shape[0]
        for z in np.arange(z_planes):
            binary_mask = cell_mask[z, :, :]
            image = cell_image[z, :, :]
            pil, areas, centroids, sum_intensity, mean_intensity = (
                IA_F.calculate_region_properties(binary_mask, image=image)
            )
            for i in np.arange(len(mean_intensity)):
                if mean_intensity[i] < intensity_threshold:
                    new_cell_mask[z, pil[i][:, 0], pil[i][:, 0]] = 0
        return new_cell_mask

    def colocalise_with_threshold(
        self,
        analysis_file,
        threshold,
        protein_string,
        lo_string,
        cell_string,
        analysis_type="spot_to_cell",
        imtype=".tif",
        blur_degree=1,
        calc_clr=False,
        aboveT=1,
        lower_cell_size_threshold=0,
        upper_cell_size_threshold=np.inf,
        lower_lo_size_threshold=0,
        upper_lo_size_threshold=np.inf,
        z_project_first=[False, False],
        cell_threshold=0,
    ):
        """
        Does colocalisation analysis of spots vs mask with an additional threshold.

        Args:
            analysis_file (string): The analysis file of puncta. If large objects are to be
                                    analysed as puncta, make sure this is a large object file.
            threshold (float): The photon threshold for puncta/large objects.
            protein_string (str): string of protein images
            lo_string (str): string of larger object images
            cell_string (str): string of cell images
            analysis_type (str, optional): Type of analysis to perform. Options are:
                - "spot_to_cell": Calculate spot to cell metrics.
                - "lo_to_cell": Calculate lo to cell metrics.
                - "lo_to_spot": Calculate lo to spot metrics.
                Default is "spot_to_cell".

            imtype (str): image end string
            blur_degree (int): degree of blur to apply to puncta
            calc_clr (bool): calculate clr yes/no
            aboveT (int): do the calculation above or below threshold
            lower_cell_size_threshold (float): lower threshold of cell size
            upper_cell_size_threshold (float): upper threshold of cell size
            lower_lo_size_threshold (float): lower threshold of lo size
            upper_lo_size_threshold (float): upper threshold of lo size
            z_project_first (list of boolean): z project instructions for cell/protein size threshold
            cell_threshold (float): intensity value cell should be above

        Returns:
            lo_analysis (pl.DataFrame): polars dataframe of the cell analysis
            spot_analysis (pl.DataFrame): polars dataframe of the spot analysis
        """
        end_str = {
            "spot_to_cell": f"{cell_string}_cellMask.tiff",
            "lo_to_spot": f"{lo_string}_loMask.tiff",
            "lo_to_cell": f"{lo_string}_loMask.tiff",
        }.get(analysis_type, None)

        spots_with_intensities = pl.read_csv(analysis_file)
        if analysis_type == "lo_to_cell":
            column = "mean_intensity_in_photons"
        else:
            column = "sum_intensity_in_photons"

        if (
            analysis_type == "lo_to_cell"
            and column not in spots_with_intensities.columns
        ):
            print("Large object analysis file not loaded in. Code will fail.")
            return

        condition = (
            (pl.col(column) > threshold) if aboveT else (pl.col(column) <= threshold)
        )
        spots_with_intensities = spots_with_intensities.filter(condition)

        if len(spots_with_intensities) == 0:
            return np.NAN, np.NAN

        analysis_directory = os.path.split(analysis_file)[0]
        image_filenames = np.unique(spots_with_intensities["image_filename"].to_numpy())
        columns = (
            [
                "clr",
                "norm_std",
                "norm_CSR",
                "expected_spots",
                "coincidence",
                "chance_coincidence",
                "n_iter",
                "z_plane",
                "image_filename",
            ]
            if calc_clr
            else [
                "coincidence",
                "chance_coincidence",
                "n_iter",
                "z_plane",
                "image_filename",
            ]
        )

        start = time.time()
        lo_analysis, spot_analysis = None, None

        for i, image in enumerate(image_filenames):
            raw_data_path = os.path.split(image)[0]
            common_path = os.path.split(image.split(imtype)[0])[-1].split(
                protein_string
            )[0]
            image_file = spots_with_intensities.filter(
                pl.col("image_filename") == image
            )
            if analysis_type in ["lo_to spot", "lo_to_cell"]:
                image_file = image_file.filter(
                    pl.col("area") >= lower_lo_size_threshold
                )
                image_file = image_file.filter(pl.col("area") < upper_lo_size_threshold)
            z_planes = np.unique(image_file["z"].to_numpy())
            if len(z_planes) > 0:
                lo_mask = self._read_mask(
                    analysis_directory, common_path, end_str, lo_string
                )
                if analysis_type not in ["lo_to_spot", "lo_to_cell"]:
                    l_thresh = lower_cell_size_threshold
                    u_thresh = upper_cell_size_threshold
                else:
                    l_thresh = lower_lo_size_threshold
                    u_thresh = upper_lo_size_threshold
                lo_mask, _, _, _ = self.threshold_cell_areas(
                    lo_mask,
                    lower_cell_size_threshold=l_thresh,
                    upper_cell_size_threshold=u_thresh,
                    z_project=z_project_first,
                )
                if (threshold > 0) and analysis_type == "lo_to_cell":
                    lo_mask = self.filter_lo_intensity(image_file, lo_mask, z_planes)
                if len(lo_mask.shape) > 2:
                    image_size = lo_mask.shape[1:]
                else:
                    image_size = lo_mask.shape
                    z_planes = image_file.replace_column(
                        image_file.get_column_index("z"),
                        pl.Series("z", np.ones_like(image_file["z"].to_numpy())),
                    )
                    z_planes = np.unique(image_file["z"].to_numpy())
                if end_str != f"{cell_string}_cellMask.tiff":
                    raw_cell_data = IO.read_tiff(
                        os.path.join(raw_data_path, common_path + cell_string + ".tif")
                    )
                    cell_mask, _, _, _ = self.threshold_cell_areas(
                        self._read_mask(
                            analysis_directory,
                            common_path,
                            f"{cell_string}_cellMask.tiff",
                            cell_string,
                        ),
                        lower_cell_size_threshold=lower_cell_size_threshold,
                        upper_cell_size_threshold=upper_cell_size_threshold,
                        z_project=z_project_first,
                    )
                    cell_mask = self.filter_cellmask_intensity(
                        cell_mask, raw_cell_data, cell_threshold
                    )
                    if cell_threshold > 0:
                        file_path = os.path.join(
                            analysis_directory,
                            common_path
                            + cell_string
                            + "_it_"
                            + str(cell_threshold).replace(".", "p")
                            + "_lst_"
                            + str(lower_cell_size_threshold).replace(".", "p")
                            + ".tif",
                        )
                        IO.write_tiff(cell_mask, file_path, bit=np.uint8)
                if analysis_type != "lo_to_cell":
                    if calc_clr:
                        dataarray, raw_colocalisation = self._process_spots_parallel(
                            image_file,
                            z_planes,
                            lo_mask,
                            image_size,
                            self._parallel_coloc_per_z_clr_spot,
                            blur_degree,
                        )
                    else:
                        dataarray, raw_colocalisation = self._process_spots_parallel(
                            image_file,
                            z_planes,
                            lo_mask,
                            image_size,
                            self._parallel_coloc_per_z_noclr_spot,
                            blur_degree,
                        )
                else:
                    dataarray, raw_colocalisation = self._process_masks_parallel(
                        z_planes,
                        lo_mask,
                        cell_mask,
                        image_size,
                        self._parallel_coloc_per_z_los,
                    )

                image_file = image_file.with_columns(incell=raw_colocalisation)
                dataarray = np.vstack(
                    [
                        np.asarray(dataarray, dtype="object"),
                        np.repeat(image, len(z_planes)),
                    ]
                )

                lo_analysis = np.array(
                    dataarray
                    if lo_analysis is None
                    else np.hstack([lo_analysis, dataarray])
                )
                spot_analysis = (
                    image_file
                    if spot_analysis is None
                    else pl.concat([spot_analysis, image_file])
                )
            print(
                f"Computing colocalisation     File {i + 1}/{len(image_filenames)}    Time elapsed: {time.time() - start:.3f} s",
                end="\r",
                flush=True,
            )
        lo_save = {}
        for i, column in enumerate(columns):
            lo_save[column] = lo_analysis[i, :]
        df = H_F.clean_database(pl.DataFrame(data=lo_save), columns)
        return df, spot_analysis

    def _read_mask(self, analysis_directory, common_path, end_str, default_str):
        mask_path = os.path.join(
            analysis_directory,
            common_path + end_str if end_str else f"{default_str}_loMask.tiff",
        )
        return IO.read_tiff(mask_path) if os.path.exists(mask_path) else None

    def _process_spots_parallel(
        self, image_file, z_planes, lo_mask, image_size, parallel_func, blur_degree
    ):
        xcoords = [
            image_file.filter(pl.col("z") == z_plane)["x"].to_numpy()
            for z_plane in z_planes
        ]
        ycoords = [
            image_file.filter(pl.col("z") == z_plane)["y"].to_numpy()
            for z_plane in z_planes
        ]
        masks = [
            (
                lo_mask[int(z_plane) - 1, :, :]
                if lo_mask.shape[0] > z_planes[-1]
                else lo_mask[j, :, :]
            )
            for j, z_plane in enumerate(z_planes)
        ]
        image_sizes = [image_size for z_plane in z_planes]
        blur_degrees = [blur_degree for z_plane in z_planes]

        pool = Pool(nodes=self.cpu_number)
        pool.restart()
        results = pool.map(
            parallel_func, xcoords, ycoords, masks, image_sizes, blur_degrees
        )
        pool.close()
        pool.terminate()

        coincidence = np.array([i[0] for i in results])
        chance_coincidence = np.array([i[1] for i in results])
        raw_colocalisation = np.concatenate([i[2] for i in results])
        n_iter = np.array([i[3] for i in results])

        dataarray = np.vstack([coincidence, chance_coincidence, n_iter, z_planes])
        if parallel_func == self._parallel_coloc_per_z_clr_spot:
            clr = np.array([i[0] for i in results])
            norm_std = np.array([i[1] for i in results])
            norm_CSR = np.array([i[2] for i in results])
            expected_spots = np.array([i[3] for i in results])
            dataarray = np.vstack(
                [
                    clr,
                    norm_std,
                    norm_CSR,
                    expected_spots,
                    coincidence,
                    chance_coincidence,
                    n_iter,
                    z_planes,
                ]
            )

        return dataarray, raw_colocalisation

    def _process_masks_parallel(
        self, z_planes, lo_mask, cell_mask, image_size, parallel_func
    ):
        masks_lo = [
            (
                lo_mask[int(z_plane) - 1, :, :]
                if lo_mask.shape[0] > len(z_planes)
                else lo_mask[j, :, :]
            )
            for j, z_plane in enumerate(z_planes)
        ]
        masks_cell = [
            (
                cell_mask[int(z_plane) - 1, :, :]
                if lo_mask.shape[0] > len(z_planes)
                else cell_mask[j, :, :]
            )
            for j, z_plane in enumerate(z_planes)
        ]
        image_sizes = [image_size for z_plane in z_planes]

        pool = Pool(nodes=self.cpu_number)
        pool.restart()
        results = pool.map(parallel_func, masks_lo, masks_cell, image_sizes)
        pool.close()
        pool.terminate()

        coincidence = np.array([i[0] for i in results])
        chance_coincidence = np.array([i[1] for i in results])
        raw_colocalisation = np.concatenate([i[2] for i in results])
        n_iter = np.array([i[3] for i in results])

        dataarray = np.vstack([coincidence, chance_coincidence, n_iter, z_planes])
        return dataarray, raw_colocalisation

    def _parallel_coloc_per_z_clr_spot(
        self, xcoords, ycoords, mask, image_size, blur_degree
    ):
        centroids = np.asarray(np.vstack([xcoords, ycoords]), dtype=int).T
        mask_indices = self.generate_indices(mask, image_size, is_mask=True)
        spot_indices = self.generate_indices(centroids, image_size)
        return C_F.calculate_coincidence(
            spot_indices,
            mask_indices,
            image_size,
            blur_degree=blur_degree,
            analysis_type="colocalisation_likelihood",
        )

    def _parallel_coloc_per_z_noclr_spot(
        self, xcoords, ycoords, mask, image_size, blur_degree
    ):
        centroids = np.asarray(np.vstack([xcoords, ycoords]), dtype=int).T
        mask_indices = self.generate_indices(mask, image_size, is_mask=True)
        spot_indices = self.generate_indices(centroids, image_size)
        return C_F.calculate_coincidence(
            spot_indices,
            mask_indices,
            image_size,
            blur_degree=blur_degree,
            analysis_type="spot_to_mask",
        )

    def _parallel_coloc_per_z_los(self, mask_lo, mask_cell, image_size):
        mask_lo_indices, n_largeobjs = self.generate_indices(
            mask_lo, image_size, is_mask=True, is_lo=True
        )
        mask_cell_indices = self.generate_indices(mask_cell, image_size, is_mask=True)
        return C_F.calculate_coincidence(
            spot_indices=None,
            largeobj_indices=mask_lo_indices,
            mask_indices=mask_cell_indices,
            n_largeobjs=n_largeobjs,
            image_size=image_size,
            analysis_type="largeobj",
        )

    def number_of_puncta_per_segmented_cell_with_threshold(
        self,
        analysis_file,
        analysis_data_raw,
        threshold,
        lower_cell_size_threshold=100,
        upper_cell_size_threshold=np.inf,
        blur_degree=1,
        cell_string="C0",
        protein_string="C1",
        imtype=".tif",
        z_project_first=[True, True],
        median=None,
    ):
        """
        Does analysis of number of oligomers in a mask area per "segmented" cell area.

        Args:
            analysis_file (pl.DataFrame): The analysis file location of puncta set 1.
            threshold (float): The photon threshold for puncta set 1.
            lower_cell_size_threshold (float): lower cell size threshold
            upper_cell_size_threshold (float): upper cell size threshold
            out_cell (boolean): exclude puncta that are inside cells
            pixel_size (float): size of pixels
            blur_degree (int): degree to blur spots
            cell_string (string): will use this to find corresponding cell files
            protein_string (string): will use this to find corresponding files
            imtype (string): image type previously analysed
            aboveT (int): do the calculation above or below threshold
            z_project_first (boolean, boolean): if both True (default), does a z
                                    projection before
                                    thresholding cell size.
                                    If both false, does not z project and
                                    then does the analysis per z plane.

        Returns:
            cell_punctum_analysis (pl.DataFrame): polars datarray of the cell analysis
        """
        C_F = CoincidenceFunctions.Coincidence_Functions()
        filter_op = ">"
        analysis_data = analysis_data_raw.filter(
            eval(f"pl.col('sum_intensity_in_photons') {filter_op} threshold")
        )
        del analysis_data_raw
        typestr = "> threshold"
        analysis_string = (
            " protein cell load " if median else " puncta cell likelihood "
        )
        analysis_type = "protein_load" if median else "spot_to_cell"
        analysis_directory = os.path.split(analysis_file)[0]

        if len(analysis_data) == 0:
            return np.NAN

        files = np.unique(analysis_data["image_filename"].to_numpy())
        cell_punctum_analysis = None
        start = time.time()

        for i, file in enumerate(files):
            cell_file = os.path.join(
                analysis_directory,
                os.path.split(file.split(imtype)[0])[-1].split(protein_string)[0]
                + str(cell_string)
                + "_cellMask.tiff",
            )
            if not os.path.isfile(cell_file):
                continue

            raw_cell_mask = IO.read_tiff(cell_file)
            subset = analysis_data.filter(pl.col("image_filename") == file)
            cell_mask, pil_mask, centroids, areas = self.threshold_cell_areas(
                raw_cell_mask,
                lower_cell_size_threshold,
                upper_cell_size_threshold=upper_cell_size_threshold,
                z_project=z_project_first,
            )
            image_size = (
                cell_mask.shape if len(cell_mask.shape) < 3 else cell_mask.shape[1:]
            )
            x, y = subset["x"].to_numpy(), subset["y"].to_numpy()
            bounds = (x < image_size[0]) & (x >= 0) & (y < image_size[1]) & (y >= 0)
            x, y = x[bounds], y[bounds]
            centroids_puncta = np.asarray(np.vstack([x, y]), dtype=int)
            spot_indices, filter_array = np.unique(
                np.ravel_multi_index(centroids_puncta, image_size, order="F"),
                return_index=True,
            )
            intensity = (
                subset["sum_intensity_in_photons"].to_numpy()[bounds][filter_array]
                if median
                else None
            )
            filename_tosave = np.full_like(centroids[:, 0], file, dtype="object")

            def areaanalysis(coords):
                xm, ym = coords[:, 0], coords[:, 1]
                if np.any(xm > image_size[0]) or np.any(ym > image_size[1]):
                    return np.NAN, np.NAN
                coordinates_mask = np.asarray(np.vstack([xm, ym]), dtype=int)
                mask_indices = np.ravel_multi_index(
                    coordinates_mask, image_size, order="F"
                )
                olig_cell_ratio, n_olig_in_cell, _ = C_F.calculate_coincidence(
                    spot_indices=spot_indices,
                    mask_indices=mask_indices,
                    spot_intensities=intensity,
                    median_intensity=median,
                    image_size=image_size,
                    n_iter=1,
                    blur_degree=blur_degree,
                    analysis_type=analysis_type,
                )
                return olig_cell_ratio, n_olig_in_cell

            with Pool(nodes=self.cpu_number) as pool:
                results = pool.map(areaanalysis, pil_mask)

            n_cell_ratios = np.array([r[0] for r in results])
            n_spots_in_object = np.array([r[1] for r in results])

            if len(areas) > 0:
                data = {
                    "area/pixels": areas,
                    "x_centre": centroids[:, 0],
                    "y_centre": centroids[:, 1],
                    (
                        "puncta_cell_likelihood"
                        if median is None
                        else "cell_protein_load"
                    ): n_cell_ratios,
                    "n_puncta_in_cell": n_spots_in_object,
                    "image_filename": filename_tosave,
                }

                cell_punctum_analysis = (
                    pl.concat([cell_punctum_analysis, pl.DataFrame(data)])
                    if cell_punctum_analysis is not None
                    else pl.DataFrame(data)
                )

            print(
                f"Computing {typestr} {analysis_string} File {i + 1}/{len(files)}    Time elapsed: {time.time() - start:.3f} s",
                end="\r",
                flush=True,
            )

        return cell_punctum_analysis.rechunk()

    def colocalise_spots_with_threshold(
        self,
        spots_1_with_intensities,
        spots_2_with_intensities,
        threshold_1,
        threshold_2,
        spot_1_string,
        spot_2_string,
        imtype=".tif",
        image_size=(1200, 1200),
        blur_degree=1,
        aboveT=1,
    ):
        """
        Redo colocalisation analayses of spots above a photon threshold in an
        analysis file, to spots above a second threshold in a separate analysis file.

        Args:
            spots_1_with_intensities (pl.DataFrame): Analysis file (channel 1) to be re-done.
            analysis_2_data (pl.DataFrame): Analysis file (channel 2) to be re-done.
            threshold_1 (float): The photon threshold for channel 1
            threshold_2 (float): The photon threshold for channel 2
            spot_1_string (str): string of spot 1
            spot_2_string (str): string of spot 2
            imtype (str): image type
            image_size (list): original image size
            blur_degree (int): blur degree for colocalisation analysis
            aboveT (int): do the calculation above or below threshold
        """
        filter_op = ">" if aboveT == 1 else "<="
        typestr = "> threshold" if aboveT == 1 else "<= threshold"
        spots_1_with_intensities = spots_1_with_intensities.filter(
            eval(f"pl.col('sum_intensity_in_photons') {filter_op} threshold_1")
        )
        spots_2_with_intensities = spots_2_with_intensities.filter(
            eval(f"pl.col('sum_intensity_in_photons') {filter_op} threshold_2")
        )

        overall_filenames = np.unique(
            np.hstack(
                [
                    [
                        i.split(imtype)[0].split(spot_1_string)[0]
                        for i in spots_1_with_intensities["image_filename"].to_numpy()
                    ],
                    [
                        i.split(imtype)[0].split(spot_2_string)[0]
                        for i in spots_2_with_intensities["image_filename"].to_numpy()
                    ],
                ]
            )
        )
        columns = ["coincidence", "chance_coincidence", "z", "image_filename"]

        if len(overall_filenames) == 0:
            return None, None, None, None
        start = time.time()

        # TODO: add in case for where no spots in first image
        for i, image in enumerate(overall_filenames):
            image_1_file = spots_1_with_intensities.filter(
                pl.col("image_filename") == image + spot_1_string + imtype
            )

            image_2_file = spots_2_with_intensities.filter(
                pl.col("image_filename") == image + spot_2_string + imtype
            )
            if (len(image_1_file) > 0) & (len(image_2_file) > 0):
                z_planes = np.intersect1d(
                    image_1_file["z"].to_numpy(), image_2_file["z"].to_numpy()
                )

                image_1_file = image_1_file.filter(pl.col("z").is_in(z_planes))
                image_2_file = image_2_file.filter(pl.col("z").is_in(z_planes))

                dataarray_1 = np.zeros([len(z_planes), len(columns)])
                dataarray_2 = np.zeros([len(z_planes), len(columns)])

                dataarray_1[:, 2] = z_planes
                dataarray_2[:, 2] = z_planes

                temp_1_pl = pl.DataFrame(data=dataarray_1, schema=columns)
                temp_2_pl = pl.DataFrame(data=dataarray_2, schema=columns)

                for j, z_plane in enumerate(z_planes):
                    x_1_coords = image_1_file.filter(pl.col("z") == z_plane)[
                        "x"
                    ].to_numpy()
                    y_1_coords = image_1_file.filter(pl.col("z") == z_plane)[
                        "y"
                    ].to_numpy()

                    x_2_coords = image_2_file.filter(pl.col("z") == z_plane)[
                        "x"
                    ].to_numpy()
                    y_2_coords = image_2_file.filter(pl.col("z") == z_plane)[
                        "y"
                    ].to_numpy()

                    centroids1 = np.asarray(
                        np.vstack([x_1_coords, y_1_coords]), dtype=int
                    ).T
                    centroids2 = np.asarray(
                        np.vstack([x_2_coords, y_2_coords]), dtype=int
                    ).T

                    spot_1_indices = self.generate_indices(centroids1, image_size)
                    spot_2_indices = self.generate_indices(centroids2, image_size)
                    (
                        temp_1_pl[j, 0],
                        temp_1_pl[j, 1],
                        temp_2_pl[j, 0],
                        temp_2_pl[j, 1],
                        raw_1_coincidence,
                        raw_2_coincidence,
                    ) = C_F.calculate_coincidence(
                        spot_indices=spot_1_indices,
                        mask_indices=None,
                        second_spot_indices=spot_2_indices,
                        image_size=image_size,
                        n_iter=100,
                        blur_degree=blur_degree,
                        analysis_type="spot_to_spot",
                    )

                    if j == 0:
                        rc1 = raw_1_coincidence
                        rc2 = raw_2_coincidence
                    else:
                        rc1 = np.hstack([rc1, raw_1_coincidence])
                        rc2 = np.hstack([rc2, raw_2_coincidence])

                image_1_file = image_1_file.with_columns(channelcol=rc1).rename(
                    {"channelcol": "coincidence_with_channel_" + spot_2_string}
                )
                image_2_file = image_2_file.with_columns(channelcol=rc2).rename(
                    {"channelcol": "coincidence_with_channel_" + spot_1_string}
                )

                temp_1_pl = temp_1_pl.with_columns(
                    image_filename=np.full_like(
                        z_planes, image + spot_1_string + imtype, dtype="object"
                    )
                )

                temp_2_pl = temp_2_pl.with_columns(
                    image_filename=np.full_like(
                        z_planes, image + spot_2_string + imtype, dtype="object"
                    )
                )
                if i == 0:
                    plane_1_analysis = temp_1_pl
                    plane_2_analysis = temp_2_pl
                    spot_1_analysis = image_1_file
                    spot_2_analysis = image_2_file
                else:
                    plane_1_analysis = pl.concat([plane_1_analysis, temp_1_pl])
                    plane_2_analysis = pl.concat([plane_2_analysis, temp_2_pl])
                    spot_1_analysis = pl.concat([spot_1_analysis, image_1_file])
                    spot_2_analysis = pl.concat([spot_2_analysis, image_2_file])
                print(
                    "Computing "
                    + typestr
                    + " spot-to-spot coincidence     File {}/{}    Time elapsed: {:.3f} s".format(
                        i + 1, len(overall_filenames), time.time() - start
                    ),
                    end="\r",
                    flush=True,
                )
        return (plane_1_analysis, plane_2_analysis, spot_1_analysis, spot_2_analysis)

    def threshold_cell_areas(
        self,
        cell_mask_raw,
        lower_cell_size_threshold=100,
        upper_cell_size_threshold=np.inf,
        z_project=[True, True],
    ):
        """
        Removes small and/or large objects from a cell mask.

        Args:
            cell_mask_raw (np.ndarray): Cell mask object
            lower_cell_size_threshold (float): Lower size threshold
            upper_cell_size_threshold (float): Upper size threshold
            z_project (bool): If True, z-projects cell mask

        Returns:
            tuple: Processed cell mask, pixel image locations, centroids, areas
        """
        # Z-project if needed
        if z_project[0] and len(cell_mask_raw.shape) > 2:
            cell_mask = np.sum(cell_mask_raw, axis=0).clip(0, 1)
        else:
            cell_mask = cell_mask_raw.copy()
        # Handle multi-dimensional and 2D masks differently
        if len(cell_mask.shape) > 2:
            # Process 3D mask
            cell_mask_new = cell_mask.copy()
            for plane in range(cell_mask.shape[0]):
                plane_mask = cell_mask_new[plane, :, :]
                pil, areas, centroids, _, _ = IA_F.calculate_region_properties(
                    plane_mask
                )
                # Vectorized filtering
                mask = (areas >= lower_cell_size_threshold) & (
                    areas < upper_cell_size_threshold
                )
                # Update mask
                for c in np.where(~mask)[0]:
                    cell_mask_new[plane, pil[c][:, 0], pil[c][:, 1]] = 0
            # Reconstruct final mask
            if z_project[1]:
                cell_mask_new = np.sum(cell_mask_new, axis=0).clip(0, 1)
        else:
            # Process 2D mask
            cell_mask_new = cell_mask.copy()
            pil, areas, centroids, _, _ = IA_F.calculate_region_properties(
                cell_mask_new
            )
            # Vectorized filtering
            mask = (areas >= lower_cell_size_threshold) & (
                areas < upper_cell_size_threshold
            )
            # Update mask
            for c in np.where(~mask)[0]:
                cell_mask_new[pil[c][:, 0], pil[c][:, 1]] = 0
        # Final region properties calculation
        if z_project[1]:
            pil, areas, centroids, _, _ = IA_F.calculate_region_properties(
                cell_mask_new
            )
        else:
            pil = None
            areas = None
            centroids = None
        return cell_mask_new, pil, centroids, areas

    def create_labelled_cellmasks(
        self,
        cell_analysis,
        puncta,
        cell_mask,
        lower_cell_size_threshold=100,
        upper_cell_size_threshold=np.inf,
        z_project=[True, True],
        parameter="n_puncta_in_cell",
    ):
        """
        create_labelled_cellmasks plots values of specified parameter in cell objects

        Args:
            cell_analysis (polars dataarray): cell analysis
            puncta (polars dataarray): puncta analysis
            cell_mask (np.2darray): cell mask object
            lower_cell_size_threshold (float): how big of an area do you take as lower threshold
            upper_cell_size_threshold (float): how big of an area do you take as upper threshold
            z_project (boolean): project z first or not for cell mask
            parameter (string): parameter to plot

        Returns:
            cell_mask_toplot_analysis (np.2darray): cell mask of analysed quantity
            new_cell_mask (np.2darray): thresholded cell mask

        """
        new_cell_mask, pil, centroids, areas = self.threshold_cell_areas(
            cell_mask,
            lower_cell_size_threshold,
            upper_cell_size_threshold=upper_cell_size_threshold,
            z_project=z_project,
        )

        cell_mask_toplot_analysis = copy(new_cell_mask)

        analysis = copy(cell_analysis)

        for i, mask in enumerate(pil):
            centroid = centroids[i]
            index_to_use = np.argmin(
                np.sqrt(
                    np.square(
                        centroid
                        - np.vstack(
                            [
                                analysis["x_centre"].to_numpy(),
                                analysis["y_centre"].to_numpy(),
                            ]
                        )
                    )
                )
            )
            cell_mask_toplot_analysis[mask[:, 0], mask[:, 1]] = analysis[
                parameter
            ].to_numpy()[index_to_use]
        return (
            cell_mask_toplot_analysis,
            new_cell_mask,
        )
