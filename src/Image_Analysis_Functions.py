# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to analysis of images based on their
radiality, relating to the RASP concept.
jsb92, 2024/01/02
"""
import numpy as np
from scipy.signal.windows import gaussian as gauss
from scipy.signal import fftconvolve
import skimage as ski
from skimage.filters import gaussian, threshold_li
from skimage.measure import label, regionprops_table
import skimage.draw as draw
from scipy.ndimage import (
    binary_opening,
    binary_closing,
    binary_fill_holes,
    median_filter,
    gaussian_filter,
)
from sklearn.cluster import HDBSCAN
from numba import jit
import polars as pl
import pathos
from pathos.pools import ProcessPool as Pool
import os
import sys

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
import IOFunctions
import HelperFunctions
import SpotDetectionFunctions

SD_F = SpotDetectionFunctions.SpotDetection_Functions()


HF = HelperFunctions.Helper_Functions()
IO = IOFunctions.IO_Functions()


class ImageAnalysis_Functions:
    def __init__(self, cpu_load=0.9):
        self.cpu_number = int(pathos.helpers.cpu_count() * cpu_load)
        self._pool = None
        return

    def _get_pool(self):
        if self._pool is None:
            self._pool = Pool(nodes=self.cpu_number)
        return self._pool

    def calculate_gradient_field(self, image, kernel, FS=True, sigma=None):
        """
        Calculate the gradient field of an image and compute focus-related measures.

        Args:
            image (numpy.ndarray): The input image.
            kernel (numpy.ndarray): The kernel for low-pass filtering.
            FS (bool): If True, compute focus score. Default True.
            sigma (float, optional): Gaussian sigma. When provided for 3D images,
                uses scipy.ndimage.gaussian_filter (5x faster, numerically identical
                to kernel-based FFT at truncate=2.0).

        Returns:
            filtered_image (numpy.ndarray): Image after low-pass filtering.
            gradient_x (numpy.ndarray): X-gradient of the filtered image.
            gradient_y (numpy.ndarray): Y-gradient of the filtered image.
            focus_score (numpy.ndarray): Focus score of the image.
            concentration_factor (numpy.ndarray): Concentration factor of the image.
        """
        # Use uninitialized arrays — edges not reached by np.diff are zeroed explicitly.
        gradient_x = np.empty_like(image)
        gradient_y = np.empty_like(image)

        # Low-pass filtering
        if len(image.shape) > 2:
            if sigma is not None:
                # gaussian_filter with matching truncate gives numerically identical
                # results (max diff < 1e-11) at ~5x the speed of the FFT loop.
                filtered_image = gaussian_filter(
                    np.asarray(image, dtype=float),
                    sigma=(0, sigma, sigma),
                    truncate=2.0,
                    mode="nearest",
                )
            else:
                filtered_image = np.empty_like(image)
                for channel in range(image.shape[0]):
                    image_padded = np.pad(
                        image[channel, :, :],
                        (
                            (kernel.shape[0] // 2, kernel.shape[0] // 2),
                            (kernel.shape[1] // 2, kernel.shape[1] // 2),
                        ),
                        mode="edge",
                    )
                    filtered_image[channel, :, :] = fftconvolve(
                        image_padded, kernel, mode="valid"
                    )
            gradient_x[:, :, :-1] = np.diff(filtered_image, axis=2)
            gradient_x[:, :, -1] = 0.0
            gradient_y[:, :-1, :] = np.diff(filtered_image, axis=1)
            gradient_y[:, -1, :] = 0.0
        else:
            filtered_image = np.empty_like(image)
            image_padded = np.pad(
                image,
                (
                    (kernel.shape[0] // 2, kernel.shape[0] // 2),
                    (kernel.shape[1] // 2, kernel.shape[1] // 2),
                ),
                mode="edge",
            )
            filtered_image[:, :] = fftconvolve(image_padded, kernel, mode="valid")
            gradient_x[:, :-1] = np.diff(filtered_image, axis=1)
            gradient_x[:, -1] = 0.0
            gradient_y[:-1, :] = np.diff(filtered_image, axis=0)
            gradient_y[-1, :] = 0.0

        if FS == True:
            gradient_magnitude = np.sqrt(
                np.add(np.square(gradient_x), np.square(gradient_y))
            )
            if len(image.shape) > 2:
                sum_gradient = np.sum(gradient_magnitude, axis=(1, 2))
            else:
                sum_gradient = np.sum(gradient_magnitude, axis=(0, 1))
            concentration_factor = np.divide(sum_gradient, np.max(sum_gradient))
            focus_score = np.log(sum_gradient)
        else:
            focus_score = None
            concentration_factor = None

        return filtered_image, gradient_x, gradient_y, focus_score, concentration_factor

    def calculate_radiality(self, pil_small, img, gradient_x, gradient_y, d=2):
        """
        Calculate radiality measures based on pixel neighborhoods and gradients.

        Args:
            pil_small (list): List of pixel indices.
            img (numpy.2darray): The input image.
            gradient_x (numpy.ndarray): X-gradient of the image.
            gradient_y (numpy.ndarray): Y-gradient of the image.
            d (integer): pixel ring size

        Returns:
            radiality (numpy.ndarray): Radiality measures.
        """
        xy = np.zeros([len(pil_small), 2])
        r0 = np.zeros(len(pil_small))
        for index in np.arange(len(pil_small)):
            pil_t = pil_small[index]
            r0[index], mi = np.max(img[pil_t[:, 0], pil_t[:, 1]]), np.argmax(
                img[pil_t[:, 0], pil_t[:, 1]]
            )
            xy[index, :] = pil_t[mi]

        xy_default = (
            np.asarray(
                np.unravel_index(
                    np.unique(
                        np.ravel_multi_index(
                            np.asarray(draw.circle_perimeter(5, 5, d)),
                            img.shape,
                            order="F",
                        )
                    ),
                    img.shape,
                    order="F",
                )
            ).T
            - 5
        )

        x = np.asarray(
            np.tile(xy_default[:, 0], (len(pil_small), 1)).T + xy[:, 0], dtype=int
        ).T

        y = np.asarray(
            np.tile(xy_default[:, 1], (len(pil_small), 1)).T + xy[:, 1], dtype=int
        ).T

        x_lim, y_lim = img.shape
        x[x >= x_lim] = x_lim - 1
        x[x < 0] = 0
        y[y >= y_lim] = y_lim - 1
        y[y < 0] = 0

        g2 = np.sqrt(np.add(np.square(gradient_x[x, y]), np.square(gradient_y[x, y])))

        flatness = np.mean(np.divide(img[x, y].T, r0), axis=0)
        integrated_grad = np.sum(g2, axis=1)
        radiality = np.vstack([flatness, integrated_grad]).T

        return radiality

    def default_SMD_routine(self, image, d=2, pfa=1e-6, localise_in_first_frame=False):
        """
        Daisy-chains analyses to get
        basic image properties (centroids, radiality)
        from a SMD image (i.e. no background)

        Args:
            image (array): image as numpy array
            d (int): integer for pyRASP intensity analyses
            pfa (float): probability of false alarm for spot detection code

        Returns:
            centroids (2D array): centroid positions per oligomer
            estimated_intensity (numpy.ndarray): Estimated sum intensity per oligomer.
            estimated_background (numpy.ndarray): Estimated mean background per oligomer.
            estimated_background_perpixel (numpy.ndarray): Estimated mean background per pixel.

        """
        to_return = None
        if len(image.shape) > 2:
            for i in np.arange(image.shape[0]):
                to_save = {}
                img = image[i, :, :]
                if localise_in_first_frame == False:
                    centroids = SD_F.detect_puncta_in_image(image=img, pfa=pfa)
                else:
                    if i == 0:
                        centroids = SD_F.detect_puncta_in_image(image=img, pfa=pfa)
                (
                    estimated_intensity,
                    estimated_background,
                    estimated_background_perpixel,
                ) = self.estimate_intensity(img, centroids)
                to_save["sum_intensity_in_photons"] = estimated_intensity
                to_save["bg_per_punctum"] = estimated_background
                to_save["bg_per_pixel"] = estimated_background_perpixel
                to_save["x"] = centroids[:, 0]
                to_save["y"] = centroids[:, 1]
                to_save["frame"] = np.full_like(centroids[:, 1], i) + 1

                if to_return is None:
                    to_return = pl.DataFrame(to_save)
                else:
                    temp = pl.DataFrame(to_save)
                    to_return = pl.concat([to_return, temp])
        return to_return

    def default_spotanalysis_routine(
        self,
        image,
        k1,
        k2,
        img2,
        Gx,
        Gy,
        thres=0.05,
        large_thres=100.0,
        areathres=30.0,
        rdl=[50.0, 0.0, 0.0],
        d=2,
    ):
        """
        Daisy-chains analyses to get
        basic image properties (centroids, radiality)
        from a single image

        Args:
            image (array): image as numpy array
            k1 (array): gaussian blur kernel
            k2 (array): ricker wavelet kernel
            thres (float): percentage threshold
            areathres (float): area threshold
            rdl (array): radiality thresholds

        Returns:
            centroids (2D array): centroid positions per oligomer
            estimated_intensity (numpy.ndarray): Estimated sum intensity per oligomer.
            estimated_background (numpy.ndarray): Estimated mean background per oligomer.
            estimated_background_perpixel (numpy.ndarray): Estimated mean background per pixel.
            areas_large (np.1darray): 1d array of areas of large objects
            centroids_large (np.2darray): centroids of large objects
            meanintensities_large (np.1darray): mean intensities of large objects
            sumintensities_large (np.1darray): sum intensities of large objects
            lo_mask (list of np.2darray): pixels where large objects found

        """
        large_mask = self.detect_large_features(image, large_thres)
        (
            pil_large,
            areas_large,
            centroids_large,
            sumintensities_large,
            meanintensities_large,
        ) = self.calculate_region_properties(large_mask, image)
        to_keep = np.where(areas_large > areathres)[0]
        pil_large = pil_large[to_keep]
        areas_large = areas_large[to_keep]
        centroids_large = centroids_large[to_keep, :]
        meanintensities_large = meanintensities_large[to_keep]
        sumintensities_large = sumintensities_large[to_keep]
        lo_mask = np.zeros_like(large_mask, dtype=bool)
        if len(pil_large) > 0:
            indices_to_keep = np.concatenate(pil_large)
            lo_mask[tuple(indices_to_keep.T)] = True
            peakintensities_large = np.array(
                [
                    np.max(image[pil_large[i][:, 0], pil_large[i][:, 1]])
                    for i in range(len(pil_large))
                ]
            )
            stdintensities_large = np.array(
                [
                    np.std(image[pil_large[i][:, 0], pil_large[i][:, 1]])
                    for i in range(len(pil_large))
                ]
            )
        else:
            peakintensities_large = np.array([])
            stdintensities_large = np.array([])

        dl_mask, centroids, radiality, idxs = self.small_feature_kernel(
            image, large_mask, img2, Gx, Gy, k2, thres, areathres, rdl, d
        )
        estimated_intensity, estimated_background, estimated_background_perpixel = (
            self.estimate_intensity(image, centroids)
        )
        to_keep = ~np.isnan(estimated_intensity)
        estimated_intensity = estimated_intensity[to_keep]
        estimated_background = estimated_background[to_keep]
        estimated_background_perpixel = estimated_background_perpixel[to_keep]
        centroids = centroids[to_keep, :]

        to_return = [
            centroids,
            estimated_intensity,
            estimated_background,
            estimated_background_perpixel,
            areas_large,
            centroids_large,
            meanintensities_large,
            sumintensities_large,
            lo_mask,
            stdintensities_large,  # index 9
            peakintensities_large,  # index 10
        ]

        return to_return

    def create_kernel(self, background_sigma, wavelet_sigma):
        """
        Create Gaussian and Ricker wavelet kernels.

        Args:
            background_sigma (float): Standard deviation for Gaussian kernel.
            wavelet_sigma (float): Standard deviation for Ricker wavelet.

        Returns:
            gaussian_kernel (numpy.ndarray): Gaussian kernel for background suppression.
            ricker_kernel (numpy.ndarray): Ricker wavelet for feature enhancement.
        """
        gaussian_kernel = self.create_gaussian_kernel(
            (background_sigma, background_sigma),
            (
                2 * int(np.ceil(2 * background_sigma)) + 1,
                2 * int(np.ceil(2 * background_sigma)) + 1,
            ),
        )
        ricker_kernel = self.ricker_wavelet(wavelet_sigma)
        return gaussian_kernel, ricker_kernel

    def create_gaussian_kernel(self, sigmas, size):
        """
        Create a 2D Gaussian kernel.

        Args:
            sigmas (tuple): Standard deviations in X and Y directions.
            size (tuple): Size of the kernel.

        Returns:
            kernel (numpy.ndarray): 2D Gaussian kernel.
        """
        kernel_x = gauss(size[0], sigmas[0])[:, np.newaxis]
        kernel_y = gauss(size[1], sigmas[1])
        kernel = np.multiply(kernel_x, kernel_y)
        kernel = np.divide(kernel, np.nansum(kernel))
        return kernel

    def ricker_wavelet(self, sigma):
        """
        Create a 2D Ricker wavelet.

        Args:
            sigma (float): Standard deviation for the wavelet.

        Returns:
            wavelet (numpy.ndarray): 2D Ricker wavelet.
        """
        amplitude = np.divide(
            2.0, np.multiply(np.sqrt(np.multiply(3.0, sigma)), np.power(np.pi, 0.25))
        )
        length = int(np.ceil(np.multiply(4, sigma)))

        x = np.linspace(-length, length, 2 * length + 1)
        X, Y = np.meshgrid(x, x)

        sigma_sq = np.square(sigma)
        common_term = np.add(
            np.divide(np.square(X), np.multiply(2.0, sigma_sq)),
            np.divide(np.square(Y), np.multiply(2.0, sigma_sq)),
        )
        wavelet = np.multiply(
            amplitude, np.multiply(np.subtract(1, common_term), np.exp(-common_term))
        )
        return wavelet

    def create_filled_region(self, image_size, indices_to_keep):
        """
        Fill a region in a boolean matrix based on specified indices.

        Args:
            image_size (tuple): Size of the boolean matrix.
            indices_to_keep (list): List of indices to set as True.

        Returns:
            boolean_matrix (numpy.ndarray): Boolean matrix with specified indices set to True.
        """
        # Concatenate all indices to keep into a single array
        indices_to_keep = np.concatenate(indices_to_keep)

        # Create a boolean matrix of size image_size
        boolean_matrix = np.zeros(image_size, dtype=bool)

        # Set the elements at indices specified in indices_to_keep to True
        # Utilize tuple unpacking for efficient indexing and assignment
        boolean_matrix[tuple(indices_to_keep.T)] = True

        return boolean_matrix

    def infocus_indices(self, focus_scores, cluster_size=4):
        """
        Identify in-focus indices based on HDBSCAN.

        Args:
            focus_scores (numpy.ndarray): Focus scores for different slices.
            cluster_size (int): how big the smallest cluster is

        Returns:
            in_focus_indices (list): List containing the first and last in-focus indices.
        """
        hdb = HDBSCAN(min_cluster_size=cluster_size)

        hdb.fit(focus_scores.reshape(-1, 1))
        if len(np.unique(hdb.labels_)) == 1:
            return np.array([0, 0])  # no in-focus slices
        else:
            mean_scores = np.zeros(len(np.unique(hdb.labels_)))
            for i, l in enumerate(np.unique(hdb.labels_)):
                mean_scores[i] = np.mean(focus_scores[hdb.labels_ == l])
            focus_label = np.unique(hdb.labels_)[np.argmax(mean_scores)]
            infocus_indices = np.where(hdb.labels_ == focus_label)[0]
            in_focus_indices = np.array(
                [np.min(infocus_indices), np.max(infocus_indices) + 1]
            )
            return in_focus_indices

    @staticmethod
    @jit(nopython=True)
    def infocus_indices_old(focus_scores, threshold_differential):
        """
        Identify in-focus indices based on focus scores and a threshold differential.

        Args:
            focus_scores (numpy.ndarray): Focus scores for different slices.
            threshold_differential (float): Threshold for differential focus scores.

        Returns:
            in_focus_indices (list): List containing the first and last in-focus indices.
        """
        # Calculate the Euclidean distance between each slice in focus_scores
        focus_score_diff = np.diff(focus_scores)

        # Mask distances less than or equal to 0 as NaN
        focus_score_diff[focus_score_diff <= 0] = np.nan

        # Perform DBSCAN from the start
        dist1 = np.hstack(
            (np.array([0.0]), focus_score_diff > threshold_differential)
        )  # Mark as True if distance exceeds threshold

        # Calculate the Euclidean distance from the end
        focus_score_diff_end = np.diff(focus_scores[::-1].copy())

        # Perform DBSCAN from the end
        dist2 = np.hstack(
            (np.array([0.0]), focus_score_diff_end < threshold_differential)
        )  # Mark as True if distance is below threshold

        # Refine the DBSCAN results
        dist1 = np.diff(dist1)
        dist2 = np.diff(dist2)

        # Find indices indicating the transition from out-of-focus to in-focus and vice versa
        transition_to_in_focus = np.where(dist1 == -1)[0]
        transition_to_out_focus = np.where(dist2 == 1)[0]

        # Determine the first and last slices in focus
        first_in_focus = (
            0 if len(transition_to_in_focus) == 0 else transition_to_in_focus[0]
        )  # First slice in focus
        last_in_focus = (
            len(focus_scores)
            if len(transition_to_out_focus) == 0
            else (len(focus_scores) - transition_to_out_focus[-1]) + 1
        )  # Last slice in focus
        if last_in_focus > len(focus_scores):
            last_in_focus = len(focus_scores)

        # Ensure consistency and handle cases where the first in-focus slice comes after the last
        first_in_focus = first_in_focus if first_in_focus <= last_in_focus else 1

        # Return indices for in-focus images
        in_focus_indices = np.array([first_in_focus, last_in_focus])
        return in_focus_indices

    def estimate_intensity(self, image, centroids):
        """
        Estimate intensity values for each centroid in the image.

        Args:
            image (numpy.2darray): Input image.
            centroids (numpy.ndarray): Centroid locations.

        Returns:
            estimated_intensity (numpy.ndarray): Estimated sum intensity per oligomer.
            estimated_background (numpy.ndarray): Estimated mean background per oligomer.
            estimated_background_perpixel (numpy.ndarray): Estimated mean background per pixel.
        """
        centroids = np.asarray(centroids, dtype=int)
        image_size = image.shape
        indices = np.ravel_multi_index(centroids.T, image_size, order="F")
        estimated_intensity = np.zeros(
            len(indices), dtype=float
        )  # Estimated sum intensity per oligomer

        x_in, y_in, x_out, y_out = self.intensity_pixel_indices(centroids, image_size)

        estimated_background = np.mean(image[x_out, y_out], axis=0)
        estimated_intensity = np.sum(
            np.subtract(image[x_in, y_in], estimated_background), axis=0
        )

        estimated_intensity[estimated_intensity < 0] = np.nan
        estimated_background[estimated_background < 0] = np.nan

        # correct for averaged background; report background summed
        return (
            estimated_intensity,
            estimated_background * len(x_in),
            estimated_background,
        )

    def intensity_pixel_indices(self, centroid_loc, image_size):
        """
        Calculate pixel indices for inner and outer regions around the given index.

        Args:
            centroid_loc (2D array): xy location of the pixel.
            image_size (tuple): Size of the image.

        Returns:
            inner_indices (numpy.ndarray): Pixel indices for the inner region.
            outer_indices (numpy.ndarray): Pixel indices for the outer region.
        """

        def calculate_offsets(octagon_shape):
            x, y = np.where(octagon_shape)
            x -= int(octagon_shape.shape[0] / 2)
            y -= int(octagon_shape.shape[1] / 2)
            return x, y

        small_oct = ski.morphology.octagon(2, 4)
        outer_ind = ski.morphology.octagon(2, 5)
        inner_ind = np.pad(
            small_oct, int((outer_ind.shape[0] - small_oct.shape[0]) / 2)
        )
        outer_ind -= inner_ind

        x_inner, y_inner = calculate_offsets(inner_ind)
        x_outer, y_outer = calculate_offsets(outer_ind)

        x_inner = np.tile(x_inner, (len(centroid_loc), 1)).T + centroid_loc[:, 0]
        y_inner = np.tile(y_inner, (len(centroid_loc), 1)).T + centroid_loc[:, 1]
        x_outer = np.tile(x_outer, (len(centroid_loc), 1)).T + centroid_loc[:, 0]
        y_outer = np.tile(y_outer, (len(centroid_loc), 1)).T + centroid_loc[:, 1]

        x_inner[x_inner < 0] = 0
        y_inner[y_inner < 0] = 0
        x_inner[x_inner >= image_size[0]] = image_size[0] - 1
        y_inner[y_inner >= image_size[1]] = image_size[1] - 1

        x_outer[x_outer < 0] = 0
        y_outer[y_outer < 0] = 0
        x_outer[x_outer >= image_size[0]] = image_size[0] - 1
        y_outer[y_outer >= image_size[1]] = image_size[1] - 1

        return x_inner, y_inner, x_outer, y_outer

    def detect_large_features(
        self, image, threshold1, threshold2=0, sigma1=2.0, sigma2=60.0
    ):
        """
        Detects large features in an image based on a given threshold.

        Args:
            image (numpy.ndarray): Original image.
            threshold1 (float): Threshold for determining features. Only this is
                used for the determination of large protein aggregates.
            threshold2 (float): Threshold for determining cell features. If above
                0, gets used and cellular features are detected.
            sigma1 (float): first gaussian blur width
            sigma2 (float): second gaussian blur width

        Returns:
            large_mask (numpy.ndarray): Binary mask for the large features.
        """
        # Apply Gaussian filters with different sigmas and subtract to enhance features
        enhanced_image = gaussian(image, sigma=sigma1, truncate=2.0) - gaussian(
            image, sigma=sigma2, truncate=2.0
        )

        # Create a binary mask for large features based on the threshold
        large_mask = enhanced_image > threshold1

        if threshold2 > 0:
            large_mask = binary_opening(large_mask, structure=ski.morphology.disk(1))
            large_mask = binary_closing(large_mask, structure=ski.morphology.disk(5))
            pixel_index_list, *rest = self.calculate_region_properties(large_mask)
            idx1 = np.zeros_like(pixel_index_list, dtype=bool)
            imcopy = np.copy(image)

            for i in np.arange(len(pixel_index_list)):
                idx1[i] = 1 * (
                    np.sum(
                        imcopy[pixel_index_list[i][:, 0], pixel_index_list[i][:, 1]]
                        > threshold2
                    )
                    / len(pixel_index_list[i][:, 0])
                    > 0.1
                )

            if len(idx1) > 0:
                large_mask = self.create_filled_region(
                    image.shape, pixel_index_list[idx1]
                )
                large_mask = binary_fill_holes(large_mask)

        return large_mask.astype(bool)

    def _overlap_with_mask(self, centroids, mask):
        """
        Tests, for each centroid, whether the pixel it sits on is True in mask.

        Args:
            centroids (np.ndarray): centroid positions, column 0/1 indexing
                mask.shape[0]/[1] directly (same convention used throughout
                this module, e.g. intensity_pixel_indices).
            mask (np.ndarray): binary mask, same shape as the image the
                centroids were detected on.

        Returns:
            overlap (np.ndarray): boolean array, one value per centroid.
        """
        if mask is None or len(centroids) == 0:
            return np.zeros(len(centroids), dtype=bool)
        rows = np.clip(centroids[:, 0].astype(int), 0, mask.shape[0] - 1)
        cols = np.clip(centroids[:, 1].astype(int), 0, mask.shape[1] - 1)
        return mask[rows, cols]

    def _detect_cell_mask_3d(
        self,
        image_cell,
        cell_sigma1,
        cell_sigma2,
        cell_lower_size_threshold,
        cell_upper_size_threshold,
        cell_hole_threshold,
        cell_erosionsize,
        cell_spacing,
    ):
        """
        Runs the full 3D cell-mask pipeline on a cell-channel image: per-plane
        Otsu-thresholded DoG detection (detect_large_features_3D), then 3D
        cleanup (threshold_cell_areas_3d) that connects objects across z,
        removes objects outside the size range or that don't span enough
        planes, and closes the result.

        Args:
            image_cell (np.ndarray or None): cell-channel image, either a 3D
                (z, y, x) stack or a single 2D (y, x) plane.
            cell_sigma1/cell_sigma2 (float): DoG-enhancement blur widths.
            cell_lower_size_threshold/cell_upper_size_threshold (float): cell
                size range in voxels.
            cell_hole_threshold (float): hole-fill size threshold (voxels).
            cell_erosionsize (int): closing footprint size.
            cell_spacing (tuple): voxel spacing (z, y, x) in um.

        Returns:
            cell_mask (np.ndarray or None): cleaned cell mask, same
                dimensionality as image_cell, or None if image_cell is None or
                detection fails.
        """
        if image_cell is None:
            return None
        is_2d = image_cell.ndim == 2
        image_cell_3d = image_cell[np.newaxis, :, :] if is_2d else image_cell
        try:
            raw_cell_mask = self.detect_large_features_3D(
                image_cell_3d,
                filter_function=ski.filters.threshold_otsu,
                sigma1=cell_sigma1,
                sigma2=cell_sigma2,
                hole_threshold=cell_hole_threshold,
                cell_threshold=cell_lower_size_threshold,
            )
            cell_mask, _, _, _ = self.threshold_cell_areas_3d(
                raw_cell_mask,
                lower_cell_size_threshold=cell_lower_size_threshold,
                upper_cell_size_threshold=cell_upper_size_threshold,
                spacing=cell_spacing,
                n_planes=min(3, raw_cell_mask.shape[0]),
                erosionsize=cell_erosionsize,
            )
        except Exception:
            return None
        return cell_mask[0] if is_2d else cell_mask

    def detect_large_features_3D(
        self,
        image,
        filter_function,
        sigma1=2.0,
        sigma2=60.0,
        hole_threshold=100,
        cell_threshold=2000,
    ):
        """
        Detects large features in an 3D image based on given filter function.

        Args:
            image (numpy.ndarray): Original image.
            filter_function (function): Threshold determining function, e.g. ski.filters.threshold_yen.
            sigma1 (float): first gaussian blur width
            sigma2 (float): second gaussian blur width
            hole_threshold (float): hole size threshold
            cell_threshold (float): cell size threshold

        Returns:
            large_mask (numpy.ndarray): Binary mask for the large features.
        """

        large_mask = np.zeros_like(image)
        for i in np.arange(image.shape[0]):
            enhanced_image = gaussian(
                image[i, :, :], sigma=sigma1, truncate=2.0
            ) - gaussian(image[i, :, :], sigma=sigma2, truncate=2.0)

            # Create a binary mask for large features based on the threshold
            large_mask[i, :, :] = enhanced_image > filter_function(enhanced_image)

            large_mask[i, :, :] = binary_opening(
                large_mask[i, :, :], structure=ski.morphology.disk(1)
            )
            large_mask[i, :, :] = binary_closing(
                large_mask[i, :, :], structure=ski.morphology.disk(5)
            )
        large_mask = binary_fill_holes(large_mask)
        large_mask = ski.morphology.remove_small_holes(
            large_mask, area_threshold=hole_threshold
        )
        large_mask = ski.morphology.remove_small_objects(
            large_mask, min_size=cell_threshold
        )
        return large_mask.astype(bool)

    def threshold_cell_areas_3d(
        self,
        cell_mask,
        lower_cell_size_threshold=10000,
        upper_cell_size_threshold=200000,
        spacing=(0.5, 0.11, 0.11),
        n_planes=3,
        erosionsize=5,
        plane_max=0.15,
    ):
        """
        Removes small or objects from a cell mask.

        Args:
            cell_mask_raw (np.ndarray): Cell mask object
            lower_cell_size_threshold (float): Lower size threshold
            upper_cell_size_threshold (float): Upper size threshold
            spacing (tuple): pixel spacing
            n_planes (int): number of planes object has to be across
            erosionsize (int): how big a dilation ball to use
            plane_max (float): if a plane is occupied by more than this fraction,
                                delete it

        Returns:
            tuple: Processed cell mask, pixel image locations, areas, centroids
        """
        try:
            if len(cell_mask.shape) < 3:
                raise Exception("Data is not 3D, as required.")
        except Exception as error:
            print("Caught this error: " + repr(error))
            return
        todel = np.zeros(cell_mask.shape[0])
        # first, delete any planes that are filled by more than plane_max
        for i in np.arange(cell_mask.shape[0]):
            if np.mean(cell_mask[i, :, :]) > plane_max:
                cell_mask[i, :, :] = 0
                todel[i] = 1

        filled = binary_fill_holes(cell_mask)

        cell_mask_new = ski.morphology.remove_small_holes(
            filled,
            area_threshold=lower_cell_size_threshold,
        )
        cell_mask_new = ski.morphology.remove_small_objects(
            cell_mask_new, min_size=lower_cell_size_threshold
        )
        objects = ski.measure.label(cell_mask_new)
        large_objects = ski.morphology.remove_small_objects(
            objects, min_size=upper_cell_size_threshold
        )
        small_objects = objects ^ large_objects
        cell_mask_new = np.asarray(small_objects.clip(0, 1), dtype=bool)
        if erosionsize > 0:
            cell_mask_new = np.asarray(
                ski.morphology.binary_closing(
                    cell_mask_new,
                    footprint=ski.morphology.footprint_rectangle(
                        (1, erosionsize, erosionsize)
                    ),
                ),
                dtype=bool,
            )
        pil_raw, _, _, _, _ = self.calculate_region_properties(
            cell_mask_new, dims=3, spacing=(1, 1, 1)
        )

        for i in np.arange(len(pil_raw)):
            if len(np.unique(pil_raw[i][:, 0])) < n_planes:
                cell_mask_new[pil_raw[i][:, 0], pil_raw[i][:, 1], pil_raw[i][:, 2]] = 0

        for i in np.arange(cell_mask_new.shape[0]):
            if todel[i] == 1:
                cell_mask_new[i, :, :] = 0
        pil, areas, centroids, _, _ = self.calculate_region_properties(
            cell_mask_new, dims=3, spacing=spacing
        )

        return cell_mask_new, pil, areas, centroids

    def detect_large_features_3D_aggregates(
        self, image, threshold, sigma1=2.0, sigma2=60.0
    ):
        """
        Detects large protein aggregates in a 3D image using a fixed intensity
        threshold, then joins detections across z-planes via 3D connected
        components.

        Unlike detect_large_features_3D (which uses an automatic threshold per
        plane), this applies the same fixed threshold used in the per-plane
        large-feature detector so that identical objects are found but each
        aggregate is reported exactly once in 3D.

        Args:
            image (numpy.ndarray): 3-D image stack (z, x, y).
            threshold (float): Intensity threshold for the DoG-enhanced image.
            sigma1 (float): Inner Gaussian blur width. Default 2.0.
            sigma2 (float): Outer Gaussian blur width (background). Default 60.0.

        Returns:
            large_mask (numpy.ndarray): 3-D boolean mask for detected aggregates.
        """
        large_mask = np.zeros_like(image, dtype=bool)
        for i in range(image.shape[0]):
            enhanced = gaussian(image[i, :, :], sigma=sigma1, truncate=2.0) - gaussian(
                image[i, :, :], sigma=sigma2, truncate=2.0
            )
            plane_mask = enhanced > threshold
            plane_mask = binary_opening(plane_mask, structure=ski.morphology.disk(1))
            large_mask[i, :, :] = plane_mask
        # Join across z-planes with 3D connectivity
        large_mask = binary_fill_holes(large_mask)
        return large_mask.astype(bool)

    def calculate_region_properties(
        self, binary_mask, image=None, dims=2, spacing=(0.5, 0.11, 0.11)
    ):
        """
        Calculate properties for labeled regions in a binary mask.

        Args:
            binary_mask (numpy.ndarray): Binary mask of connected components.
            image (numpy.ndarray): image of same dimension as binary mask. Optional.
            dims (float): if 3, gets 3 centroid values. if 2, 2.
            spacing (tuple): pixel spacing. Converts area into real units.

        Returns:
            pixel_index_list (list): List containing pixel indices for each labeled object.
            areas (numpy.ndarray): Array containing areas (in um2 or um3) of each labeled object.
            centroids (numpy.ndarray): Array containing centroids (x, y) of each labeled object.
        """
        # Find connected components and count the number of objects
        if len(spacing) > dims:
            spacing = spacing[1:]
        labeled_image, num_objects = label(
            binary_mask, connectivity=dims, return_num=True
        )
        # Initialize arrays for storing properties
        centroids = np.zeros((num_objects, dims))

        if image is not None:
            properties = ("centroid", "area", "coords", "intensity_mean")
        else:
            properties = ("centroid", "area", "coords")

        # Get region properties
        props = regionprops_table(
            labeled_image, intensity_image=image, properties=properties, spacing=spacing
        )
        for i in np.arange(dims):
            centroids[:, i] = np.asarray(
                props["centroid-" + str(int(i))] / spacing[i], dtype=int
            )
        centroids = np.array(centroids, dtype=int)
        areas = props["area"]
        pixel_index_list = props["coords"]
        if image is not None:
            sum_intensity = props["intensity_mean"] * areas
            mean_intensity = props["intensity_mean"]
        else:
            sum_intensity = None
            mean_intensity = None
        return pixel_index_list, areas, centroids, sum_intensity, mean_intensity

    def small_feature_kernel(
        self, img, large_mask, img2, Gx, Gy, k2, thres, area_thres, rdl, d=2
    ):
        """
        Find small features in an image and determine diffraction-limited (dl) and non-diffraction-limited (ndl) features.

        Args:
            img (numpy.ndarray): Original image.
            large_mask (numpy.ndarray): Binary mask for large features.
            img2 (numpy.ndarray): Smoothed image for background suppression.
            Gx (numpy.ndarray): Gradient image in x-direction.
            Gy (numpy.ndarray): Gradient image in y-direction.
            k2 (numpy.ndarray): The kernel for blob feature enhancement.
            thres (float): Converting real-valued image into a binary mask.
            area_thres (float): The maximum area in pixels a diffraction-limited object can be.
            rdl (list): Radiality threshold [min_radiality, max_radiality, area].
            d (integer): pixel radius

        Returns:
            dl_mask (numpy.ndarray): Binary mask for diffraction-limited (dl) features.
            centroids (numpy.ndarray): Centroids for dl features.
            radiality (numpy.ndarray): Radiality value for all features (before the filtering based on the radiality).
            idxs (numpy.ndarray): Indices for objects that satisfy the decision boundary.
        """
        img1 = np.maximum(np.subtract(img, img2), 0)
        pad_size = np.subtract(np.asarray(k2.shape), 1) // 2
        img1 = fftconvolve(np.pad(img1, pad_size, mode="edge"), k2, mode="valid")

        if thres < 1:
            thres = np.percentile(img1.ravel(), 100 * (1 - thres))
        BW = img1 > thres
        BW = np.logical_or(BW, large_mask)
        BW = binary_opening(BW, structure=ski.morphology.disk(1))

        imsz = img.shape
        pixel_idx_list, areas, centroids, _, _ = self.calculate_region_properties(BW)

        border_value = int(np.multiply(d, 5))
        idxb = np.logical_and(
            centroids[:, 0] > border_value, centroids[:, 0] < imsz[1] - border_value
        )
        idxb = np.logical_and(
            idxb,
            np.logical_and(
                centroids[:, 1] > border_value,
                centroids[:, 1] < imsz[0] - (border_value - 1),
            ),
        )
        idxs = np.logical_and(areas < area_thres, idxb)

        pil_small = pixel_idx_list[idxs]
        centroids = centroids[idxs]
        radiality = self.calculate_radiality(pil_small, img2, Gx, Gy, d)

        idxs = np.logical_and(radiality[:, 0] <= rdl[0], radiality[:, 1] >= rdl[1])
        centroids = np.floor(centroids[idxs])
        centroids = np.asarray(centroids)
        if len(pil_small[idxs]) > 1:
            dl_mask = self.create_filled_region(imsz, pil_small[idxs])
        else:
            dl_mask = np.full_like(img, False)
        return dl_mask, centroids, radiality, idxs

    def compute_image_props(
        self,
        image,
        k1,
        k2,
        img2,
        Gx,
        Gy,
        thres=0.05,
        large_thres=450.0,
        areathres=30.0,
        rdl=[50.0, 0.0, 0.0],
        d=2,
        z_planes=0,
        calib=False,
    ):
        """
        Gets basic image properties (dl_mask, centroids, radiality)
        from a single image

        Args:
            image (array): image as numpy array
            k1 (array): gaussian blur kernel
            k2 (array): ricker wavelet kernel
            thres (float): percentage threshold
            areathres (float): area threshold
            rdl (array): radiality thresholds
            d (int): radiality ring
            z_planes (array): If multiple z planes, give z planes
            calib (bool): If True, for radiality calibration

        Returns:
            dl_mask (numpy.ndarray): Binary mask for diffraction-limited (dl) features.
            centroids (numpy.ndarray): Centroids for dl features.
            radiality (numpy.ndarray): Radiality value for all features (before the filtering based on the radiality).
            large_mask (np.ndarray): Binary mask for large (above diffraction-limited) features.

        """
        if isinstance(z_planes, int):
            if calib == True:
                large_mask = np.full_like(image, False)
            else:
                large_mask = self.detect_large_features(image, large_thres)
            dl_mask, centroids, radiality, idxs = self.small_feature_kernel(
                image, large_mask, img2, Gx, Gy, k2, thres, areathres, rdl, d
            )

        else:
            radiality = {}
            centroids = {}
            large_mask = np.zeros_like(image)
            dl_mask = np.zeros_like(image)

            def run_over_z(image, img2, Gx, Gy):
                if calib == True:
                    large_mask = np.full_like(image, False)
                else:
                    large_mask = self.detect_large_features(image, large_thres)

                dl_mask, centroids, radiality, idxs = self.small_feature_kernel(
                    image,
                    large_mask,
                    img2,
                    Gx,
                    Gy,
                    k2,
                    thres,
                    areathres,
                    rdl,
                    d,
                )
                return dl_mask, centroids, radiality, idxs

            image_planes = [image[i, :, :] for i in range(image.shape[0])]
            planes_img2 = [img2[i, :, :] for i in range(img2.shape[0])]
            planes_Gx = [Gx[i, :, :] for i in range(Gx.shape[0])]
            planes_Gy = [Gy[i, :, :] for i in range(Gy.shape[0])]

            results = self._get_pool().map(
                run_over_z,
                image_planes,
                planes_img2,
                planes_Gx,
                planes_Gy,
            )

            dl_mask = [i[0] for i in results]
            centroids = [i[1] for i in results]
            radiality = [i[2] for i in results]
            large_mask = [i[3] for i in results]
        return dl_mask, centroids, radiality, large_mask

    def compute_spot_and_cell_props(
        self,
        image,
        image_cell,
        k1,
        k2,
        img2,
        Gx,
        Gy,
        prot_thres=0.05,
        large_prot_thres=100.0,
        areathres=30.0,
        rdl=[50.0, 0.0, 0.0],
        z=0,
        cell_sigma1=2.0,
        cell_sigma2=40.0,
        cell_lower_size_threshold=2000.0,
        cell_upper_size_threshold=np.inf,
        cell_hole_threshold=100.0,
        cell_erosionsize=3,
        cell_spacing=(0.5, 0.11, 0.11),
        d=2,
        image_bulk=None,
        bulk_threshold=100.0,
        bulk_sigma1=2.0,
        bulk_sigma2=60.0,
    ):
        """
        Gets basic image properties (centroids, radiality)
        from a single image and generates cell mask from another image channel.

        Cell masks are generated with a full 3D, per-plane-Otsu-thresholded
        detector (detect_large_features_3D) rather than the fixed-threshold 2D
        detector used for the bulk-stain mask, followed by 3D cleanup
        (threshold_cell_areas_3d) that connects objects across z, removes
        small/large objects and objects that don't span enough planes, and
        closes the result -- so the returned/saved cell_mask is already a
        clean 3D segmentation, not a stack of independently-thresholded planes.

        Args:
            image (array): image of protein stain as numpy array
            image_cell (array): image of cell stain as numpy array
            k1 (array): gaussian blur kernel
            k2 (array): ricker wavelet kernel
            prot_thres (float): percentage threshold for protein
            large_prot_thres (float): Protein threshold intensity
            areathres (float): area threshold
            rdl (array): radiality thresholds
            z (array): z planes to image, default 0
            cell_sigma1 (float): cell blur value 1 (DoG-enhancement inner sigma)
            cell_sigma2 (float): cell blur value 2 (DoG-enhancement outer/background sigma)
            cell_lower_size_threshold (float): minimum cell size in voxels; also
                used as the remove_small_objects threshold inside
                detect_large_features_3D
            cell_upper_size_threshold (float): maximum cell size in voxels
            cell_hole_threshold (float): hole-fill size threshold (voxels) inside
                detect_large_features_3D
            cell_erosionsize (int): closing footprint size used by the final
                3D cleanup pass
            cell_spacing (tuple): voxel spacing (z, y, x) in um, used to convert
                cell sizes into real units
            d (integer): pixel radius value
            image_bulk (array): image of bulk stain as numpy array, optional
            bulk_threshold (float): bulk stain DoG threshold
            bulk_sigma1 (float): bulk stain blur value 1
            bulk_sigma2 (float): bulk stain blur value 2

        Returns:
            to_save (pl.DataFrame): spot properties ready to save; has an
                "overlap_with_bulk_stain" boolean column if image_bulk given
            to_save_largeobjects (pl.DataFrame): large object properties ready to save
            lo_mask (np.ndarray): lo masks
            cell_mask (np.ndarray): cell mask
            bulk_mask (np.ndarray): bulk stain mask
        """

        columns = [
            "x",
            "y",
            "z",
            "sum_intensity_in_photons",
            "bg_per_punctum",
            "bg_per_pixel",
            "zi",
            "zf",
        ]
        columns_large = [
            "x",
            "y",
            "z",
            "area",
            "sum_intensity_in_photons",
            "mean_intensity_in_photons",
            "std_intensity_in_photons",
            "peak_intensity_in_photons",
            "zi",
            "zf",
        ]

        if not isinstance(z, int):
            z_planes = np.arange(z[0], z[1])

            def analyse_zplanes(
                zp,
                image_plane,
                img2_plane,
                Gx_plane,
                Gy_plane,
                imagebulk_plane,
            ):
                (
                    centroids,
                    estimated_intensity,
                    estimated_background,
                    estimated_background_perpixel,
                    areas_large,
                    centroids_large,
                    meanintensities_large,
                    sumintensities_large,
                    lo_mask,
                    _std_large,  # not used — 3D path computes these in 3D
                    _peak_large,  # not used — 3D path computes these in 3D
                ) = self.default_spotanalysis_routine(
                    image_plane,
                    k1,
                    k2,
                    img2_plane,
                    Gx_plane,
                    Gy_plane,
                    prot_thres,
                    large_prot_thres,
                    areathres,
                    rdl,
                    d,
                )

                if imagebulk_plane is not None:
                    bulk_mask = self.detect_large_features(
                        imagebulk_plane,
                        bulk_threshold,
                        0,
                        bulk_sigma1,
                        bulk_sigma2,
                    )
                    overlap_bulk = self._overlap_with_mask(centroids, bulk_mask)
                else:
                    bulk_mask = None
                    overlap_bulk = np.zeros(len(centroids), dtype=bool)
                return (
                    centroids,
                    estimated_intensity,
                    estimated_background,
                    estimated_background_perpixel,
                    areas_large,
                    centroids_large,
                    meanintensities_large,
                    sumintensities_large,
                    lo_mask,
                    bulk_mask,
                    overlap_bulk,
                )

            planes = [image[i, :, :] for i in range(image.shape[0])]
            planes_img2 = [img2[i, :, :] for i in range(img2.shape[0])]
            planes_Gx = [Gx[i, :, :] for i in range(Gx.shape[0])]
            planes_Gy = [Gy[i, :, :] for i in range(Gy.shape[0])]
            if image_bulk is not None:
                planes_imagebulk = [
                    image_bulk[i, :, :] for i in range(image_bulk.shape[0])
                ]
            else:
                planes_imagebulk = [None] * image.shape[0]

            results = self._get_pool().map(
                analyse_zplanes,
                np.arange(image.shape[0]),
                planes,
                planes_img2,
                planes_Gx,
                planes_Gy,
                planes_imagebulk,
            )

            centroids = [i[0] for i in results]
            estimated_intensity = [i[1] for i in results]
            estimated_background = [i[2] for i in results]
            estimated_background_perpixel = [i[3] for i in results]

            cell_mask = self._detect_cell_mask_3d(
                image_cell,
                cell_sigma1,
                cell_sigma2,
                cell_lower_size_threshold,
                cell_upper_size_threshold,
                cell_hole_threshold,
                cell_erosionsize,
                cell_spacing,
            )

            if image_bulk is not None and len(results) > 0:
                try:
                    bulk_mask = np.stack([i[9] for i in results], axis=0)
                except:
                    bulk_mask = None
                overlap_bulk = np.concatenate([i[10] for i in results])
            else:
                bulk_mask = None
                overlap_bulk = None

            to_save = HF.make_datarray_spot(
                centroids,
                estimated_intensity,
                estimated_background,
                estimated_background_perpixel,
                columns,
                z_planes,
            )
            if overlap_bulk is not None and to_save is not None:
                to_save = to_save.with_columns(
                    pl.Series("overlap_with_bulk_stain", overlap_bulk.astype(bool))
                )

            # Reuse per-plane lo_masks already computed in the pool instead of
            # re-running DoG — apply binary_opening per plane then fill in 3D.
            per_plane_lo = np.stack([r[8] for r in results], axis=0)
            for _pi in range(per_plane_lo.shape[0]):
                per_plane_lo[_pi] = binary_opening(
                    per_plane_lo[_pi], structure=ski.morphology.disk(1)
                )
            lo_mask_3d = binary_fill_holes(per_plane_lo)
            if np.any(lo_mask_3d):
                pil_lo, areas_lo, centroids_lo, sumint_lo, meanint_lo = (
                    self.calculate_region_properties(
                        lo_mask_3d, image, dims=3, spacing=(1, 1, 1)
                    )
                )
                stdint_lo = np.array(
                    [
                        np.std(image[pil_lo[i][:, 0], pil_lo[i][:, 1], pil_lo[i][:, 2]])
                        for i in range(len(pil_lo))
                    ],
                    dtype=float,
                )
                peakint_lo = np.array(
                    [
                        np.max(image[pil_lo[i][:, 0], pil_lo[i][:, 1], pil_lo[i][:, 2]])
                        for i in range(len(pil_lo))
                    ],
                    dtype=float,
                )
                zi_lo = np.array(
                    [
                        pil_lo[i][:, 0].min() + z_planes[0] + 1
                        for i in range(len(pil_lo))
                    ],
                    dtype=float,
                )
                zf_lo = np.array(
                    [
                        pil_lo[i][:, 0].max() + z_planes[0] + 1
                        for i in range(len(pil_lo))
                    ],
                    dtype=float,
                )
                voxel_volume_um3 = (
                    0.5 * 0.11 * 0.11
                )  # z_step=0.5 µm, xy=0.11 µm → µm³/voxel
                to_save_largeobjects = pl.DataFrame(
                    {
                        "x": centroids_lo[:, 1].astype(float),
                        "y": centroids_lo[:, 2].astype(float),
                        "z": (centroids_lo[:, 0] + z_planes[0] + 1).astype(float),
                        "area": areas_lo.astype(float) * voxel_volume_um3,
                        "sum_intensity_in_photons": sumint_lo,
                        "mean_intensity_in_photons": meanint_lo.astype(float),
                        "std_intensity_in_photons": stdint_lo,
                        "peak_intensity_in_photons": peakint_lo,
                        "zi": zi_lo,
                        "zf": zf_lo,
                    }
                )
            else:
                to_save_largeobjects = None
            lo_mask = lo_mask_3d
        else:
            (
                centroids,
                estimated_intensity,
                estimated_background,
                estimated_background_perpixel,
                areas_large,
                centroids_large,
                meanintensities_large,
                sumintensities_large,
                lo_mask,
                stdintensities_large,
                peakintensities_large,
            ) = self.default_spotanalysis_routine(
                image,
                k1,
                k2,
                img2,
                Gx,
                Gy,
                prot_thres,
                large_prot_thres,
                areathres,
                rdl,
                d,
            )
            cell_mask = self._detect_cell_mask_3d(
                image_cell,
                cell_sigma1,
                cell_sigma2,
                cell_lower_size_threshold,
                cell_upper_size_threshold,
                cell_hole_threshold,
                cell_erosionsize,
                cell_spacing,
            )
            bulk_mask = (
                self.detect_large_features(
                    image_bulk,
                    bulk_threshold,
                    0,
                    bulk_sigma1,
                    bulk_sigma2,
                )
                if image_bulk is not None
                else None
            )

            to_save = HF.make_datarray_spot(
                centroids,
                estimated_intensity,
                estimated_background,
                estimated_background_perpixel,
                columns[:-2],
            )
            if bulk_mask is not None and to_save is not None:
                overlap_bulk = self._overlap_with_mask(centroids, bulk_mask)
                to_save = to_save.with_columns(
                    pl.Series("overlap_with_bulk_stain", overlap_bulk.astype(bool))
                )
            to_save_largeobjects = HF.make_datarray_largeobjects(
                areas_large,
                centroids_large,
                sumintensities_large,
                meanintensities_large,
                columns_large[:-2],
                stdintensities_large=stdintensities_large,
                peakintensities_large=peakintensities_large,
            )

        return to_save, to_save_largeobjects, lo_mask, cell_mask, bulk_mask
