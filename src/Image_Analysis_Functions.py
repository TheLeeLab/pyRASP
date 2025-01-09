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
from skimage.filters import gaussian
from skimage.measure import label, regionprops_table
import skimage.draw as draw
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
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

HF = HelperFunctions.Helper_Functions()
IO = IOFunctions.IO_Functions()
cpu_number = int(pathos.helpers.cpu_count() * 0.9)


class ImageAnalysis_Functions:
    def __init__(self):
        self = self
        return

    def calculate_gradient_field(self, image, kernel, FS=True):
        """
        Calculate the gradient field of an image and compute focus-related measures.

        Args:
            image (numpy.ndarray): The input image.
            kernel (numpy.ndarray): The kernel for low-pass filtering.

        Returns:
            filtered_image (numpy.ndarray): Image after low-pass filtering.
            gradient_x (numpy.ndarray): X-gradient of the filtered image.
            gradient_y (numpy.ndarray): Y-gradient of the filtered image.
            focus_score (numpy.ndarray): Focus score of the image.
            concentration_factor (numpy.ndarray): Concentration factor of the image.
        """
        # Initialize variables
        filtered_image = np.zeros_like(image)
        gradient_x = np.zeros_like(image)
        gradient_y = np.zeros_like(image)

        # Low-pass filtering using convolution
        if len(image.shape) > 2:
            for channel in np.arange(image.shape[2]):
                image_padded = np.pad(
                    image[:, :, channel],
                    (
                        (kernel.shape[0] // 2, kernel.shape[0] // 2),
                        (kernel.shape[1] // 2, kernel.shape[1] // 2),
                    ),
                    mode="edge",
                )
                filtered_image[:, :, channel] = fftconvolve(
                    image_padded, kernel, mode="valid"
                )
        else:
            image_padded = np.pad(
                image,
                (
                    (kernel.shape[0] // 2, kernel.shape[0] // 2),
                    (kernel.shape[1] // 2, kernel.shape[1] // 2),
                ),
                mode="edge",
            )
            filtered_image[:, :] = fftconvolve(image_padded, kernel, mode="valid")
        # Gradient calculation
        if len(image.shape) > 2:
            gradient_x[:, :-1, :] = np.diff(
                filtered_image, axis=1
            )  # x gradient (right to left)
            gradient_y[:-1, :, :] = np.diff(
                filtered_image, axis=0
            )  # y gradient (bottom to top)
        else:
            gradient_x[:, :-1] = np.diff(
                filtered_image, axis=1
            )  # x gradient (right to left)
            gradient_y[:-1, :] = np.diff(
                filtered_image, axis=0
            )  # y gradient (bottom to top)

        if FS == True:
            gradient_magnitude = np.sqrt(
                np.add(np.square(gradient_x), np.square(gradient_y))
            )
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
            img (numpy.ndarray): The input image.
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

        g2 = np.sqrt(np.add(np.square(gradient_x[x, y]), np.square(gradient_y[x, y])))

        flatness = np.mean(np.divide(img[x, y].T, r0), axis=0)
        integrated_grad = np.sum(g2, axis=1)
        radiality = np.vstack([flatness, integrated_grad]).T

        return radiality

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

    @staticmethod
    @jit(nopython=True)
    def infocus_indices(focus_scores, threshold_differential):
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
            image (numpy.ndarray): Input image.
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

        estimated_background = np.mean(image[y_out, x_out], axis=0)
        estimated_intensity = np.sum(
            np.subtract(image[y_in, x_in], estimated_background), axis=0
        )

        estimated_intensity[estimated_intensity < 0] = np.NAN
        estimated_background[estimated_background < 0] = np.NAN

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
        inner_ind = np.zeros_like(outer_ind)
        inner_ind[1:-1, 1:-1] = small_oct
        outer_ind -= inner_ind

        x_inner, y_inner = calculate_offsets(inner_ind)
        x_outer, y_outer = calculate_offsets(outer_ind)

        x_inner = np.tile(x_inner, (len(centroid_loc), 1)).T + centroid_loc[:, 0]
        y_inner = np.tile(y_inner, (len(centroid_loc), 1)).T + centroid_loc[:, 1]
        x_outer = np.tile(x_outer, (len(centroid_loc), 1)).T + centroid_loc[:, 0]
        y_outer = np.tile(y_outer, (len(centroid_loc), 1)).T + centroid_loc[:, 1]

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

    def calculate_region_properties(self, binary_mask, image=None):
        """
        Calculate properties for labeled regions in a binary mask.

        Args:
            binary_mask (numpy.ndarray): Binary mask of connected components.
            image (numpy.ndarray): image of same dimension as binary mask. Optional.

        Returns:
            pixel_index_list (list): List containing pixel indices for each labeled object.
            areas (numpy.ndarray): Array containing areas of each labeled object.
            centroids (numpy.ndarray): Array containing centroids (x, y) of each labeled object.
        """
        # Find connected components and count the number of objects
        labeled_image, num_objects = label(binary_mask, connectivity=2, return_num=True)
        # Initialize arrays for storing properties
        centroids = np.zeros((num_objects, 2))

        if image is not None:
            properties = ("centroid", "area", "coords", "intensity_mean")
        else:
            properties = ("centroid", "area", "coords")

        # Get region properties
        props = regionprops_table(
            labeled_image, intensity_image=image, properties=properties
        )
        centroids[:, 0] = props["centroid-1"]
        centroids[:, 1] = props["centroid-0"]
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

        BW = np.zeros_like(img1, dtype=bool)
        if thres < 1:
            thres = np.percentile(img1.ravel(), 100 * (1 - thres))
        BW[img1 > thres] = 1
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
                    large_mask = self.detect_large_features(
                        image, large_thres
                    )

                dl_mask, centroids, radiality, idxs = (
                    self.small_feature_kernel(
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
                )
                return dl_mask, centroids, radiality, idxs

            image_planes = [image[:, :, i] for i in range(image.shape[-1])]
            planes_img2 = [img2[:, :, i] for i in range(img2.shape[-1])]
            planes_Gx = [Gx[:, :, i] for i in range(Gx.shape[-1])]
            planes_Gy = [Gy[:, :, i] for i in range(Gy.shape[-1])]

            pool = Pool(nodes=cpu_number)
            pool.restart()
            results = pool.map(
                run_over_z,
                image_planes,
                planes_img2,
                planes_Gx,
                planes_Gy,
            )
            pool.close()
            pool.terminate()
            
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
        cell_threshold1=100.0,
        cell_threshold2=200,
        cell_sigma1=2.0,
        cell_sigma2=40.0,
        d=2,
    ):
        """
        Gets basic image properties (centroids, radiality)
        from a single image and generates cell mask from another image channel

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
            cell_threshold1 (float): 1st cell intensity threshold
            cell_threshold2 (float): 2nd cell intensity threshold
            cell_sigma1 (float): cell blur value 1
            cell_sigma2 (float): cell blur value 2
            d (integer): pixel radius value

        Returns:
            to_save (pl.DataFrame): spot properties ready to save
            to_save_largeobjects (pl.DataFrame): large object properties ready to save
            lo_mask (np.ndarray): lo masks
            cell_mask (np.ndarray): cell mask
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
            "zi",
            "zf",
        ]

        if not isinstance(z, int):
            z_planes = np.arange(z[0], z[1])

            def analyse_zplanes(
                zp, image_plane, img2_plane, Gx_plane, Gy_plane, imagecell_plane
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

                if imagecell_plane is not None:
                    cell_mask = self.detect_large_features(
                        imagecell_plane,
                        cell_threshold1,
                        cell_threshold2,
                        cell_sigma1,
                        cell_sigma2,
                    )
                else:
                    cell_mask = None
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
                    cell_mask,
                )

            planes = [image[:, :, i] for i in range(image.shape[-1])]
            planes_img2 = [img2[:, :, i] for i in range(img2.shape[-1])]
            planes_Gx = [Gx[:, :, i] for i in range(Gx.shape[-1])]
            planes_Gy = [Gy[:, :, i] for i in range(Gy.shape[-1])]
            if image_cell is not None:
                planes_imagecell = [
                    image_cell[:, :, i] for i in range(image_cell.shape[-1])
                ]
            else:
                planes_imagecell = [None] * image.shape[-1]

            pool = Pool(nodes=cpu_number)
            pool.restart()
            results = pool.map(
                analyse_zplanes,
                np.arange(image.shape[-1]),
                planes,
                planes_img2,
                planes_Gx,
                planes_Gy,
                planes_imagecell,
            )
            pool.close()
            pool.terminate()

            centroids = [i[0] for i in results]
            estimated_intensity = [i[1] for i in results]
            estimated_background = [i[2] for i in results]
            estimated_background_perpixel = [i[3] for i in results]
            areas_large = [i[4] for i in results]
            centroids_large = [i[5] for i in results]
            meanintensities_large = [i[6] for i in results]
            sumintensities_large = [i[7] for i in results]
            lo_mask = np.dstack([i[8] for i in results])

            if image_cell is not None:
                cell_mask = np.dstack([i[9] for i in results])
            else:
                cell_mask = None

            to_save = HF.make_datarray_spot(
                centroids,
                estimated_intensity,
                estimated_background,
                estimated_background_perpixel,
                columns,
                np.arange(len(z_planes)),
            )
            to_save_largeobjects = HF.make_datarray_largeobjects(
                areas_large,
                centroids_large,
                sumintensities_large,
                meanintensities_large,
                columns_large,
                np.arange(len(z_planes)),
            )
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
            cell_mask = (
                self.detect_large_features(
                    image_cell,
                    cell_threshold1,
                    cell_threshold2,
                    cell_sigma1,
                    cell_sigma2,
                )
                if image_cell is not None
                else None
            )

            to_save = HF.make_datarray_spot(
                centroids,
                estimated_intensity,
                estimated_background,
                estimated_background_perpixel,
                columns[:-2],
            )
            to_save_largeobjects = HF.make_datarray_largeobjects(
                areas_large,
                centroids_large,
                sumintensities_large,
                meanintensities_large,
                columns_large[:-2],
            )

        return to_save, to_save_largeobjects, lo_mask, cell_mask
