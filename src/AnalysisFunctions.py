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
from pathos.pools import ThreadPool as Pool
from rdfpy import rdf
import time

import MultiD_RD_functions

import os
import sys

module_dir = os.path.dirname(__file__)
sys.path.append(module_dir)
import IOFunctions

IO = IOFunctions.IO_Functions()
cpu_number = int(pathos.helpers.cpu_count() * 0.8)


class Analysis_Functions:
    def __init__(self):
        self = self
        return

    def count_spots(self, database, z_planes):
        """
        Counts spots per z plane

        Args:
            database (polars DataFrame): pandas array of spots
            z_planes (np.1darray): is range of zplanes

        Returns:
            n_spots (polars DataFrame)
        """

        spots_per_plane = np.zeros_like(z_planes)
        for z in enumerate(z_planes):
            spots_per_plane[z[0]] = len(database["z"].to_numpy() == (z[1] + 1))

        data = {"z": z_planes + 1, "n_spots": spots_per_plane}
        n_spots = pl.DataFrame(data)
        return n_spots

    def count_spots_withthreshold(self, database, threshold):
        """
        Counts spots per z plane

        Args:
            database (pandas array): pandas array of spots
            threshold (float): intensity threshold

        Returns:
            n_spots (pandas array)
        """
        columns = [
            "z",
            "n_spots_abovethreshold",
            "n_spots_belowthreshold",
            "filename",
            "threshold",
        ]

        for i, filename in enumerate(np.unique(database.image_filename)):
            dataslice = database.filter(pl.col("image_filename") == filename)
            z_planes = np.unique(dataslice["z"].to_numpy())
            spots_per_plane = np.zeros([2, len(z_planes)])
            for z in enumerate(z_planes):
                spots_per_plane[0, z[0]] = sum(
                    dataslice.filter(pl.col("z") == z[1])[
                        "sum_intensity_in_photons"
                    ].to_numpy()
                    > threshold
                )
                spots_per_plane[1, z[0]] = sum(
                    dataslice.filter(pl.col("z") == z[1])[
                        "sum_intensity_in_photons"
                    ].to_numpy()
                    <= threshold
                )

            stack = np.vstack(
                [
                    z_planes,
                    spots_per_plane[0, :],
                    spots_per_plane[1, :],
                    np.full_like(z_planes, filename, dtype="object"),
                    np.full_like(z_planes, threshold),
                ]
            ).T
            if i == 0:
                data = stack
            else:
                data = np.vstack([data, stack])
        n_spots = pl.DataFrame(data=data, schema=columns)
        return n_spots

    def calculate_gradient_field(self, image, kernel):
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

        gradient_magnitude = np.sqrt(
            np.add(np.square(gradient_x), np.square(gradient_y))
        )
        sum_gradient = np.sum(gradient_magnitude, axis=(0, 1))
        concentration_factor = np.divide(sum_gradient, np.max(sum_gradient))
        focus_score = np.log(sum_gradient)

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

    def generate_largeaggregate_indices(self, pil, image_size):
        """
        makes large aggregate indices from list of xy coordinates

        Args:
            pil (list): list of 2d arrays
            image_size (tuple): Image dimensions (height, width).

        Returns:
            la_indices (list): list of indices of large aggregates
        """
        la_indices = []
        for i in np.arange(len(pil)):
            if i == 0:
                la_indices = [
                    np.ravel_multi_index(
                        [pil[i][:, 0], pil[i][:, 1]], image_size, order="F"
                    )
                ]
            else:
                la_indices.append(
                    np.ravel_multi_index(
                        [pil[i][:, 0], pil[i][:, 1]], image_size, order="F"
                    )
                )
        return la_indices

    def generate_mask_and_spot_indices(self, mask, centroids, image_size):
        """
        makes mask and spot indices from xy coordinates

        Args:
            mask (2D array): boolean matrix
            centroids (2D array): xy centroid coordinates
            image_size (tuple): Image dimensions (height, width).

        Returns:
            mask_indices (1D array): indices of mask
            spot_indices (1D array): indices of spots
        """
        mask_coords = np.transpose((mask > 0).nonzero())
        mask_indices = np.ravel_multi_index(
            [mask_coords[:, 0], mask_coords[:, 1]], image_size, order="F"
        )
        spot_indices = np.ravel_multi_index(centroids.T, image_size, order="F")
        return mask_indices, spot_indices

    def generate_spot_and_spot_indices(self, centroids1, centroids2, image_size):
        """
        makes mask and spot indices from xy coordinates

        Args:
            centroids1 (2D array): xy centroid coordinates
            centroids2 (2D array): xy centroid coordinates
            image_size (tuple): Image dimensions (height, width).

        Returns:
            spot1_indices (1D array): indices of spots1
            spot2_indices (1D array): indices of spots2
        """
        spot1_indices = np.ravel_multi_index(centroids1.T, image_size, order="F")
        spot2_indices = np.ravel_multi_index(centroids2.T, image_size, order="F")
        return spot1_indices, spot2_indices

    def calculate_spot_to_mask_coincidence(
        self, spot_indices, mask_indices, image_size, n_iter=100, blur_degree=1
    ):
        """
        gets spot colocalisation likelihood ratio, as well as reporting error
        bounds on the likelihood ratio for one image

        Args:
            spot_indices (1D array): indices of spots
            mask_indices (1D array): indices of pixels in mask
            image_size (tuple): Image dimensions (height, width).
            n_iter (int): default 100; number of iterations to start with
            blur_degree (int): number of pixels to blur spot indices with
                                (i.e. number of pixels surrounding centroid to
                                consider part of spot). Default 1

        Returns:
            coincidence (float): coincidence FoV
            chance_coincidence (float): chance coincidence
            raw_colocalisation (np.1darray): binary yes/no of coincidence per spot
            n_iter (int): how many iterations it took to converge
        """
        original_n_spots = len(spot_indices)  # get number of spots

        if original_n_spots == 0:
            n_iter_rec = 0
            coincidence = np.NAN
            chance_coincidence = np.NAN
            raw_colocalisation = np.full_like(spot_indices, np.NAN)
            return coincidence, chance_coincidence, raw_colocalisation, n_iter_rec

        if blur_degree > 0:
            spot_indices = self.dilate_pixels(
                spot_indices, image_size, width=blur_degree + 1, edge=blur_degree
            )
        n_spots = len(spot_indices)
        n_iter_rec = n_iter
        possible_indices = np.arange(
            0, np.prod(image_size)
        )  # get list of where is possible to exist in an image
        raw_colocalisation = self.test_spot_spot_overlap(
            spot_indices, mask_indices, original_n_spots, raw=True
        )
        nspots_in_mask = self.test_spot_mask_overlap(
            spot_indices, mask_indices
        )  # get nspots in mask
        coincidence = np.divide(nspots_in_mask, n_spots)  # generate coincidence

        random_spot_locations = np.random.choice(
            possible_indices, size=(n_iter, original_n_spots)
        )  # get random spot locations
        if blur_degree > 0:
            random_spot_locations = self.dilate_pixel_matrix(
                random_spot_locations,
                image_size,
                width=blur_degree + 1,
                edge=blur_degree,
            )
        chance_coincidence_raw = np.zeros([n_iter])  # generate CSR array to fill in
        for i in np.arange(n_iter):
            chance_coincidence_raw[i] = self.test_spot_mask_overlap(
                random_spot_locations[i, :], mask_indices
            )

        chance_coincidence = np.divide(np.nanmean(chance_coincidence_raw), n_spots)
        return coincidence, chance_coincidence, raw_colocalisation, n_iter_rec

    def calculate_spot_to_spot_coincidence(
        self, spot_1_indices, spot_2_indices, image_size, n_iter=1000, blur_degree=1
    ):
        """
        gets spot to spot coincidence between two channels

        Args:
            spot_1_indices (1D array): indices of spots 1
            spot_2_indices (1D array): indices of spots 2
            image_size (tuple): Image dimensions (height, width).
            n_iter (int): default 1000; number of iterations to get chance coincidence
            blur_degree (int): number of pixels to blur spot indices with
                                (i.e. number of pixels surrounding centroid to
                                consider part of spot). Default 1

        Returns:
            coincidence_1 (float): coincidence from 1 to 2 FoV
            chance_coincidence_1 (float): chance coincidence from 1 to 2
            coincidence_2 (float): coincidence from 2 to 1 FoV
            chance_coincidence_2 (float): chance coincidence from 2 to 1
            raw_coincidence_1 (np.1darray): raw colocalisation values per spot 1 to 2
            raw_coincidence_2 (np.1darray): raw colocalisation values per spot 2 to 1
        """
        original_n_spots_1 = len(spot_1_indices)  # get number of spots 1
        original_n_spots_2 = len(spot_2_indices)  # get number of spots 2

        if (original_n_spots_1 == 0) or (original_n_spots_2 == 0):
            coincidence_1 = np.NAN
            chance_coincidence_1 = np.NAN
            coincidence_2 = np.NAN
            chance_coincidence_2 = np.NAN
            raw_coincidence_1 = np.full_like(spot_1_indices, np.NAN)
            raw_coincidence_2 = np.full_like(spot_1_indices, np.NAN)
            return (
                coincidence_1,
                chance_coincidence_1,
                coincidence_2,
                chance_coincidence_2,
                raw_coincidence_1,
                raw_coincidence_2,
            )

        if blur_degree > 0:
            spot_1_indices = self.dilate_pixels(
                spot_1_indices, image_size, width=blur_degree + 1, edge=blur_degree
            )
            spot_2_indices = self.dilate_pixels(
                spot_2_indices, image_size, width=blur_degree + 1, edge=blur_degree
            )

        possible_indices = np.arange(
            0, np.prod(image_size)
        )  # get list of where is possible to exist in an image
        raw_coincidence_1 = self.test_spot_spot_overlap(
            spot_1_indices, spot_2_indices, original_n_spots_1
        )  # get nspots in mask
        raw_coincidence_1[raw_coincidence_1 >= 1.0] = 1.0
        raw_coincidence_2 = self.test_spot_spot_overlap(
            spot_2_indices, spot_1_indices, original_n_spots_2
        )  # get nspots in mask
        raw_coincidence_2[raw_coincidence_2 >= 1.0] = 1.0

        coincidence_1 = np.divide(
            np.sum(raw_coincidence_1), original_n_spots_1
        )  # generate coincidence
        coincidence_2 = np.divide(
            np.sum(raw_coincidence_2), original_n_spots_2
        )  # generate coincidence

        random_spot_locations_1 = np.random.choice(
            possible_indices, size=(n_iter, original_n_spots_1)
        )  # get random spot locations
        random_spot_locations_2 = np.random.choice(
            possible_indices, size=(n_iter, original_n_spots_2)
        )  # get random spot locations

        if blur_degree > 0:
            random_spot_locations_1 = self.dilate_pixel_matrix(
                random_spot_locations_1,
                image_size,
                width=blur_degree + 1,
                edge=blur_degree,
            )
            random_spot_locations_2 = self.dilate_pixel_matrix(
                random_spot_locations_2,
                image_size,
                width=blur_degree + 1,
                edge=blur_degree,
            )

        CC_1 = np.zeros([n_iter])  # generate CSR array to fill in
        CC_2 = np.zeros([n_iter])  # generate CSR array to fill in
        for i in np.arange(n_iter):
            rc_1 = self.test_spot_spot_overlap(
                random_spot_locations_1[i, :],
                spot_2_indices,
                original_n_spots_1,
            )
            rc_1[rc_1 >= 1.0] = 1.0
            CC_1[i] = np.divide(
                np.sum(rc_1),
                original_n_spots_1,
            )
            rc_2 = self.test_spot_spot_overlap(
                random_spot_locations_2[i, :],
                spot_1_indices,
                original_n_spots_2,
            )
            rc_2[rc_2 >= 1.0] = 1.0
            CC_2[i] = np.divide(
                np.sum(rc_2),
                original_n_spots_2,
            )

        chance_coincidence_1 = np.nanmean(CC_1)
        chance_coincidence_2 = np.nanmean(CC_2)

        return (
            coincidence_1,
            chance_coincidence_1,
            coincidence_2,
            chance_coincidence_2,
            raw_coincidence_1,
            raw_coincidence_2,
        )

    def calculate_largeobj_coincidence(
        self, largeobj_indices, mask_indices, n_largeobjs, image_size
    ):
        """
        gets spot colocalisation likelihood ratio, as well as reporting error
        bounds on the likelihood ratio for one image

        Args:
            largeobj_indices (1D array): indices of large objects
            mask_indices (1D array): indices of pixels in mask
            image_size (tuple): Image dimensions (height, width).

        Returns:
            coincidence (float): coincidence FoV
            chance_coincidence (float): chance coincidence
            raw_colocalisation (np.1darray): binary yes/no of coincidence per spot
        """

        if n_largeobjs == 0:
            coincidence = 0
            chance_coincidence = 0
            raw_colocalisation = np.full_like(largeobj_indices, np.NAN)
            return coincidence, chance_coincidence, raw_colocalisation

        highest_index = np.prod(image_size)
        mask_fill = self.calculate_mask_fill(mask_indices, image_size)  # get mask_fill
        expected_spots = np.multiply(
            mask_fill, n_largeobjs
        )  # get expected number of spots
        if np.isclose(expected_spots, 0.0, atol=1e-4):
            coincidence = 0
            chance_coincidence = 0
            raw_colocalisation = np.zeros(n_largeobjs)
            return coincidence, chance_coincidence, raw_colocalisation
        else:
            raw_colocalisation = self.test_largeobj_mask_overlap(
                largeobj_indices, mask_indices, n_largeobjs, raw=True
            )
            nspots_in_mask = np.sum(raw_colocalisation)
            coincidence = np.divide(nspots_in_mask, n_largeobjs)  # generate coincidence

            n_iter = 100
            random_spot_locations = self.random_largeobj_locations(
                largeobj_indices, highest_index, n_iter
            )  # need to write large aggregate randomisation function
            chance_coincidence_raw = np.zeros(n_iter)
            for i in np.arange(n_iter):
                chance_coincidence_raw[i] = np.divide(
                    self.test_largeobj_mask_overlap(
                        random_spot_locations[i], mask_indices, n_largeobjs, raw=False
                    ),
                    n_largeobjs,
                )
            return coincidence, np.mean(chance_coincidence_raw), raw_colocalisation

    def calculate_spot_colocalisation_likelihood_ratio(
        self,
        spot_indices,
        mask_indices,
        image_size,
        tol=0.01,
        n_iter=100,
        blur_degree=1,
        max_iter=1000,
    ):
        """
        gets spot colocalisation likelihood ratio, as well as reporting error
        bounds on the likelihood ratio for one image

        Args:
            spot_indices (1D array): indices of spots
            mask_indices (1D array): indices of pixels in mask
            image_size (tuple): Image dimensions (height, width).
            tol (float): default 0.01; tolerance for convergence
            n_iter (int): default 100; number of iterations to start with
            blur_degree (int): number of pixels to blur spot indices with
                                (i.e. number of pixels surrounding centroid to
                                consider part of spot). Default 1

        Returns:
            colocalisation_likelihood_ratio (float): likelihood ratio of spots for mask
            perc_std (float): standard deviation on this CLR based on bootstrapping
            meanCSR (float): mean of randomised spot data
            expected_spots (float): number of spots we expect based on mask % of image
            coincidence (float): coincidence FoV
            chance_coincidence (float): chance coincidence
            raw_colocalisation (np.1darray): binary yes/no of coincidence per spot
            n_iter (int): how many iterations it took to converge
        """
        original_n_spots = len(spot_indices)  # get number of spots

        if original_n_spots == 0:
            n_iter_rec = 0
            colocalisation_likelihood_ratio = np.NAN
            norm_CSR = np.NAN
            norm_std = np.NAN
            coincidence = np.NAN
            chance_coincidence = np.NAN
            raw_colocalisation = np.full(original_n_spots, np.NAN)
            return (
                colocalisation_likelihood_ratio,
                norm_std,
                norm_CSR,
                0,
                coincidence,
                chance_coincidence,
                raw_colocalisation,
                n_iter_rec,
            )

        if blur_degree > 0:
            spot_indices = self.dilate_pixels(
                spot_indices, image_size, width=blur_degree + 1, edge=blur_degree
            )
        n_spots = len(spot_indices)
        n_iter_rec = n_iter
        possible_indices = np.arange(
            0, np.prod(image_size)
        )  # get list of where is possible to exist in an image
        mask_fill = self.calculate_mask_fill(mask_indices, image_size)  # get mask_fill
        expected_spots_iter = np.multiply(
            mask_fill, n_spots
        )  # get expected number of spots
        expected_spots = np.multiply(
            mask_fill, original_n_spots
        )  # get expected number of spots
        if np.isclose(expected_spots_iter, 0.0, atol=1e-4):
            n_iter_rec = 0
            colocalisation_likelihood_ratio = np.NAN
            norm_CSR = np.NAN
            norm_std = np.NAN
            coincidence = np.NAN
            chance_coincidence = np.NAN
            raw_colocalisation = np.full(original_n_spots, np.NAN)
            return (
                colocalisation_likelihood_ratio,
                norm_std,
                norm_CSR,
                expected_spots,
                coincidence,
                chance_coincidence,
                raw_colocalisation,
                n_iter_rec,
            )
        else:
            raw_colocalisation = self.test_spot_spot_overlap(
                spot_indices, mask_indices, original_n_spots, raw=True
            )
            nspots_in_mask = self.test_spot_mask_overlap(
                spot_indices, mask_indices
            )  # get nspots in mask
            colocalisation_likelihood_ratio = np.divide(
                nspots_in_mask, expected_spots_iter
            )  # generate colocalisation likelihood ratio
            coincidence = np.divide(nspots_in_mask, n_spots)  # generate coincidence

            random_spot_locations = np.random.choice(
                possible_indices, size=(n_iter, original_n_spots)
            )  # get random spot locations
            if blur_degree > 0:
                random_spot_locations = self.dilate_pixel_matrix(
                    random_spot_locations,
                    image_size,
                    width=blur_degree + 1,
                    edge=blur_degree,
                )
            CSR = np.zeros([n_iter])  # generate CSR array to fill in
            for i in np.arange(n_iter):
                CSR[i] = self.test_spot_mask_overlap(
                    random_spot_locations[i, :], mask_indices
                )

            meanCSR = np.divide(
                np.nanmean(CSR), expected_spots_iter
            )  # should be close to 1
            chance_coincidence = np.divide(np.nanmean(CSR), n_spots)
            CSR_diff = np.abs(meanCSR - 1.0)
            while (CSR_diff > tol) and (
                n_iter_rec < max_iter
            ):  # do n_iter more tests iteratively until convergence
                n_iter_rec = n_iter_rec + n_iter  # add n_iter iterations
                CSR_new = np.zeros([n_iter])
                random_spot_locations = np.random.choice(
                    possible_indices, size=(n_iter, original_n_spots)
                )  # get random spot locations
                if blur_degree > 0:
                    random_spot_locations = self.dilate_pixel_matrix(
                        random_spot_locations,
                        image_size,
                        width=blur_degree + 1,
                        edge=blur_degree,
                    )
                for i in np.arange(n_iter):
                    CSR_new[i] = self.test_spot_mask_overlap(
                        random_spot_locations[i, :], mask_indices
                    )
                CSR = np.hstack([CSR, CSR_new])  # stack
                meanCSR = np.divide(
                    np.nanmean(CSR), expected_spots_iter
                )  # should be close to 1
                CSR_diff = np.abs(meanCSR - 1.0)
            if (expected_spots > 0) and (np.mean(CSR) > 0):
                norm_CSR = np.divide(
                    np.nanmean(CSR), expected_spots_iter
                )  # should be close to 1
                norm_std = np.divide(
                    np.nanstd(CSR), np.nanmean(CSR)
                )  # std dev (normalised)
            else:
                norm_CSR = np.NAN
                norm_std = np.NAN
            return (
                colocalisation_likelihood_ratio,
                norm_std,
                norm_CSR,
                expected_spots,
                coincidence,
                chance_coincidence,
                raw_colocalisation,
                n_iter_rec,
            )

    def default_spotanalysis_routine(
        self,
        image,
        k1,
        k2,
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
            pil_large (list of np.2darray): pixels where large objects found

        """
        large_mask = self.detect_large_features(image, large_thres)
        pil_large, areas_large, centroids_large = self.calculate_region_properties(
            large_mask
        )
        to_keep = np.where(areas_large > areathres)[0]
        pil_large = pil_large[to_keep]
        areas_large = areas_large[to_keep]
        centroids_large = centroids_large[to_keep, :]
        meanintensities_large = np.zeros_like(areas_large)
        sumintensities_large = np.zeros_like(areas_large)
        for i in np.arange(len(centroids_large)):
            sumintensities_large[i] = np.sum(
                image[pil_large[i][:, 0], pil_large[i][:, 1]]
            )
        meanintensities_large = np.divide(sumintensities_large, areas_large)
        img2, Gx, Gy, focusScore, cfactor = self.calculate_gradient_field(image, k1)
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
            pil_large,
        ]

        return to_return

    @staticmethod
    @jit(nopython=True)
    def calculate_mask_fill(mask_indices, image_size):
        """
        calculate amount of image filled by mask.

        Args:
            mask_indices (1D array): indices of pixels in mask
            image_size (tuple): Image dimensions (height, width).

        Returns:
            mask_fill (float): proportion of image filled by mask.
        """

        mask_fill = np.divide(len(mask_indices), np.prod(np.array(image_size)))
        return mask_fill

    def random_largeobj_locations(self, largeobj_indices, possible_index_high, n_iter):
        """
        Randomises large object locations.

        Args:
            largeobj_indices (list): indices of large objects
            possible_index_high (int): highest possible index
            n_iter (int): number of random permutations to make

        Returns:
            largeobj_locations (dict): dict of n_iter in size of randomised large object locations.
        """
        largeobj_locations = {}

        for i in np.arange(n_iter):
            largeobj_locations[i] = []
            for j in np.arange(len(largeobj_indices)):
                if j == 0:
                    high = possible_index_high - np.max(largeobj_indices[j])
                    low = -np.min(largeobj_indices[j])
                    largeobj_locations[i] = [
                        largeobj_indices[j]
                        + np.random.randint(low=low, high=high, size=1)
                    ]
                else:
                    high = possible_index_high - np.max(largeobj_indices[j])
                    low = -np.min(largeobj_indices[j])
                    largeobj_locations[i].append(
                        largeobj_indices[j]
                        + np.random.randint(low=low, high=high, size=1)
                    )
        return largeobj_locations

    def test_largeobj_mask_overlap(
        self, largeobj_indices, mask_indices, n_largeobjects, raw=False
    ):
        """
        Tests which spots overlap with a given mask.

        Args:
            largeobj_indices (list): indices of large objects
            mask_indices (1D array): indices of mask
            n_largeobjects (int): n large objects
            raw (boolean): if raw is true, will report "per object" overlap

        Returns:
            n_large_objects_in_mask (float): number of spots that overlap with the other spots.
        """
        n_large_objects_in_mask = np.zeros(n_largeobjects)
        for i in np.arange(n_largeobjects):
            n_large_objects_in_mask[i] = np.asarray(
                np.sum(np.isin(largeobj_indices[i], mask_indices)), dtype=bool
            )
        if raw == True:
            return n_large_objects_in_mask
        else:
            return np.sum(n_large_objects_in_mask)

    def test_spot_spot_overlap(
        self, spot_1_indices, spot_2_indices, n_spot1, raw=False
    ):
        """
        Tests which spots overlap with a given mask.

        Args:
            spot_1_indices (1D array): indices of spots 1
            spot_2_indices (1D array): indices of spots 2
            n_spots1_in_spots2 (int): n spots in spot 1 array
            raw (boolean): if raw is true, will report "per spot" overlap

        Returns:
            n_spots1_in_spots2 (float): number of spots that overlap with the other spots.
        """
        newdims = (n_spot1, int(len(spot_1_indices) / n_spot1))
        if raw == False:
            n_spots1_in_spots2 = np.sum(
                np.isin(spot_1_indices.reshape(newdims), spot_2_indices), axis=1
            )
        else:
            n_spots1_in_spots2 = np.asarray(
                np.sum(
                    np.isin(spot_1_indices.reshape(newdims), spot_2_indices), axis=1
                ),
                dtype=bool,
            )
        return n_spots1_in_spots2

    def test_spot_mask_overlap(self, spot_indices, mask_indices):
        """
        Tests which spots overlap with a given mask.

        Args:
            spot_indices (1D array): indices of spots
            mask_indices (1D array): indices of pixels in mask

        Returns:
            n_spots_in_mask (float): number of spots that overlap with the mask.
        """

        n_spots_in_mask = np.sum(np.isin(mask_indices, np.unique(spot_indices)))
        return n_spots_in_mask

    def dilate_pixel_matrix(self, index_matrix, image_size, width=5, edge=1):
        """
        Dilate a pixel matrix index to form a matrix of neighbourhoods.

        Args:
            indices (np.2darray): 2D Array of pixel indices. Second dimension
            is number of spots.
            image_size (tuple): Image dimensions (height, width).
            width: width of dilation (default 5)
            edge: edge of dilation (default 1)

        Returns:
            dilated_indices (numpy.ndarray): Dilated pixel indices forming a neighborhood.
        """
        x, y = np.where(ski.morphology.octagon(width, edge))
        x = x - int(ski.morphology.octagon(width, edge).shape[0] / 2)
        y = y - int(ski.morphology.octagon(width, edge).shape[1] / 2)
        centroid = np.asarray(
            np.unravel_index(index_matrix, image_size, order="F"), dtype=int
        )

        x = (
            np.tile(x, (len(index_matrix), 1)).T[:, :, np.newaxis]
            + np.asarray(centroid[0, :], dtype=int)
        ).T

        y = (
            np.tile(y, (len(index_matrix), 1)).T[:, :, np.newaxis]
            + np.asarray(centroid[1, :], dtype=int)
        ).T

        new_dims = (index_matrix.shape[0], int(len(x.ravel()) / index_matrix.shape[0]))
        dilated_index_matrix = np.ravel_multi_index(
            np.vstack([x.ravel(), y.ravel()]), image_size, order="F", mode="wrap"
        ).reshape(new_dims)

        return dilated_index_matrix

    def dilate_pixels(self, indices, image_size, width=5, edge=1):
        """
        Dilate a pixel index to form a neighborhood.

        Args:
            indices (np.1darray): Array of pixel indices.
            image_size (tuple): Image dimensions (height, width).
            width: width of dilation (default 5)
            edge: edge of dilation (default 1)

        Returns:
            dilated_indices (numpy.ndarray): Dilated pixel indices forming a neighborhood.
        """
        x, y = np.where(ski.morphology.octagon(width, edge))
        x = x - int(ski.morphology.octagon(width, edge).shape[0] / 2)
        y = y - int(ski.morphology.octagon(width, edge).shape[1] / 2)
        centroid = np.asarray(
            np.unravel_index(indices, image_size, order="F"), dtype=int
        )

        x = (np.tile(x, (len(indices), 1)).T + np.asarray(centroid[0, :], dtype=int)).T

        y = (np.tile(y, (len(indices), 1)).T + np.asarray(centroid[1, :], dtype=int)).T

        new_dims = (len(indices), int(len(x.ravel()) / len(indices)))
        dilated_indices = np.ravel_multi_index(
            np.vstack([x.ravel(), y.ravel()]), image_size, order="F", mode="wrap"
        ).reshape(new_dims)

        if len(indices) == 1:
            return dilated_indices[0]
        else:
            return dilated_indices.ravel()

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
        x = np.arange(-length, length + 1)
        y = np.arange(-length, length + 1)
        X, Y = np.meshgrid(x, y)
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

        small_oct = ski.morphology.octagon(2, 4)
        outer_ind = ski.morphology.octagon(2, 5)
        inner_ind = np.zeros_like(outer_ind)
        inner_ind[1:-1, 1:-1] = small_oct
        outer_ind = outer_ind - inner_ind
        x_inner, y_inner = np.where(inner_ind)
        x_inner = x_inner - int(inner_ind.shape[0] / 2)
        y_inner = y_inner - int(inner_ind.shape[1] / 2)

        x_outer, y_outer = np.where(outer_ind)
        x_outer = x_outer - int(outer_ind.shape[0] / 2)
        y_outer = y_outer - int(outer_ind.shape[1] / 2)

        x_inner = np.tile(x_inner, (len(centroid_loc[:, 0]), 1)).T + centroid_loc[:, 0]
        y_inner = np.tile(y_inner, (len(centroid_loc[:, 1]), 1)).T + centroid_loc[:, 1]
        x_outer = np.tile(x_outer, (len(centroid_loc[:, 0]), 1)).T + centroid_loc[:, 0]
        y_outer = np.tile(y_outer, (len(centroid_loc[:, 1]), 1)).T + centroid_loc[:, 1]

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

            # add in pixel threshold?

        return large_mask.astype(bool)

    def calculate_region_properties(self, binary_mask):
        """
        Calculate properties for labeled regions in a binary mask.

        Args:
            binary_mask (numpy.ndarray): Binary mask of connected components.

        Returns:
            pixel_index_list (list): List containing pixel indices for each labeled object.
            areas (numpy.ndarray): Array containing areas of each labeled object.
            centroids (numpy.ndarray): Array containing centroids (x, y) of each labeled object.
        """
        # Find connected components and count the number of objects
        labeled_image, num_objects = label(binary_mask, connectivity=2, return_num=True)
        # Initialize arrays for storing properties
        centroids = np.zeros((num_objects, 2))

        # Get region properties
        props = regionprops_table(
            labeled_image, properties=("centroid", "area", "coords")
        )
        centroids[:, 0] = props["centroid-1"]
        centroids[:, 1] = props["centroid-0"]
        areas = props["area"]
        pixel_index_list = props["coords"]
        return pixel_index_list, areas, centroids

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
        pixel_idx_list, areas, centroids = self.calculate_region_properties(BW)

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

    def compute_spot_and_cell_props(
        self,
        image,
        image_cell,
        k1,
        k2,
        prot_thres=0.05,
        large_prot_thres=100.0,
        areathres=30.0,
        rdl=[50.0, 0.0, 0.0],
        z=0,
        cell_threshold1=200.0,
        cell_threshold2=200,
        cell_sigma1=2.0,
        cell_sigma2=40.0,
        d=2,
        analyse_clr=True,
    ):
        """
        Gets basic image properties (centroids, radiality)
        from a single image and compare to a cell mask from another image channel

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
            analyse_clr (boolean): analyse clr yes/no

        Returns:
            to_save (pl.DataFrame): spot properties ready to save
            to_save_largeobjects (pl.DataFrame): large object properties ready to save
            to_save_cell (pl.DataFrame): cell properties ready to save
            cell_mask (np.ndarray): cell mask
        """

        columns = [
            "x",
            "y",
            "z",
            "sum_intensity_in_photons",
            "bg_per_punctum",
            "bg_per_pixel",
            "incell",
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
            "incell",
            "zi",
            "zf",
        ]
        if analyse_clr == True:
            columns_cell = [
                "colocalisation_likelihood_ratio",
                "std",
                "CSR",
                "expected_spots",
                "coincidence",
                "chance_coincidence",
                "coincidence_largeobj",
                "chance_coincidence_largeobj",
                "n_iterations",
                "z",
            ]
        else:
            columns_cell = [
                "coincidence",
                "chance_coincidence",
                "coincidence_largeobj",
                "chance_coincidence_largeobj",
                "n_iterations",
                "z",
            ]

        if not isinstance(z, int):
            z_planes = np.arange(z[0], z[1])
            cell_mask = np.zeros_like(image_cell, dtype=bool)
            (
                clr,
                norm_std,
                norm_CSR,
                expected_spots,
                coincidence,
                chance_coincidence,
                raw_colocalisation,
                n_iter,
                coincidence_large,
                chance_coincidence_large,
                raw_coloc_large,
            ) = self.gen_CSRmats(image_cell.shape[2])

            centroids = {}  # do this so can parallelise
            raw_colocalisation = {}
            estimated_intensity = {}
            estimated_background = {}
            estimated_background_perpixel = {}
            areas_large = {}
            centroids_large = {}
            meanintensities_large = {}
            sumintensities_large = {}
            pil_large = {}

            def analyse_zplanes(zp):
                img_z = image[:, :, zp]
                img_cell_z = image_cell[:, :, zp]
                (
                    centroids[zp],
                    estimated_intensity[zp],
                    estimated_background[zp],
                    estimated_background_perpixel[zp],
                    areas_large[zp],
                    centroids_large[zp],
                    meanintensities_large[zp],
                    sumintensities_large[zp],
                    pil_large[zp],
                ) = self.default_spotanalysis_routine(
                    img_z, k1, k2, prot_thres, large_prot_thres, areathres, rdl, d
                )

                cell_mask[:, :, zp] = self.detect_large_features(
                    img_cell_z,
                    cell_threshold1,
                    cell_threshold2,
                    cell_sigma1,
                    cell_sigma2,
                )

                image_size = img_z.shape
                mask_indices, spot_indices = self.generate_mask_and_spot_indices(
                    cell_mask[:, :, zp],
                    np.asarray(centroids[zp], dtype=int),
                    image_size,
                )

                if analyse_clr == True:
                    (
                        clr[zp],
                        norm_std[zp],
                        norm_CSR[zp],
                        expected_spots[zp],
                        coincidence[zp],
                        chance_coincidence[zp],
                        raw_colocalisation[zp],
                        n_iter[zp],
                    ) = self.calculate_spot_colocalisation_likelihood_ratio(
                        spot_indices, mask_indices, image_size
                    )
                else:
                    (
                        coincidence[zp],
                        chance_coincidence[zp],
                        raw_colocalisation[zp],
                        n_iter[zp],
                    ) = self.calculate_spot_to_mask_coincidence(
                        spot_indices, mask_indices, image_size
                    )

                large_aggregate_indices = self.generate_largeaggregate_indices(
                    pil_large[zp], image_size
                )
                (
                    coincidence_large[zp],
                    chance_coincidence_large[zp],
                    raw_coloc_large[zp],
                ) = self.calculate_largeobj_coincidence(
                    large_aggregate_indices,
                    mask_indices,
                    len(pil_large[zp]),
                    image_size,
                )

            pool = Pool(nodes=cpu_number)
            pool.restart()
            pool.map(analyse_zplanes, z_planes)
            pool.close()
            pool.terminate()

            to_save = self.make_datarray_spot(
                centroids,
                estimated_intensity,
                estimated_background,
                estimated_background_perpixel,
                raw_colocalisation,
                columns,
                z_planes,
            )

            to_save_largeobjects = self.make_datarray_largeobjects(
                areas_large,
                centroids_large,
                sumintensities_large,
                meanintensities_large,
                raw_coloc_large,
                columns_large,
                z_planes,
            )

            to_save_cell = self.make_datarray_cell(
                clr,
                norm_std,
                norm_CSR,
                expected_spots,
                coincidence,
                chance_coincidence,
                coincidence_large,
                chance_coincidence_large,
                n_iter,
                columns_cell,
                z_planes,
                analyse_clr,
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
                pil_large,
            ) = self.default_spotanalysis_routine(
                image, k1, k2, prot_thres, large_prot_thres, areathres, rdl, d
            )

            cell_mask = self.detect_large_features(
                image_cell, cell_threshold1, cell_threshold2, cell_sigma1, cell_sigma2
            )

            image_size = image.shape
            mask_indices, spot_indices = self.generate_mask_and_spot_indices(
                cell_mask, centroids, image_size
            )

            if analyse_clr == True:
                (
                    clr,
                    norm_std,
                    norm_CSR,
                    expected_spots,
                    coincidence,
                    chance_coincidence,
                    raw_localisation,
                    n_iter,
                ) = self.calculate_spot_colocalisation_likelihood_ratio(
                    spot_indices, mask_indices, image_size
                )
            else:
                coincidence, chance_coincidence, raw_localisation, n_iter = (
                    self.calculate_spot_to_mask_coincidence(
                        spot_indices, mask_indices, image_size
                    )
                )

            large_aggregate_indices = self.generate_largeaggregate_indices(
                pil_large, image_size
            )
            coincidence_large, chance_coincidence_large, raw_coloc_large = (
                self.calculate_largeobj_coincidence(
                    large_aggregate_indices, mask_indices, len(pil_large), image_size
                )
            )

            to_save = self.make_datarray_spot(
                centroids,
                estimated_intensity,
                estimated_background,
                estimated_background_perpixel,
                raw_localisation,
                columns[:-2],
            )

            to_save_largeobjects = self.make_datarray_largeobjects(
                areas_large,
                centroids_large,
                sumintensities_large,
                meanintensities_large,
                raw_coloc_large,
                columns_large[:-2],
            )

            to_save_cell = self.make_datarray_cell(
                clr,
                norm_std,
                norm_CSR,
                expected_spots,
                coincidence,
                chance_coincidence,
                coincidence_large,
                chance_coincidence_large,
                n_iter,
                columns_cell[:-1],
                analyse_clr,
            )
        return to_save, to_save_largeobjects, to_save_cell, cell_mask

    def compute_spot_props(
        self,
        image,
        k1,
        k2,
        thres=0.05,
        large_thres=100.0,
        areathres=30.0,
        rdl=[50.0, 0.0, 0.0],
        z=0,
        d=2,
    ):
        """
        Gets basic image properties (centroids, radiality)
        from a single image

        Args:
            image (np.ndarray): image as numpy array, or dict of arrays
            k1 (array): gaussian blur kernel
            k2 (array): ricker wavelet kernel
            thres (float): percentage threshold
            areathres (float): area threshold
            rdl (array): radiality thresholds
            z (array): z planes to image, default 0
            d (int): Pixel radius value

        Returns:
            to_save (pl.DataFrame): data array to save as pandas object,
                                            or dict of pandas objects
            to_save_largeobjects (pl.DataFrame): data array of large objects
                                                        or dict of large objects
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
            centroids = {}
            estimated_intensity = {}
            estimated_background = {}
            estimated_background_perpixel = {}
            areas_large = {}
            centroids_large = {}
            meanintensities_large = {}
            sumintensities_large = {}
            pil_large = {}

            def analyse_zplanes(zp):
                (
                    centroids[zp],
                    estimated_intensity[zp],
                    estimated_background[zp],
                    estimated_background_perpixel[zp],
                    areas_large[zp],
                    centroids_large[zp],
                    meanintensities_large[zp],
                    sumintensities_large[zp],
                    pil_large[zp],
                ) = self.default_spotanalysis_routine(
                    image[:, :, zp], k1, k2, thres, large_thres, areathres, rdl, d
                )

            pool = Pool(nodes=cpu_number)
            pool.restart()
            pool.map(analyse_zplanes, z_planes)
            pool.close()
            pool.terminate()

            to_save = self.make_datarray_spot(
                centroids,
                estimated_intensity,
                estimated_background,
                estimated_background_perpixel,
                0,
                columns,
                z_planes,
                cell_analysis=False,
            )

            to_save_largeobjects = self.make_datarray_largeobjects(
                areas_large,
                centroids_large,
                sumintensities_large,
                meanintensities_large,
                0,
                columns_large,
                z_planes,
                cell_analysis=False,
            )

        else:
            centroids, estimated_intensity, estimated_background,
            estimated_background_perpixel, areas_large, centroids_large,
            meanintensities_large, sumintensities_large, pil_large = (
                self.default_spotanalysis_routine(
                    image, k1, k2, thres, large_thres, areathres, rdl, d
                )
            )

            to_save = self.make_datarray_spot(
                centroids,
                estimated_intensity,
                estimated_background,
                estimated_background_perpixel,
                0,
                columns[:-2],
            )

            to_save_largeobjects = self.make_datarray_largeobjects(
                areas_large,
                centroids_large,
                sumintensities_large,
                meanintensities_large,
                0,
                columns_large[:-2],
                cell_analysis=False,
            )

        return to_save, to_save_largeobjects

    @staticmethod
    @jit(nopython=True)
    def bincalculator(data):
        """bincalculator function
        reads in data and generates bins according to Freedman-Diaconis rule

        Args:
            data (np.1darray): data to calculate bins

        Returns:
            bins (np.1darray): bins for histogram according to Freedman-Diaconis rule"""
        N = len(data)
        sigma = np.std(data)

        binwidth = np.multiply(np.multiply(np.power(N, np.divide(-1, 3)), sigma), 3.5)
        bins = np.linspace(
            np.min(data),
            np.max(data),
            int((np.max(data) - np.min(data)) / binwidth) + 1,
        )
        return bins

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

        def gaussian(height, center_x, center_y, width_x, width_y, bg):
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
                (gaussian(*p)(*np.indices(data.shape)) - data)
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

    def rejectoutliers(self, data):
        """rejectoutliers function
        # rejects outliers from data, does iqr method (i.e. anything below
        lower quartile (25 percent) or above upper quartile (75 percent)
        is rejected)

        Args:
            data (np.1darray): data matrix

        Returns:
            newdata (np.1darray): data matrix"""
        from scipy.stats import iqr

        IQR = iqr(data)
        q1, q2 = np.percentile(data, q=(25, 75))

        nd1 = data[data <= (1.5 * IQR) + q2]
        newdata = nd1[nd1 >= q1 - (1.5 * IQR)]
        return newdata

    def rejectoutliers_ind(self, data):
        """rejectoutliers function
        # gets indices to reject outliers from data, does iqr method (i.e. anything
        below lower quartile (25 percent) or above upper quartile (75 percent)
        is rejected)

        Args:
            data (np.1darray): data matrix

        Returns:
            ind_arr (np.1darray): index array"""
        from scipy.stats import iqr

        IQR = iqr(data)
        q1, q2 = np.percentile(data, q=(25, 75))

        indices = np.arange(len(data), dtype=int)
        nd1 = indices[data >= (1.5 * IQR) + q2]
        nd2 = indices[data <= q1 - (1.5 * IQR)]
        return np.hstack([nd1, nd2])

    def compute_image_props(
        self,
        image,
        k1,
        k2,
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
            img2, Gx, Gy, focusScore, cfactor = self.calculate_gradient_field(image, k1)
            dl_mask, centroids, radiality, idxs = self.small_feature_kernel(
                image, large_mask, img2, Gx, Gy, k2, thres, areathres, rdl, d
            )

        else:
            radiality = {}
            centroids = {}
            large_mask = np.zeros_like(image)
            dl_mask = np.zeros_like(image)
            img2 = np.zeros_like(image)
            Gx = np.zeros_like(image)
            Gy = np.zeros_like(image)

            def run_over_z(z):
                if calib == True:
                    large_mask[:, :, z] = np.full_like(image[:, :, z], False)
                else:
                    large_mask[:, :, z] = self.detect_large_features(
                        image[:, :, z], large_thres
                    )
                img2[:, :, z], Gx[:, :, z], Gy[:, :, z], focusScore, cfactor = (
                    self.calculate_gradient_field(image[:, :, z], k1)
                )
                dl_mask[:, :, z], centroids[z], radiality[z], idxs = (
                    self.small_feature_kernel(
                        image[:, :, z],
                        large_mask[:, :, z],
                        img2[:, :, z],
                        Gx[:, :, z],
                        Gy[:, :, z],
                        k2,
                        thres,
                        areathres,
                        rdl,
                        d,
                    )
                )

            pool = Pool(nodes=cpu_number)
            pool.restart()
            pool.map(run_over_z, z_planes)
            pool.close()
            pool.terminate()
        return dl_mask, centroids, radiality, large_mask

    def spot_to_spot_rdf(self, coordinates, pixel_size=0.11, dr=1.0):
        """
        Generates spot_to_spot_rdf

        Uses code from 10.5281/zenodo.4625675, please cite this software if used in a paper.

        Args:
            coordinates (np.2darray): array of 2d (or 3D) coordinates
            pixel_size (np.float): pixel size in same units as dr (default: 0.11)
            dr (np.float): step for radial distribution function

        Returns:
            g_r (np.1darray): radial distribution function
            radii (np.1darray): radius vector

        """
        g_r, radii = rdf(np.multiply(coordinates, pixel_size), dr=dr)
        return g_r, radii

    def spot_to_mask_rdf(
        self,
        coordinates_spot,
        coordinates_mask,
        pixel_size=0.11,
        dr=0.1,
        r_max=30.0,
        image_size=(1200, 1200),
    ):
        """
        Generates spot_to_mask_rdf

        Args:
            coordinates_spot (np.2darray): array of 2D spot coordinates
            coordinates_mask (np.2darray): array of 2D mask coordinates
            pixel_size (np.float): pixel size in same units as dr (default: 0.11)
            dr (np.float): step for radial distribution function
            max_radius (np.float): maximum radius value
            image_size (array): maximum image size

        Returns:
            g_r (np.1darray): radial distribution function
            radii (np.1darray): radius vector
        """
        g_r, radii = MultiD_RD_functions.multid_rdf(
            coordinates_spot * pixel_size,
            coordinates_mask * pixel_size,
            r_max,
            dr,
            boxdims=(
                [[0.0, 0.0], [image_size[0] * pixel_size, image_size[1] * pixel_size]]
            ),
            parallel=True,
        )
        return g_r, radii

    def gen_CSRmats(self, image_z_shape):
        """
        Generates empty matrices for the CSR

        Args:
            image_z_shape (int): shape of new array

        Returns:
            clr (ndarray): empty array
            norm_std (ndarray): empty array
            norm_CSR (ndarray): empty array
            expected_spots (ndarray): empty array
            coincidence (ndarray): empty array
            chance_coincidence (ndarray): empty array
            raw_localisation (dict): empty dict
            n_iter (ndarray): empty array

        """

        clr = np.zeros(image_z_shape)
        norm_std = np.zeros(image_z_shape)
        norm_CSR = np.zeros(image_z_shape)
        expected_spots = np.zeros(image_z_shape)
        coincidence = np.zeros(image_z_shape)
        chance_coincidence = np.zeros(image_z_shape)
        raw_colocalisation = {}
        n_iter = np.zeros(image_z_shape)
        coincidence_large = np.zeros(image_z_shape)
        chance_coincidence_large = np.zeros(image_z_shape)
        raw_colocalisation_large = {}
        return (
            clr,
            norm_std,
            norm_CSR,
            expected_spots,
            coincidence,
            chance_coincidence,
            raw_colocalisation,
            n_iter,
            coincidence_large,
            chance_coincidence_large,
            raw_colocalisation_large,
        )

    def make_datarray_largeobjects(
        self,
        areas_large,
        centroids_large,
        sumintensities_large,
        meanintensities_large,
        raw_coloc,
        columns,
        z_planes=0,
        cell_analysis=True,
    ):
        """
        makes a datarray in pandas for large object information

        Args:
            areas_large (np.1darray): areas in pixels
            centroids_large (np.1darray): centroids of large objects
            meanintensities_large (np.1darray): mean intensities of large objects
            raw_coloc (np.1darray): if large spot is in cell or not
            columns (list of strings): column labels
            z_planes: z_planes to put in array (if needed); if int, assumes only
                one z-plane

        Returns:
            to_save_largeobjects (pandas DataArray) pandas array to save
            columns_large = ['x', 'y', 'z', 'area', 'mean_intensity_in_photons', 'zi', 'zf']

        """
        if isinstance(z_planes, int):
            if cell_analysis == True:
                dataarray = np.vstack(
                    [
                        centroids_large[:, 0],
                        centroids_large[:, 1],
                        np.full_like(centroids_large[:, 0], 1),
                        areas_large,
                        sumintensities_large,
                        meanintensities_large,
                        raw_coloc,
                    ]
                )
            else:
                dataarray = np.vstack(
                    [
                        centroids_large[:, 0],
                        centroids_large[:, 1],
                        np.full_like(centroids_large[:, 0], 1),
                        areas_large,
                        sumintensities_large,
                        meanintensities_large,
                    ]
                )
        else:
            for z in z_planes:
                if cell_analysis == True:
                    stack = np.vstack(
                        [
                            centroids_large[z][:, 0],
                            centroids_large[z][:, 1],
                            np.full_like(centroids_large[z][:, 0], z + 1),
                            areas_large[z],
                            sumintensities_large[z],
                            meanintensities_large[z],
                            raw_coloc[z],
                            np.full_like(centroids_large[z][:, 0], 1 + z_planes[0]),
                            np.full_like(centroids_large[z][:, 0], 1 + z_planes[-1]),
                        ]
                    )
                else:
                    stack = np.vstack(
                        [
                            centroids_large[z][:, 0],
                            centroids_large[z][:, 1],
                            np.full_like(centroids_large[z][:, 0], z + 1),
                            areas_large[z],
                            sumintensities_large[z],
                            meanintensities_large[z],
                            np.full_like(centroids_large[z][:, 0], 1 + z_planes[0]),
                            np.full_like(centroids_large[z][:, 0], 1 + z_planes[-1]),
                        ]
                    )
                if z == z_planes[0]:
                    dataarray = stack
                else:
                    da = stack
                    dataarray = np.hstack([dataarray, da])
        return pl.DataFrame(data=dataarray.T, schema=columns)

    def make_datarray_spot(
        self,
        centroids,
        estimated_intensity,
        estimated_background,
        estimated_background_perpixel,
        raw_colocalisation,
        columns,
        z_planes=0,
        cell_analysis=True,
    ):
        """
        makes a datarray in pandas for spot information

        Args:
            centroids (ndarray): centroid positions
            estimated_intensity (ndarray): estimated intensities
            estimated_background (ndarray): estimated backgrounds per punctum
            estimated_background_perpixel (np.ndarray): estimated background per pixel
            raw_colocalisation (np.ndarray): if spot is in cell mask or not
            columns (list of strings): column labels
            z_planes: z_planes to put in array (if needed); if int, assumes only
                one z-plane

        Returns:
            to_save (pl.DataFrame) pandas array to save

        """
        if isinstance(z_planes, int):
            if cell_analysis == True:
                dataarray = np.vstack(
                    [
                        centroids[:, 0],
                        centroids[:, 1],
                        np.full_like(centroids[:, 0], 1),
                        estimated_intensity,
                        estimated_background,
                        estimated_background_perpixel,
                        raw_colocalisation,
                    ]
                )
            else:
                dataarray = np.vstack(
                    [
                        centroids[:, 0],
                        centroids[:, 1],
                        np.full_like(centroids[:, 0], 1),
                        estimated_intensity,
                        estimated_background,
                        estimated_background_perpixel,
                    ]
                )
        else:
            for z in z_planes:
                if cell_analysis == True:
                    stack = np.vstack(
                        [
                            centroids[z][:, 0],
                            centroids[z][:, 1],
                            np.full_like(centroids[z][:, 0], z + 1),
                            estimated_intensity[z],
                            estimated_background[z],
                            estimated_background_perpixel[z],
                            raw_colocalisation[z],
                            np.full_like(centroids[z][:, 0], 1 + z_planes[0]),
                            np.full_like(centroids[z][:, 0], 1 + z_planes[-1]),
                        ]
                    )
                else:
                    stack = np.vstack(
                        [
                            centroids[z][:, 0],
                            centroids[z][:, 1],
                            np.full_like(centroids[z][:, 0], z + 1),
                            estimated_intensity[z],
                            estimated_background[z],
                            estimated_background_perpixel[z],
                            np.full_like(centroids[z][:, 0], 1 + z_planes[0]),
                            np.full_like(centroids[z][:, 0], 1 + z_planes[-1]),
                        ]
                    )
                if z == z_planes[0]:
                    dataarray = stack
                else:
                    da = stack
                    dataarray = np.hstack([dataarray, da])
        return pl.DataFrame(data=dataarray.T, schema=columns)

    def colocalise_with_threshold(
        self,
        analysis_file,
        threshold,
        protein_string,
        cell_string,
        imtype=".tif",
        blur_degree=1,
        calc_clr=False,
        aboveT=1,
    ):
        """
        Does colocalisation analysis of spots vs mask with an additional threshold.

        Args:
            analysis_file (string): The analysis file of puncta.
            threshold (float): The photon threshold for puncta.
            protein_string (str): string of protein images
            cell_string (str): string of cell images
            imtype (str): image end string
            blur_degree (int): degree of blur to apply to puncta
            calc_clr (boolean): calculate clr yes/no
            aboveT (int): do the calculation above or below threshold

        Returns:
            cell_analysis (pl.DataFrame): polars dataframe of the cell analysis
            spot_analysis (pl.DataFrame): polars dataframe of the spot analysis
        """

        spots_with_intensities = pl.read_csv(analysis_file)
        if aboveT == 1:
            spots_with_intensities = spots_with_intensities.filter(
                pl.col("sum_intensity_in_photons") > threshold
            )
        else:
            spots_with_intensities = spots_with_intensities.filter(
                pl.col("sum_intensity_in_photons") <= threshold
            )

        if len(spots_with_intensities) > 0:
            analysis_directory = os.path.split(analysis_file)[0]
            image_filenames = np.unique(
                spots_with_intensities["image_filename"].to_numpy()
            )

            if calc_clr == False:
                columns = [
                    "coincidence",
                    "chance_coincidence",
                    "n_iter",
                    "image_filename",
                ]
            else:
                columns = [
                    "clr",
                    "norm_std",
                    "norm_CSR",
                    "expected_spots",
                    "coincidence",
                    "chance_coincidence",
                    "n_iter",
                    "image_filename",
                ]

            for i, image in enumerate(image_filenames):
                cell_mask = IO.read_tiff(
                    os.path.join(
                        analysis_directory,
                        os.path.split(image.split(imtype)[0])[-1].split(protein_string)[
                            0
                        ]
                        + str(cell_string)
                        + "_cellMask.tiff",
                    )
                )
                image_size = cell_mask.shape[:-1]
                image_file = spots_with_intensities.filter(
                    pl.col("image_filename") == image
                )
                z_planes = np.unique(image_file["z"].to_numpy())

                dataarray = np.zeros([len(z_planes), len(columns)])

                temp_pl = pl.DataFrame(data=dataarray, schema=columns)

                for j, z_plane in enumerate(z_planes):
                    filtered_file = image_file.filter(pl.col("z") == z_plane)
                    xcoords = filtered_file["x"].to_numpy()
                    ycoords = filtered_file["y"].to_numpy()
                    mask = cell_mask[:, :, int(z_plane) - 1]
                    centroids = np.asarray(np.vstack([xcoords, ycoords]), dtype=int).T
                    mask_indices, spot_indices = self.generate_mask_and_spot_indices(
                        mask, centroids, image_size
                    )
                    if calc_clr == False:
                        (
                            temp_pl[j, 0],
                            temp_pl[j, 1],
                            raw_colocalisation,
                            temp_pl[j, 2],
                        ) = self.calculate_spot_to_mask_coincidence(
                            spot_indices,
                            mask_indices,
                            image_size,
                            blur_degree=blur_degree,
                        )
                    else:
                        (
                            temp_pl[j, 0],
                            temp_pl[j, 1],
                            temp_pl[j, 2],
                            temp_pl[j, 3],
                            temp_pl[j, 4],
                            temp_pl[j, 5],
                            raw_colocalisation,
                            temp_pl[j, 6],
                        ) = self.calculate_spot_colocalisation_likelihood_ratio(
                            spot_indices, mask_indices, image_size, blur_degree=1
                        )
                    if j == 0:
                        rc = raw_colocalisation
                    else:
                        rc = np.hstack([rc, raw_colocalisation])
                temp_pl = temp_pl.with_columns(
                    image_filename=np.full_like(z_planes, image, dtype="object")
                )
                image_file = image_file.with_columns(incell=rc * 1)
                if i == 0:
                    cell_analysis = temp_pl
                    spot_analysis = image_file
                else:
                    cell_analysis = pl.concat([cell_analysis, temp_pl])
                    spot_analysis = pl.concat([spot_analysis, image_file])

            return cell_analysis, spot_analysis
        else:
            return np.NAN, np.NAN

    def two_puncta_channels_rdf_with_thresholds(
        self,
        analysis_data_p1,
        analysis_data_p2,
        threshold_p1,
        threshold_p2,
        pixel_size=0.11,
        dr=1.0,
        protein_string_1="C0",
        protein_string_2="C1",
        imtype=".tif",
        aboveT=1,
        image_size=(1200.0, 1200.0),
    ):
        """
        Does rdf analysis of spots wrt mask from an analysis file.

        Args:
            analysis_data_p1 (pl.DataFrame): The analysis data of puncta set 1.
            analysis_data_p2 (pl.DataFrame): The analysis data of puncta set 2.
            threshold_p1 (float): The photon threshold for puncta set 1.
            threshold_p2 (float): The photon threshold for puncta set 2.
            pixel_size (float): size of pixels
            dr (float): dr of rdf
            protein_string_1 (string): will use this to find corresponding other punctum files
            protein_string_2 (string): will use this to find corresponding other punctum files
            imtype (string): image type previously analysed
            aboveT (int): do the calculation above or below threshold

        Returns:
            rdf (pl.DataArray): polars datarray of the rdf
        """

        start = time.time()

        if aboveT == 1:
            analysis_data_p1 = analysis_data_p1.filter(
                pl.col("sum_intensity_in_photons") > threshold_p1
            )
            analysis_data_p2 = analysis_data_p2.filter(
                pl.col("sum_intensity_in_photons") > threshold_p2
            )
        else:
            analysis_data_p1 = analysis_data_p1.filter(
                pl.col("sum_intensity_in_photons") <= threshold_p1
            )
            analysis_data_p2 = analysis_data_p2.filter(
                pl.col("sum_intensity_in_photons") <= threshold_p2
            )

        if (len(analysis_data_p1) > 0) and (len(analysis_data_p2) > 0):

            files_p1 = np.unique(
                [
                    file.split(imtype)[0].split(protein_string_1)[0]
                    for file in analysis_data_p1["image_filename"].to_numpy()
                ]
            )
            files_p2 = np.unique(
                [
                    file.split(imtype)[0].split(protein_string_2)[0]
                    for file in analysis_data_p2["image_filename"].to_numpy()
                ]
            )
            files = np.unique(np.hstack([files_p1, files_p2]))

            z_planes = {}  # make dict where z planes will be stored
            for i, file in enumerate(files):
                zp_f1 = np.unique(
                    analysis_data_p1.filter(
                        pl.col("image_filename")
                        == file + str(protein_string_1) + imtype
                    )["z"].to_numpy()
                )
                zp_f2 = np.unique(
                    analysis_data_p2.filter(
                        pl.col("image_filename")
                        == file + str(protein_string_2) + imtype
                    )["z"].to_numpy()
                )

                z_planes[file] = np.unique(np.hstack([zp_f1, zp_f2]))

            g_r = {}

            for i, file in enumerate(files):
                zs = z_planes[file]
                subset_p1 = analysis_data_p1.filter(
                    pl.col("image_filename") == file + str(protein_string_1) + imtype
                )
                subset_p2 = analysis_data_p2.filter(
                    pl.col("image_filename") == file + str(protein_string_2) + imtype
                )
                for z in zs:
                    uid = str(file) + "___" + str(z)
                    filtered_subset_p1 = subset_p1.filter(pl.col("z") == z)
                    filtered_subset_p2 = subset_p2.filter(pl.col("z") == z)

                    x_p1 = filtered_subset_p1["x"].to_numpy()
                    y_p1 = filtered_subset_p1["y"].to_numpy()
                    x_p2 = filtered_subset_p2["x"].to_numpy()
                    y_p2 = filtered_subset_p2["y"].to_numpy()
                    coordinates_p1_spot = np.asarray(
                        np.vstack([x_p1, y_p1]).T, dtype=int
                    )
                    coordinates_p2_spot = np.asarray(
                        np.vstack([x_p2, y_p2]).T, dtype=int
                    )
                    if (len(coordinates_p1_spot) > 0) and (
                        len(coordinates_p2_spot) > 0
                    ):
                        g_r[uid], radii = self.spot_to_mask_rdf(
                            coordinates_p1_spot,
                            coordinates_p2_spot,
                            pixel_size=pixel_size,
                            dr=dr,
                            r_max=np.divide(
                                np.multiply(image_size[0], pixel_size), 4.0
                            ),
                            image_size=image_size,
                        )
                print(
                    "Computing RDF     File {}/{}    Time elapsed: {:.3f} s".format(
                        i + 1, len(files), time.time() - start
                    ),
                    end="\r",
                    flush=True,
                )

            g_r_overall = np.zeros([len(radii), len(g_r.keys())])

            for i, uid in enumerate(g_r.keys()):
                g_r_overall[:, i] = g_r[uid]

            g_r_mean = np.mean(g_r_overall, axis=1)
            g_r_std = np.std(g_r_overall, axis=1)
            data = {"radii": radii, "g_r_mean": g_r_mean, "g_r_std": g_r_std}

            rdf = pl.DataFrame(data)
            return rdf
        else:
            return np.NAN

    def single_spot_channel_rdf_with_threshold(
        self, analysis_file, threshold, pixel_size=0.11, dr=1.0, aboveT=1
    ):
        """
        Does rdf analysis of spots from an analysis file.

        Args:
            analysis_data (pl.DataFrame): The analysis data of puncta.
            threshold (float): The photon threshold for puncta.
            pixel_size (float): size of pixels
            dr (float): dr of rdf
            aboveT (int): do the calculation above or below threshold

        Returns:
            rdf (pl.DataArray): polars datarray of the rdf
        """

        analysis_data = pl.read_csv(analysis_file)
        if aboveT == 1:
            analysis_data = analysis_data.filter(
                pl.col("sum_intensity_in_photons") > threshold
            )
            typestr = "> threshold"
        else:
            analysis_data = analysis_data.filter(
                pl.col("sum_intensity_in_photons") <= threshold
            )
            typestr = "<= threshold"

        if len(analysis_data) > 0:

            files = np.unique(analysis_data["image_filename"].to_numpy())
            z_planes = {}  # make dict where z planes will be stored
            for i, file in enumerate(files):
                z_planes[file] = np.unique(
                    analysis_data.filter(pl.col("image_filename") == file)[
                        "z"
                    ].to_numpy()
                )

            g_r = {}
            radii = {}

            start = time.time()

            for i, file in enumerate(files):
                zs = z_planes[file]
                subset = analysis_data.filter(pl.col("image_filename") == file)
                for z in zs:
                    uid = str(file) + "___" + str(z)
                    subset_filter = subset.filter(pl.col("z") == z)
                    x = subset_filter["x"].to_numpy()
                    y = subset_filter["y"].to_numpy()
                    coordinates = np.vstack([x, y]).T
                    g_r[uid], radii[uid] = self.spot_to_spot_rdf(
                        coordinates, pixel_size=pixel_size, dr=dr
                    )
                print(
                    "Computing "
                    + typestr
                    + " RDF     File {}/{}    Time elapsed: {:.3f} s".format(
                        i + 1, len(files), time.time() - start
                    ),
                    end="\r",
                    flush=True,
                )

            radii_key, radii_overall = max(radii.items(), key=lambda x: len(set(x[1])))

            g_r_overall = np.zeros([len(radii_overall), len(g_r.keys())])

            for i, uid in enumerate(g_r.keys()):
                g_r_overall[:, i] = np.interp(
                    fp=g_r[uid], xp=radii[uid], x=radii_overall, left=0.0, right=0.0
                )
            g_r_mean = np.mean(g_r_overall, axis=1)
            g_r_std = np.std(g_r_overall, axis=1)
            data = {"radii": radii_overall, "g_r_mean": g_r_mean, "g_r_std": g_r_std}

            rdf = pl.DataFrame(data)
            return rdf
        else:
            return np.NAN

    def number_of_puncta_per_segmented_cell_with_threshold(
        self,
        analysis_file,
        threshold,
        blur_degree=1,
        cell_string="C0",
        protein_string="C1",
        imtype=".tif",
        aboveT=1,
    ):
        """
        Does analysis of number of oligomers in a mask area per "segmented"
        cell area.

        Args:
            analysis_file (pl.DataFrame): The analysis file location of puncta set 1.
            threshold (float): The photon threshold for puncta set 1.
            out_cell (boolean): exclude puncta that are inside cells
            pixel_size (float): size of pixels
            blur_degree (int): degree to blur spots
            cell_string (string): will use this to find corresponding cell files
            protein_string (string): will use this to find corresponding files
            imtype (string): image type previously analysed
            aboveT (int): do the calculation above or below threshold

        Returns:
            cell_punctum_analysis (pl.DataFrame): polars datarray of the cell analysis
        """

        analysis_data = pl.read_csv(analysis_file)
        if aboveT == 1:
            analysis_data = analysis_data.filter(
                pl.col("sum_intensity_in_photons") > threshold
            )
            typestr = "> threshold"
        else:
            analysis_data = analysis_data.filter(
                pl.col("sum_intensity_in_photons") <= threshold
            )
            typestr = "<= threshold"

        analysis_directory = os.path.split(analysis_file)[0]
        analysis_data = analysis_data.filter(pl.col("incell") == 1)

        start = time.time()

        if len(analysis_data) > 0:
            files = np.unique(analysis_data["image_filename"].to_numpy())
            z_planes = {}  # make dict where z planes will be stored
            for i, file in enumerate(files):
                z_planes[file] = np.unique(
                    analysis_data.filter(pl.col("image_filename") == file)[
                        "z"
                    ].to_numpy()
                )

            for i, file in enumerate(files):
                cell_mask = IO.read_tiff(
                    os.path.join(
                        analysis_directory,
                        os.path.split(file.split(imtype)[0])[-1].split(protein_string)[
                            0
                        ]
                        + str(cell_string)
                        + "_cellMask.tiff",
                    )
                )
                zs = z_planes[file]
                subset = analysis_data.filter(pl.col("image_filename") == file)
                image_size = cell_mask[:, :, 0].shape
                for j, z in enumerate(zs):
                    pil_mask, areas, centroids = self.calculate_region_properties(
                        cell_mask[:, :, int(z) - 1]
                    )
                    x_m = centroids[:, 0]
                    y_m = centroids[:, 1]
                    filtered_subset = subset.filter(pl.col("z") == z)
                    x = filtered_subset["x"].to_numpy()
                    y = filtered_subset["y"].to_numpy()
                    centroids_puncta = np.asarray(np.vstack([x, y]), dtype=int)
                    spot_indices = np.ravel_multi_index(
                        centroids_puncta, image_size, order="F"
                    )
                    filename_tosave = np.full_like(x_m, file, dtype="object")
                    n_spots_in_object = np.zeros_like(x_m)
                    z_tosave = np.full_like(x_m, z)

                    for k in np.arange(len(areas)):
                        coords = pil_mask[k]
                        xm = coords[:, 0]
                        ym = coords[:, 1]
                        if (np.any(xm > image_size[0])) or (np.any(ym > image_size[1])):
                            n_spots_in_object[k] = 0
                        else:
                            coordinates_mask = np.asarray(
                                np.vstack([xm, ym]), dtype=int
                            )
                            mask_indices = np.ravel_multi_index(
                                coordinates_mask, image_size, order="F"
                            )
                            na, na, raw_colocalisation, na = (
                                self.calculate_spot_to_mask_coincidence(
                                    spot_indices,
                                    mask_indices,
                                    image_size,
                                    n_iter=1,
                                    blur_degree=1,
                                )
                            )
                            n_spots_in_object[k] = np.sum(raw_colocalisation)

                    data = {
                        "area/pixels": areas,
                        "x_centre": x_m,
                        "y_centre": y_m,
                        "z": z_tosave,
                        "n_puncta_in_cell": n_spots_in_object,
                        "image_filename": filename_tosave,
                    }
                    if (i == 0) and (j == 0):
                        cell_punctum_analysis = pl.DataFrame(data)
                    else:
                        cell_punctum_analysis = pl.concat(
                            [cell_punctum_analysis, pl.DataFrame(data)], rechunk=True
                        )

                print(
                    "Computing "
                    + typestr
                    + " spots in cells     File {}/{}    Time elapsed: {:.3f} s".format(
                        i + 1, len(files), time.time() - start
                    ),
                    end="\r",
                    flush=True,
                )

            return cell_punctum_analysis
        else:
            return np.NAN

    def spot_mask_rdf_with_threshold(
        self,
        analysis_file,
        threshold,
        out_cell=True,
        pixel_size=0.11,
        dr=1.0,
        cell_string="C0",
        protein_string="C1",
        imtype=".tif",
        aboveT=1,
    ):
        """
        Does rdf analysis of spots wrt mask from an analysis file.

        Args:
            analysis_file (pl.DataFrame): The analysis file location of puncta set 1.
            threshold (float): The photon threshold for puncta set 1.
            out_cell (boolean): exclude puncta that are inside cells
            pixel_size (float): size of pixels
            dr (float): dr of rdf
            cell_string (string): will use this to find corresponding cell files
            protein_string (string): will use this to find corresponding files
            imtype (string): image type previously analysed
            aboveT (int): do the calculation above or below threshold

        Returns:
            rdf (pl.DataArray): polars datarray of the rdf
        """

        analysis_data = pl.read_csv(analysis_file)
        if aboveT == 1:
            analysis_data = analysis_data.filter(
                pl.col("sum_intensity_in_photons") > threshold
            )
        else:
            analysis_data = analysis_data.filter(
                pl.col("sum_intensity_in_photons") <= threshold
            )

        analysis_directory = os.path.split(analysis_file)[0]
        if out_cell == True:
            analysis_data = analysis_data.filter(pl.col("incell") == 0)

        start = time.time()

        if len(analysis_data) > 0:
            files = np.unique(analysis_data["image_filename"].to_numpy())
            z_planes = {}  # make dict where z planes will be stored
            for i, file in enumerate(files):
                z_planes[file] = np.unique(
                    analysis_data.filter(pl.col("image_filename") == file)[
                        "z"
                    ].to_numpy()
                )

            g_r = {}

            for i, file in enumerate(files):
                cell_mask = IO.read_tiff(
                    os.path.join(
                        analysis_directory,
                        os.path.split(file.split(imtype)[0])[-1].split(protein_string)[
                            0
                        ]
                        + str(cell_string)
                        + "_cellMask.tiff",
                    )
                )
                zs = z_planes[file]
                subset = analysis_data.filter(pl.col("image_filename") == file)
                image_size = cell_mask[:, :, 0].shape
                for z in zs:
                    uid = str(file) + "___" + str(z)

                    filtered_subset = subset.filter(pl.col("z") == z)
                    x = filtered_subset["x"].to_numpy()
                    y = filtered_subset["y"].to_numpy()
                    coordinates_spot = np.vstack([x, y]).T
                    xm, ym = np.where(cell_mask[:, :, int(z) - 1])
                    coordinates_mask = np.vstack([xm, ym]).T

                    if len(coordinates_mask) > 0:
                        g_r[uid], radii = self.spot_to_mask_rdf(
                            coordinates_spot,
                            coordinates_mask,
                            pixel_size=pixel_size,
                            dr=dr,
                            r_max=np.divide(
                                np.multiply(np.max(image_size), pixel_size), 4.0
                            ),
                            image_size=image_size,
                        )
                print(
                    "Computing RDF     File {}/{}    Time elapsed: {:.3f} s".format(
                        i + 1, len(files), time.time() - start
                    ),
                    end="\r",
                    flush=True,
                )

            g_r_overall = np.zeros([len(radii), len(g_r.keys())])

            for i, uid in enumerate(g_r.keys()):
                g_r_overall[:, i] = g_r[uid]

            g_r_mean = np.mean(g_r_overall, axis=1)
            g_r_std = np.std(g_r_overall, axis=1)

            data = {"radii": radii, "g_r_mean": g_r_mean, "g_r_std": g_r_std}
            rdf = pl.DataFrame(data)

            return rdf
        else:
            return np.NAN

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
        # todo: make faster, add z
        if aboveT == 1:
            spots_1_with_intensities = spots_1_with_intensities.filter(
                pl.col("sum_intensity_in_photons") > threshold_1
            )

            spots_2_with_intensities = spots_2_with_intensities.filter(
                pl.col("sum_intensity_in_photons") > threshold_2
            )
            typestr = "> threshold"
        else:
            spots_1_with_intensities = spots_1_with_intensities.filter(
                pl.col("sum_intensity_in_photons") <= threshold_1
            )

            spots_2_with_intensities = spots_2_with_intensities.filter(
                pl.col("sum_intensity_in_photons") <= threshold_2
            )
            typestr = "<= threshold"

        image_1_filenames = np.unique(
            spots_1_with_intensities["image_filename"].to_numpy()
        )
        image_2_filenames = np.unique(
            spots_2_with_intensities["image_filename"].to_numpy()
        )

        overall_1_filenames = [
            i.split(imtype)[0].split(spot_1_string)[0] for i in image_1_filenames
        ]
        overall_2_filenames = [
            i.split(imtype)[0].split(spot_2_string)[0] for i in image_2_filenames
        ]
        overall_filenames = np.unique(
            np.hstack([overall_1_filenames, overall_2_filenames])
        )

        columns = ["coincidence", "chance_coincidence", "z", "image_filename"]

        start = time.time()

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

                    x_2_coords = image_1_file.filter(pl.col("z") == z_plane)[
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

                    spot_1_indices, spot_2_indices = (
                        self.generate_spot_and_spot_indices(
                            centroids1, centroids2, image_size
                        )
                    )

                    (
                        temp_1_pl[j, 0],
                        temp_1_pl[j, 1],
                        temp_2_pl[j, 0],
                        temp_2_pl[j, 1],
                        raw_1_coincidence,
                        raw_2_coincidence,
                    ) = self.calculate_spot_to_spot_coincidence(
                        spot_1_indices,
                        spot_2_indices,
                        image_size,
                        n_iter=100,
                        blur_degree=blur_degree,
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

    def create_npuncta_cellmasks(self, cell_analysis, puncta, cell_mask, z_plane):
        from copy import copy

        cell_mask_toplot_AT = copy(cell_mask[:, :, z_plane - 1])
        cell_mask_toplot_UT = copy(cell_mask[:, :, z_plane - 1])
        cell_mask_toplot_R = copy(cell_mask[:, :, z_plane - 1])
        pil, areas, centroids = self.calculate_region_properties(cell_mask_toplot_AT)

        puncta = puncta.filter(pl.col("z") == z_plane)

        analysis = cell_analysis.filter(pl.col("z") == z_plane)

        for i, mask in enumerate(pil):
            cell_mask_toplot_AT[mask[:, 0], mask[:, 1]] = analysis[i, 5]
            cell_mask_toplot_UT[mask[:, 0], mask[:, 1]] = analysis[i, 4]
            if np.isfinite(analysis[i, 6]):
                val = analysis[i, 6]
            else:
                val = 0.0
            cell_mask_toplot_R[mask[:, 0], mask[:, 1]] = val
        return analysis, cell_mask_toplot_AT, cell_mask_toplot_UT, cell_mask_toplot_R

    def make_datarray_cell(
        self,
        clr,
        norm_std,
        norm_CSR,
        expected_spots,
        coincidence,
        chance_coincidence,
        coincidence_large,
        chance_coincidence_large,
        n_iter,
        columns,
        z_planes="none",
        analyse_clr=True,
    ):
        """
        makes a datarray in pandas for cell information

        Args:
            clr (ndarray): colocalisation likelihood ratios
            norm_std (np.1darray): array of standard deviations
            norm_CSR (np.1darray): array of CSRs
            expected_spots (np.1darray): array of expected spots
            coincidence (np.1darray): array of coincidence values
            chance_coincidence (np.1darray): array of chance coincidence values
            coincidence_large (np.1darray): array of coincidence values (large objects)
            chance_coincidence_large (np.1darray): array of chance coincidence values (large objects)
            n_iter (np.1darray): array of iteration muber
            columns (list of strings): column labels
            z_planes: z_planes to put in array (if needed)
            analyse_clr (boolean): If true, save clr

        Returns:
            to_save (pandas DataArray): pandas array to save

        """
        if isinstance(z_planes, str):
            if analyse_clr == True:
                dataarray_cell = np.vstack(
                    [
                        clr,
                        norm_std,
                        norm_CSR,
                        expected_spots,
                        coincidence,
                        chance_coincidence,
                        coincidence_large,
                        chance_coincidence_large,
                        n_iter,
                    ]
                )
            else:
                dataarray_cell = np.vstack(
                    [
                        coincidence,
                        chance_coincidence,
                        coincidence_large,
                        chance_coincidence_large,
                        n_iter,
                    ]
                )
        else:
            zps = np.zeros_like(coincidence)
            zps[z_planes] = z_planes + 1
            if analyse_clr == True:
                dataarray_cell = np.vstack(
                    [
                        clr,
                        norm_std,
                        norm_CSR,
                        expected_spots,
                        coincidence,
                        chance_coincidence,
                        coincidence_large,
                        chance_coincidence_large,
                        n_iter,
                        zps,
                    ]
                )
            else:
                dataarray_cell = np.vstack(
                    [
                        coincidence,
                        chance_coincidence,
                        coincidence_large,
                        chance_coincidence_large,
                        n_iter,
                        zps,
                    ]
                )
            dataarray_cell = dataarray_cell[:, np.sum(dataarray_cell, axis=0) > 0]
        return pl.DataFrame(data=dataarray_cell.T, schema=columns)
