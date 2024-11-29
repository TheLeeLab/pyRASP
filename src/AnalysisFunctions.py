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
from copy import copy

import os
import sys

module_dir = os.path.dirname(__file__)
sys.path.append(module_dir)
import IOFunctions
import MultiD_RD_functions

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
            spots_per_plane[z[0]] = sum(1 * (database["z"].to_numpy() == (z[1] + 1)))

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

    def generate_lo_indices(self, mask, image_size):
        """
        makes mask indices from xy coordinates

        Args:
            mask (2D array): boolean matrix
            centroids (2D array): xy centroid coordinates
            image_size (tuple): Image dimensions (height, width).

        Returns:
            mask_indices (1D array): indices of mask
            spot_indices (1D array): indices of spots
        """
        pil, _, _ = self.calculate_region_properties(mask)
        for i in np.arange(len(pil)):
            pil[i] = np.ravel_multi_index(
                [pil[i][:, 0], pil[i][:, 1]], image_size, order="F"
            )
        return pil, len(pil)

    def generate_mask_indices(self, mask, image_size):
        """
        makes mask indices from xy coordinates

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
        return mask_indices

    def generate_spot_indices(self, centroids, image_size):
        """
        makes spot indices from xy coordinates

        Args:
            centroids1 (2D array): xy centroid coordinates
            centroids2 (2D array): xy centroid coordinates
            image_size (tuple): Image dimensions (height, width).

        Returns:
            spot1_indices (1D array): indices of spots1
            spot2_indices (1D array): indices of spots2
        """
        spot_indices = np.ravel_multi_index(centroids.T, image_size, order="F")
        return spot_indices

    def calculate_spot_to_cell_numbers(
        self,
        spot_indices,
        mask_indices,
        image_size,
        n_iter=100,
        blur_degree=1,
        analytical_solution=True,
    ):
        """
        gets cell analysis likelihood, as well as reporting error
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
            olig_cell_ratio (float): coincidence per cell, normalised to chance
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
        n_iter_rec = n_iter
        possible_indices = np.arange(
            0, np.prod(image_size)
        )  # get list of where is possible to exist in an image

        raw_colocalisation = self.test_spot_spot_overlap(
            spot_indices, mask_indices, original_n_spots, raw=True
        )

        n_olig_in_cell = np.sum(
            raw_colocalisation
        )  # generate number of oligomers in cell
        if analytical_solution == True:
            n_olig_in_cell_random = original_n_spots * (
                len(mask_indices.ravel()) / len(possible_indices)
            )
        else:
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
            n_olig_in_cell_random = np.zeros([n_iter])  # generate CSR array to fill in
            for i in np.arange(n_iter):
                n_olig_in_cell_random[i] = np.sum(
                    self.test_spot_spot_overlap(
                        random_spot_locations[i, :],
                        mask_indices,
                        original_n_spots,
                        raw=True,
                    )
                )
        if (n_olig_in_cell == 0) or (np.nanmean(n_olig_in_cell_random) == 0):
            olig_cell_ratio = np.NAN
        else:
            olig_cell_ratio = np.divide(
                n_olig_in_cell, np.nanmean(n_olig_in_cell_random)
            )
        return olig_cell_ratio, n_olig_in_cell, n_iter_rec

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
        n_iter_rec = n_iter
        possible_indices = np.arange(
            0, np.prod(image_size)
        )  # get list of where is possible to exist in an image

        raw_colocalisation = self.test_spot_spot_overlap(
            spot_indices, mask_indices, original_n_spots, raw=True
        )

        coincidence = np.divide(
            np.sum(raw_colocalisation), original_n_spots
        )  # generate coincidence

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
            chance_coincidence_raw[i] = np.divide(
                np.sum(
                    self.test_spot_spot_overlap(
                        random_spot_locations[i, :],
                        mask_indices,
                        original_n_spots,
                        raw=True,
                    )
                ),
                original_n_spots,
            )

        chance_coincidence = np.nanmean(chance_coincidence_raw)
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

    def rejectoutliers_value(self, data, q1=None, q2=None, IQR=None):
        """rejectoutliers_value function
        # rejects outliers from data, does iqr method (i.e. anything below
        lower quartile (25 percent) or above upper quartile (75 percent)
        is rejected). If given q1, q2, and IQR, uses directly.

        Args:
            data (pl.dataarray): polars dataarray
            q1 (float): if float, uses directly
            q2 (float): if float, uses directly
            IQR (float): if float, uses directly

        Returns:
            newdata (np.1darray): data matrix"""
        if (
            isinstance(q1, type(None))
            and isinstance(q2, type(None))
            and isinstance(IQR, type(None))
        ):
            from scipy.stats import iqr

            IQR = iqr(data["sum_intensity_in_photons"].to_numpy())
            q1, q2 = np.percentile(
                data["sum_intensity_in_photons"].to_numpy(), q=(25, 75)
            )

            upper_limit = (1.5 * IQR) + q2
            lower_limit = q1 - (1.5 * IQR)
            newdata = data.filter(
                (pl.col("sum_intensity_in_photons") >= lower_limit)
                & (pl.col("sum_intensity_in_photons") <= upper_limit)
            )
        else:
            upper_limit = (1.5 * IQR) + q2
            lower_limit = q1 - (1.5 * IQR)
            newdata = data.filter(
                (pl.col("sum_intensity_in_photons") >= lower_limit)
                & (pl.col("sum_intensity_in_photons") <= upper_limit)
            )
        return newdata, q1, q2, IQR

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

    def colocalise_with_threshold(
        self,
        analysis_file,
        threshold,
        protein_string,
        lo_string,
        cell_string,
        coloc_type=1,
        imtype=".tif",
        blur_degree=1,
        calc_clr=False,
        aboveT=1,
        lower_cell_size_threshold=0,
        upper_cell_size_threshold=np.inf,
    ):
        """
        Does colocalisation analysis of spots vs mask with an additional threshold.

        Args:
            analysis_file (string): The analysis file of puncta. If large objects are to be
                                    analysed as puncta, make sure this is a large object file.
            threshold (float): The photon threshold for puncta.
            protein_string (str): string of protein images
            lo_string (str): string of larger object images
            coloc_typ (boolean): if 1 (default), for cells. if 0, for large protein objects.
                                if 2, between cell mask and large protein objects.
            imtype (str): image end string
            blur_degree (int): degree of blur to apply to puncta
            calc_clr (boolean): calculate clr yes/no
            aboveT (int): do the calculation above or below threshold
            lower_cell_size_threshold (float): lower threshold of cell size
            upper_cell_size_threshold (float): upper threshold of cell size

        Returns:
            lo_analysis (pl.DataFrame): polars dataframe of the cell analysis
            spot_analysis (pl.DataFrame): polars dataframe of the spot analysis
        """
        if coloc_type == 1:
            end_str = "_cellMask.tiff"
        elif coloc_type == 0:
            end_str = "_loMask.tiff"
        else:
            end_str = None

        spots_with_intensities = pl.read_csv(analysis_file)
        if coloc_type == 2:
            if "mean_intensity_in_photons" not in spots_with_intensities.columns:
                print("Large object analysis file not loaded in. Code will fail.")
                return
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

            if (calc_clr == False) and (coloc_type == 2):
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

            start = time.time()

            for i, image in enumerate(image_filenames):
                if end_str is not None:
                    lo_mask = IO.read_tiff(
                        os.path.join(
                            analysis_directory,
                            os.path.split(image.split(imtype)[0])[-1].split(
                                protein_string
                            )[0]
                            + str(lo_string)
                            + end_str,
                        )
                    )
                    lo_mask, _, _, _ = self.threshold_cell_areas(
                        lo_mask,
                        lower_cell_size_threshold,
                        upper_cell_size_threshold,
                        [False, False],
                    )
                else:
                    lo_mask = IO.read_tiff(
                        os.path.join(
                            analysis_directory,
                            os.path.split(image.split(imtype)[0])[-1].split(
                                protein_string
                            )[0]
                            + str(lo_string)
                            + "_loMask.tiff",
                        )
                    )
                    cell_mask = IO.read_tiff(
                        os.path.join(
                            analysis_directory,
                            os.path.split(image.split(imtype)[0])[-1].split(
                                protein_string
                            )[0]
                            + str(cell_string)
                            + "_cellMask.tiff",
                        )
                    )
                    cell_mask, _, _, _ = self.threshold_cell_areas(
                        cell_mask,
                        lower_cell_size_threshold,
                        upper_cell_size_threshold,
                        [False, False],
                    )

                image_size = lo_mask.shape[:-1]
                image_file = spots_with_intensities.filter(
                    pl.col("image_filename") == image
                )
                z_planes = np.unique(image_file["z"].to_numpy())

                dataarray = np.zeros([len(z_planes), len(columns)])

                temp_pl = pl.DataFrame(data=dataarray, schema=columns)

                for j, z_plane in enumerate(z_planes):
                    if coloc_type != 2:
                        filtered_file = image_file.filter(pl.col("z") == z_plane)
                        xcoords = filtered_file["x"].to_numpy()
                        ycoords = filtered_file["y"].to_numpy()
                        if lo_mask.shape[-1] > z_planes[-1]:
                            mask = lo_mask[:, :, int(z_plane) - 1]
                        else:
                            mask = lo_mask[:, :, j]
                        centroids = np.asarray(
                            np.vstack([xcoords, ycoords]), dtype=int
                        ).T
                        mask_indices = self.generate_mask_indices(mask, image_size)
                        spot_indices = self.generate_spot_indices(centroids, image_size)
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
                    else:
                        if lo_mask.shape[-1] > z_planes[-1]:
                            mask_lo = lo_mask[:, :, int(z_plane) - 1]
                            mask_cell = cell_mask[:, :, int(z_plane) - 1]
                        else:
                            mask_lo = lo_mask[:, :, j]
                            mask_cell = cell_mask[:, :, j]
                        mask_lo_indices, n_largeobjs = self.generate_lo_indices(
                            mask_lo, image_size
                        )
                        mask_cell_indices = self.generate_mask_indices(
                            mask_cell, image_size
                        )
                        (
                            temp_pl[j, 0],
                            temp_pl[j, 1],
                            raw_colocalisation,
                        ) = self.calculate_largeobj_coincidence(
                            mask_lo_indices,
                            mask_cell_indices,
                            n_largeobjs,
                            image_size,
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
                    lo_analysis = temp_pl
                    spot_analysis = image_file
                else:
                    lo_analysis = pl.concat([lo_analysis, temp_pl])
                    spot_analysis = pl.concat([spot_analysis, image_file])

                print(
                    "Computing colocalisation     File {}/{}    Time elapsed: {:.3f} s".format(
                        i + 1, len(image_filenames), time.time() - start
                    ),
                    end="\r",
                    flush=True,
                )
            return lo_analysis, spot_analysis
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
        lower_cell_size_threshold=100,
        upper_cell_size_threshold=np.inf,
        blur_degree=1,
        cell_string="C0",
        protein_string="C1",
        imtype=".tif",
        aboveT=1,
        z_project_first=[True, True],
        q1=None,
        q2=None,
        IQR=None,
    ):
        """
        Does analysis of number of oligomers in a mask area per "segmented"
        cell area.

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
            z_project_first (boolean): if True (default), does a z projection before
                                    thresholding cell size. If false, does the opposite.
            q1 (float): if float, adds in IQR filter
            q2 (float): if float, adds in IQR filter
            IQR (Float): if float, adds in IQR filter

        Returns:
            cell_punctum_analysis (pl.DataFrame): polars datarray of the cell analysis
        """

        analysis_data = pl.read_csv(analysis_file)
        if (
            ~isinstance(q1, type(None))
            and ~isinstance(q2, type(None))
            and ~isinstance(IQR, type(None))
        ):
            analysis_data, _, _, _ = self.rejectoutliers_value(
                analysis_data, q1, q2, IQR
            )

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
        start = time.time()
        cell_punctum_analysis = None
        if len(analysis_data) > 0:
            files = np.unique(analysis_data["image_filename"].to_numpy())

            for i, file in enumerate(files):
                raw_cell_mask = IO.read_tiff(
                    os.path.join(
                        analysis_directory,
                        os.path.split(file.split(imtype)[0])[-1].split(protein_string)[
                            0
                        ]
                        + str(cell_string)
                        + "_cellMask.tiff",
                    )
                )
                subset = analysis_data.filter(pl.col("image_filename") == file)
                cell_mask, pil_mask, centroids, areas = self.threshold_cell_areas(
                    raw_cell_mask,
                    lower_cell_size_threshold,
                    upper_cell_size_threshold=upper_cell_size_threshold,
                    z_project=z_project_first,
                )
                image_size = cell_mask.shape

                x_m = centroids[:, 0]
                y_m = centroids[:, 1]
                x = subset["x"].to_numpy()
                y = subset["y"].to_numpy()
                bounds_x = (x < image_size[0]) & (x >= 0)
                bounds_y = (y < image_size[1]) & (y >= 0)
                bounds = bounds_x * bounds_y
                x = x[bounds]
                y = y[bounds]
                centroids_puncta = np.asarray(np.vstack([x, y]), dtype=int)
                spot_indices = np.unique(
                    np.ravel_multi_index(centroids_puncta, image_size, order="F")
                )
                filename_tosave = np.full_like(x_m, file, dtype="object")
                n_spots_in_object = np.zeros_like(x_m)
                n_cell_ratios = np.zeros_like(x_m)

                for k in np.arange(len(areas)):
                    coords = pil_mask[k]
                    xm = coords[:, 0]
                    ym = coords[:, 1]
                    if (np.any(xm > image_size[0])) or (np.any(ym > image_size[1])):
                        n_cell_ratios[k] = np.NAN
                        n_spots_in_object[k] = np.NAN
                    else:
                        coordinates_mask = np.asarray(np.vstack([xm, ym]), dtype=int)
                        mask_indices = np.ravel_multi_index(
                            coordinates_mask, image_size, order="F"
                        )
                        olig_cell_ratio, n_olig_in_cell, n_iter_rec = (
                            self.calculate_spot_to_cell_numbers(
                                spot_indices,
                                mask_indices,
                                image_size,
                                n_iter=1,
                                blur_degree=1,
                            )
                        )
                        n_cell_ratios[k] = olig_cell_ratio
                        n_spots_in_object[k] = n_olig_in_cell

                if len(areas) > 0:
                    data = {
                        "area/pixels": areas,
                        "x_centre": x_m,
                        "y_centre": y_m,
                        "puncta_cell_likelihood": n_cell_ratios,
                        "n_puncta_in_cell": n_spots_in_object,
                        "image_filename": filename_tosave,
                    }
                    if isinstance(cell_punctum_analysis, type(None)):
                        cell_punctum_analysis = pl.DataFrame(data)
                    else:
                        cell_punctum_analysis = pl.concat(
                            [cell_punctum_analysis, pl.DataFrame(data)]
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
            cell_punctum_analysis = cell_punctum_analysis.rechunk()
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

        if len(overall_filenames) == 0:
            plane_1_analysis = None
            plane_2_analysis = None
            spot_1_analysis = None
            spot_2_analysis = None
            return (
                plane_1_analysis,
                plane_2_analysis,
                spot_1_analysis,
                spot_2_analysis,
            )

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

                    spot_1_indices = self.generate_spot_indices(centroids1, image_size)
                    spot_2_indices = self.generate_spot_indices(centroids2, image_size)
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
            cell_mask = np.sum(cell_mask_raw, axis=-1).clip(0, 1)
        else:
            cell_mask = cell_mask_raw.copy()
        # Handle multi-dimensional and 2D masks differently
        if len(cell_mask.shape) > 2:
            # Process 3D mask
            cell_mask_new = cell_mask.copy()
            for plane in range(cell_mask.shape[-1]):
                plane_mask = cell_mask_new[:, :, plane]
                pil, areas, centroids = self.calculate_region_properties(plane_mask)
                # Vectorized filtering
                mask = (areas >= lower_cell_size_threshold) & (
                    areas < upper_cell_size_threshold
                )
                # Update mask
                for c in np.where(~mask)[0]:
                    cell_mask_new[pil[c][:, 0], pil[c][:, 1], plane] = 0
            # Reconstruct final mask
            if z_project[1]:
                cell_mask_new = np.sum(cell_mask_new, axis=-1).clip(0, 1)
        else:
            # Process 2D mask
            cell_mask_new = cell_mask.copy()
            pil, areas, centroids = self.calculate_region_properties(cell_mask_new)
            # Vectorized filtering
            mask = (areas >= lower_cell_size_threshold) & (
                areas < upper_cell_size_threshold
            )
            # Update mask
            for c in np.where(~mask)[0]:
                cell_mask_new[pil[c][:, 0], pil[c][:, 1]] = 0
        # Final region properties calculation
        if z_project[0] and z_project[1]:
            pil, areas, centroids = self.calculate_region_properties(cell_mask_new)
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
