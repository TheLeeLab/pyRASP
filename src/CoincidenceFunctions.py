# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to coincidence, relating to the RASP 
project.
jsb92, 2025/01/01
"""
import numpy as np
from numba import jit
import skimage as ski


class Coincidence_Functions:
    def __init__(self):
        return

    def calculate_coincidence(
        self,
        spot_indices,
        mask_indices,
        image_size,
        analysis_type="spot_to_cell",
        spot_intensities=None,
        median_intensity=None,
        largeobj_indices=None,
        n_largeobjs=None,
        second_spot_indices=None,
        tol=0.01,
        n_iter=100,
        blur_degree=1,
        analytical_solution=True,
        max_iter=1000,
    ):
        """
        Calculates various types of coincidence metrics based on the specified analysis type.

        Args:
            spot_indices (np.ndarray): Array of indices for the spots.
            mask_indices (np.ndarray): Array of indices for the mask.
            image_size (tuple): Size of the image as (height, width).
            analysis_type (str, optional): Type of analysis to perform. Options are:
                - "spot_to_cell": Calculate spot to cell metrics.
                - "protein_load": Calculate protein load metrics.
                - "spot_to_mask": Calculate spot to mask coincidence.
                - "spot_to_spot": Calculate coincidence between two sets of spots.
                - "largeobj": Calculate large object coincidence.
                - "colocalisation_likelihood": Calculate colocalisation likelihood ratio.
                Default is "spot_to_cell".
            spot_intensities (np.ndarray, optional): Array of intensities for the spots. Required for "protein_load" analysis.
            median_intensity (float, optional): Median intensity of the spots. Required for "protein_load" analysis.
            largeobj_indices (list, optional): List of indices for large objects. Required for "largeobj" analysis.
            n_largeobjs (int, optional): Number of large objects. Required for "largeobj" analysis.
            second_spot_indices (np.ndarray, optional): Array of indices for the second set of spots. Required for "spot_to_spot" analysis.
            tol (float, optional): Tolerance for convergence in colocalisation likelihood ratio calculation. Default is 0.01.
            n_iter (int, optional): Number of iterations for randomization. Default is 100.
            blur_degree (int, optional): Degree of blur to apply. Default is 1.
            analytical_solution (bool, optional): Use analytical solution for randomization. Default is True.
            max_iter (int, optional): Maximum number of iterations for colocalisation likelihood ratio calculation. Default is 1000.

        Returns:
            tuple: Metrics based on the specified analysis type. The structure of the tuple varies depending on the analysis type:
                - "spot_to_cell": (olig_cell_ratio, n_olig_in_cell, n_iter)
                - "protein_load": (olig_cell_ratio, n_olig_in_cell, n_iter)
                - "spot_to_mask": (coincidence, chance_coincidence, raw_colocalisation, n_iter)
                - "spot_to_spot": (coincidence_1, chance_coincidence_1, coincidence_2, chance_coincidence_2, raw_coincidence_1, raw_coincidence_2)
                - "largeobj": (coincidence, average_chance_coincidence, raw_colocalisation)
                - "colocalisation_likelihood": (colocalisation_likelihood_ratio, norm_std, norm_CSR, expected_spots_iter, coincidence, chance_coincidence, raw_colocalisation, n_iter)

        Raises:
            ValueError: If an invalid analysis type is specified.
        """
        if len(spot_indices) == 0 and analysis_type not in ["largeobj"]:
            return self._handle_empty_spots(spot_indices)
        n_spots = len(spot_indices)
        if analysis_type not in ["spot_to_spot"]:
            spot_indices = self._apply_blur(spot_indices, image_size, blur_degree)

        if analysis_type == "spot_to_cell":
            return self._calculate_metrics(
                spot_indices,
                mask_indices,
                image_size,
                n_spots,
                n_iter,
                analytical_solution,
            )
        elif analysis_type == "protein_load":
            return self._calculate_protein_load(
                spot_indices,
                n_spots,
                mask_indices,
                spot_intensities,
                median_intensity,
                image_size,
                n_iter,
                analytical_solution,
            )
        elif analysis_type == "spot_to_mask":
            return self._calculate_spot_to_mask_coincidence(
                spot_indices, n_spots, mask_indices, image_size, n_iter, blur_degree
            )
        elif analysis_type == "spot_to_spot":
            return self._calculate_spot_to_spot_coincidence(
                spot_indices, second_spot_indices, image_size, n_iter, blur_degree
            )
        elif analysis_type == "largeobj":
            return self._calculate_largeobj_coincidence(
                largeobj_indices, mask_indices, n_largeobjs, image_size
            )
        elif analysis_type == "colocalisation_likelihood":
            return self._calculate_colocalisation_likelihood_ratio(
                spot_indices,
                n_spots,
                mask_indices,
                image_size,
                tol,
                n_iter,
                blur_degree,
                max_iter,
            )
        else:
            raise ValueError("Invalid analysis type specified.")

    def _handle_empty_spots(self, spot_indices):
        return np.nan, np.nan, np.full_like(spot_indices, np.nan), 0

    def _apply_blur(self, spot_indices, image_size, blur_degree):
        if blur_degree > 0:
            spot_indices = self.dilate_pixels(
                spot_indices, image_size, blur_degree + 1, blur_degree
            )
        return spot_indices

    def _calculate_metrics(
        self,
        spot_indices,
        mask_indices,
        image_size,
        n_spots,
        n_iter,
        analytical_solution,
    ):
        raw_colocalisation, n_olig_in_cell = self._compute_colocalisation(
            spot_indices, mask_indices, n_spots
        )

        n_olig_in_cell_random = self._calculate_random_colocalisation(
            spot_indices,
            mask_indices,
            image_size,
            n_iter,
            analytical_solution,
            n_spots,
        )

        if n_olig_in_cell == 0 or n_olig_in_cell_random == 0:
            return np.nan, n_olig_in_cell, n_iter

        olig_cell_ratio = self._compute_olig_cell_ratio(
            n_olig_in_cell, n_olig_in_cell_random
        )

        return olig_cell_ratio, n_olig_in_cell, n_iter

    def _calculate_protein_load(
        self,
        spot_indices,
        n_spots,
        mask_indices,
        spot_intensities,
        median_intensity,
        image_size,
        n_iter,
        analytical_solution,
    ):
        raw_colocalisation, n_olig_in_cell = self._compute_colocalisation(
            spot_indices, mask_indices, n_spots
        )

        n_olig_in_cell_random = self._calculate_random_colocalisation(
            spot_indices,
            mask_indices,
            image_size,
            n_iter,
            analytical_solution,
            n_spots,
        )

        if n_olig_in_cell == 0 or n_olig_in_cell_random == 0:
            return np.nan, n_olig_in_cell, n_iter

        olig_cell_ratio = self._compute_protein_load_ratio(
            n_olig_in_cell,
            n_olig_in_cell_random,
            raw_colocalisation,
            spot_intensities,
            median_intensity,
        )

        return olig_cell_ratio, n_olig_in_cell, n_iter

    def _calculate_spot_to_mask_coincidence(
        self, spot_indices, n_spots, mask_indices, image_size, n_iter, blur_degree
    ):
        raw_colocalisation = self.test_spot_spot_overlap(
            spot_indices, mask_indices, n_spots, raw=True
        )
        coincidence = np.divide(np.sum(raw_colocalisation), n_spots)

        chance_coincidence = self._calculate_chance_coincidence(
            spot_indices, n_spots, mask_indices, image_size, n_iter, blur_degree
        )
        return coincidence, chance_coincidence, raw_colocalisation, n_iter

    def _calculate_spot_to_spot_coincidence(
        self, spot_1_indices, spot_2_indices, image_size, n_iter, blur_degree
    ):
        original_n_spots_1 = len(spot_1_indices)
        original_n_spots_2 = len(spot_2_indices)

        if original_n_spots_1 == 0 or original_n_spots_2 == 0:
            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.full_like(spot_1_indices, np.nan),
                np.full_like(spot_2_indices, np.nan),
            )

        spot_1_indices = self._apply_blur(spot_1_indices, image_size, blur_degree)
        spot_2_indices = self._apply_blur(spot_2_indices, image_size, blur_degree)

        raw_coincidence_1, raw_coincidence_2 = self._compute_spot_to_spot_overlap(
            spot_1_indices, spot_2_indices, original_n_spots_1, original_n_spots_2
        )

        coincidence_1 = np.divide(np.sum(raw_coincidence_1), original_n_spots_1)
        coincidence_2 = np.divide(np.sum(raw_coincidence_2), original_n_spots_2)

        chance_coincidence_1, chance_coincidence_2 = (
            self._calculate_chance_coincidence_spots(
                spot_1_indices,
                spot_2_indices,
                image_size,
                n_iter,
                blur_degree,
                original_n_spots_1,
                original_n_spots_2,
            )
        )

        return (
            coincidence_1,
            chance_coincidence_1,
            coincidence_2,
            chance_coincidence_2,
            raw_coincidence_1,
            raw_coincidence_2,
        )

    def _calculate_largeobj_coincidence(
        self, largeobj_indices, mask_indices, n_largeobjs, image_size
    ):
        if n_largeobjs == 0:
            return 0.0, 0.0, np.array([0.0])

        mask_fill = self.calculate_mask_fill(mask_indices, image_size)
        expected_spots = np.multiply(mask_fill, n_largeobjs)
        if np.isclose(expected_spots, 0.0, atol=1e-4):
            return 0, 0, np.zeros(n_largeobjs)

        raw_colocalisation = self.test_largeobj_mask_overlap(
            largeobj_indices, mask_indices, n_largeobjs, raw=True
        )
        nspots_in_mask = np.sum(raw_colocalisation)
        coincidence = np.divide(nspots_in_mask, n_largeobjs)

        chance_coincidence = self._calculate_chance_coincidence_largeobj(
            largeobj_indices, mask_indices, image_size, n_largeobjs
        )

        return coincidence, np.mean(chance_coincidence), raw_colocalisation

    def _calculate_colocalisation_likelihood_ratio(
        self,
        spot_indices,
        n_spots,
        mask_indices,
        image_size,
        tol,
        n_iter,
        blur_degree,
        max_iter,
    ):
        if n_spots == 0:
            return (
                np.nan,
                np.nan,
                np.nan,
                0,
                np.nan,
                np.nan,
                np.full(n_spots, np.nan),
                0,
            )

        mask_fill = self.calculate_mask_fill(mask_indices, image_size)
        expected_spots_iter = np.multiply(mask_fill, n_spots)

        if np.isclose(expected_spots_iter, 0.0, atol=1e-4):
            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.full(n_spots, np.nan),
                0,
            )

        raw_colocalisation = self.test_spot_spot_overlap(
            spot_indices, mask_indices, n_spots, raw=True
        )
        nspots_in_mask = np.sum(raw_colocalisation)
        colocalisation_likelihood_ratio = np.divide(nspots_in_mask, expected_spots_iter)
        coincidence = np.divide(nspots_in_mask, n_spots)

        CSR, meanCSR = self._calculate_CSR(
            spot_indices,
            n_spots,
            mask_indices,
            image_size,
            n_iter,
            blur_degree,
            expected_spots_iter,
        )
        chance_coincidence = np.divide(np.nanmean(CSR), n_spots)
        CSR_diff = np.abs(meanCSR - 1.0)

        while (CSR_diff > tol) and (n_iter < max_iter):
            n_iter += n_iter
            CSR, meanCSR = self._calculate_CSR(
                spot_indices,
                n_spots,
                mask_indices,
                image_size,
                n_iter,
                blur_degree,
                expected_spots_iter,
            )
            CSR_diff = np.abs(meanCSR - 1.0)

        norm_CSR, norm_std = self._calculate_norm_CSR(expected_spots_iter, CSR)

        return (
            colocalisation_likelihood_ratio,
            norm_std,
            norm_CSR,
            expected_spots_iter,
            coincidence,
            chance_coincidence,
            raw_colocalisation,
            n_iter,
        )

    def _compute_colocalisation(self, spot_indices, mask_indices, original_n_spots):
        raw_colocalisation = self.test_spot_spot_overlap(
            spot_indices, mask_indices, original_n_spots, raw=True
        )
        n_olig_in_cell = np.sum(raw_colocalisation)
        return raw_colocalisation, n_olig_in_cell

    def _calculate_random_colocalisation(
        self,
        spot_indices,
        mask_indices,
        image_size,
        n_iter,
        analytical_solution,
        original_n_spots,
    ):
        if analytical_solution:
            return original_n_spots * (len(mask_indices.ravel()) / np.prod(image_size))
        else:
            return np.mean(
                [
                    np.sum(
                        self.test_spot_spot_overlap(
                            np.random.choice(
                                np.arange(np.prod(image_size)), original_n_spots
                            ).reshape(-1, 1),
                            mask_indices,
                            original_n_spots,
                            raw=True,
                        )
                    )
                    for _ in range(n_iter)
                ]
            )

    def _compute_protein_load_ratio(
        self,
        n_olig_in_cell,
        n_olig_in_cell_random,
        raw_colocalisation,
        spot_intensities,
        median_intensity,
    ):
        cell_brightness = np.median(spot_intensities[np.where(raw_colocalisation)[0]])
        return (n_olig_in_cell * cell_brightness) / (
            n_olig_in_cell_random * median_intensity
        )

    def _compute_olig_cell_ratio(self, n_olig_in_cell, n_olig_in_cell_random):
        return np.divide(n_olig_in_cell, np.nanmean(n_olig_in_cell_random))

    def _calculate_chance_coincidence(
        self, spot_indices, n_spots, mask_indices, image_size, n_iter, blur_degree
    ):
        random_spot_locations = np.random.choice(
            np.arange(np.prod(image_size)), size=(n_iter, n_spots)
        )
        if blur_degree > 0:
            random_spot_locations = self.dilate_pixels(
                random_spot_locations,
                image_size,
                width=blur_degree + 1,
                edge=blur_degree,
            )
        chance_coincidence_raw = np.zeros([n_iter])
        for i in range(n_iter):
            chance_coincidence_raw[i] = np.divide(
                np.sum(
                    self.test_spot_spot_overlap(
                        random_spot_locations[i, :],
                        mask_indices,
                        n_spots,
                        raw=True,
                    )
                ),
                n_spots,
            )
        return np.nanmean(chance_coincidence_raw)

    def _calculate_chance_coincidence_spots(
        self,
        spot_1_indices,
        spot_2_indices,
        image_size,
        n_iter,
        blur_degree,
        original_n_spots_1,
        original_n_spots_2,
    ):
        possible_indices = np.arange(np.prod(image_size))
        random_spot_locations_1 = np.random.choice(
            possible_indices, size=(n_iter, original_n_spots_1)
        )
        random_spot_locations_2 = np.random.choice(
            possible_indices, size=(n_iter, original_n_spots_2)
        )

        if blur_degree > 0:
            random_spot_locations_1 = self.dilate_pixels(
                random_spot_locations_1,
                image_size,
                width=blur_degree + 1,
                edge=blur_degree,
            )
            random_spot_locations_2 = self.dilate_pixels(
                random_spot_locations_2,
                image_size,
                width=blur_degree + 1,
                edge=blur_degree,
            )

        CC_1 = np.zeros([n_iter])
        CC_2 = np.zeros([n_iter])
        for i in range(n_iter):
            rc_1 = self.test_spot_spot_overlap(
                random_spot_locations_1[i, :], spot_2_indices, original_n_spots_1
            )
            rc_1[rc_1 >= 1.0] = 1.0
            CC_1[i] = np.divide(np.sum(rc_1), original_n_spots_1)
            rc_2 = self.test_spot_spot_overlap(
                random_spot_locations_2[i, :], spot_1_indices, original_n_spots_2
            )
            rc_2[rc_2 >= 1.0] = 1.0
            CC_2[i] = np.divide(np.sum(rc_2), original_n_spots_2)

        return np.nanmean(CC_1), np.nanmean(CC_2)

    def _calculate_chance_coincidence_largeobj(
        self, largeobj_indices, mask_indices, image_size, n_largeobjs
    ):
        random_spot_locations = self.random_largeobj_locations(
            largeobj_indices, np.prod(image_size), 100
        )
        chance_coincidence_raw = np.zeros(100)
        for i in range(100):
            chance_coincidence_raw[i] = np.divide(
                self.test_largeobj_mask_overlap(
                    random_spot_locations[i], mask_indices, n_largeobjs, raw=False
                ),
                n_largeobjs,
            )
        return chance_coincidence_raw

    def _calculate_CSR(
        self,
        spot_indices,
        n_spots,
        mask_indices,
        image_size,
        n_iter,
        blur_degree,
        expected_spots_iter,
    ):
        possible_indices = np.arange(np.prod(image_size))
        random_spot_locations = np.random.choice(
            possible_indices, size=(n_iter, n_spots)
        )
        if blur_degree > 0:
            random_spot_locations = self.dilate_pixels(
                random_spot_locations,
                image_size,
                width=blur_degree + 1,
                edge=blur_degree,
            )
        CSR = np.zeros([n_iter])
        for i in range(n_iter):
            CSR[i] = self.test_spot_spot_overlap(
                random_spot_locations[i, :], mask_indices, n_spots
            )
        meanCSR = np.divide(np.nanmean(CSR), expected_spots_iter)
        return CSR, meanCSR

    def _calculate_norm_CSR(self, expected_spots_iter, CSR):
        if expected_spots_iter > 0 and np.mean(CSR) > 0:
            norm_CSR = np.divide(np.nanmean(CSR), expected_spots_iter)
            norm_std = np.divide(np.nanstd(CSR), np.nanmean(CSR))
        else:
            norm_CSR = np.nan
            norm_std = np.nan
        return norm_CSR, norm_std

    def _compute_spot_to_spot_overlap(
        self, spot_1_indices, spot_2_indices, original_n_spots_1, original_n_spots_2
    ):
        raw_coincidence_1 = self.test_spot_spot_overlap(
            spot_1_indices, spot_2_indices, original_n_spots_1
        )
        raw_coincidence_2 = self.test_spot_spot_overlap(
            spot_2_indices, spot_1_indices, original_n_spots_2
        )
        return raw_coincidence_1, raw_coincidence_2

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
        n_spots1_in_spots2 = np.asarray(
            np.sum(np.isin(spot_1_indices.reshape(newdims), spot_2_indices), axis=1),
            dtype=bool,
        )
        if raw == True:
            return n_spots1_in_spots2
        else:
            return np.sum(n_spots1_in_spots2)

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

        if len(indices.shape) > 1:
            x = (
                np.tile(x, (len(indices), 1)).T[:, :, np.newaxis]
                + np.asarray(centroid[0, :], dtype=int)
            ).T

            y = (
                np.tile(y, (len(indices), 1)).T[:, :, np.newaxis]
                + np.asarray(centroid[1, :], dtype=int)
            ).T
            new_dims = (indices.shape[0], int(len(x.ravel()) / indices.shape[0]))
        else:
            x = (
                np.tile(x, (len(indices), 1)).T + np.asarray(centroid[0, :], dtype=int)
            ).T

            y = (
                np.tile(y, (len(indices), 1)).T + np.asarray(centroid[1, :], dtype=int)
            ).T
            new_dims = (len(indices), int(len(x.ravel()) / len(indices)))
        dilated_indices = np.ravel_multi_index(
            np.vstack([x.ravel(), y.ravel()]), image_size, order="F", mode="wrap"
        ).reshape(new_dims)

        if len(indices) == 1:
            return dilated_indices[0]
        elif len(indices.shape) == 1:
            return dilated_indices.ravel()
        else:
            return dilated_indices

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
