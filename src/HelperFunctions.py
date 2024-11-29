# -*- coding: utf-8 -*-
"""
This class contains helper functions pertaining to analysis of images based on their 
radiality, relating to the RASP concept.
jsb92, 2024/01/02
"""
import numpy as np
import polars as pl


class Helper_Functions:
    def __init__(self):
        self = self
        return

    def make_datarray_largeobjects(
        self,
        areas_large,
        centroids_large,
        sumintensities_large,
        meanintensities_large,
        columns,
        z_planes=0,
    ):
        """
        makes a datarray in pandas for large object information

        Args:
            areas_large (np.1darray): areas in pixels
            centroids_large (np.1darray): centroids of large objects
            meanintensities_large (np.1darray): mean intensities of large objects
            columns (list of strings): column labels
            z_planes: z_planes to put in array (if needed); if int, assumes only
                one z-plane

        Returns:
            to_save_largeobjects (pandas DataArray) pandas array to save
            columns_large = ['x', 'y', 'z', 'area', 'mean_intensity_in_photons', 'zi', 'zf']

        """
        if isinstance(z_planes, int):
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
            dataarray = None
            for z in z_planes:
                if len(areas_large[z]) > 0:
                    stack = np.asarray(
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
                    if dataarray is not None:
                        dataarray = np.vstack([dataarray, np.squeeze(stack.T)])
                    else:
                        dataarray = np.squeeze(stack.T)
        return pl.DataFrame(data=dataarray.T, schema=columns)

    def make_datarray_spot(
        self,
        centroids,
        estimated_intensity,
        estimated_background,
        estimated_background_perpixel,
        columns,
        z_planes=0,
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
