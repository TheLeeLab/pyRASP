#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2025/01/10 13:31

@author: jbeckwith
"""
import sys
import os
import skimage as ski

module_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(module_dir)
from src import AnalysisFunctions
from src import HelperFunctions
import pytest
import numpy as np
import polars as pl

""" run from the command line with pytest Test_AnalysisFunctions.py """
""" Checking gives expected results """


class TestClass:

    @classmethod
    def setup_class(self):
        # setup things
        self.A_F = AnalysisFunctions.Analysis_Functions()
        self.H_F = HelperFunctions.Helper_Functions()
        return

    @classmethod
    def teardown_class(self):
        # teardown things
        del self.A_F
        del self.H_F
        return

    def test_count_spots(self):
        columns = ["sum_intensity_in_photons", "z", "image_filename"]
        z_planes = np.arange(1, 26)
        n_spots = np.random.randint(low=0, high=1000, size=len(z_planes))
        filename = "0".zfill(4)
        data = None
        for i, z in enumerate(z_planes):
            sum_intensity_in_photons = np.ones(n_spots[i])
            z_data = np.full_like(sum_intensity_in_photons, z)
            image_filename = np.full(len(sum_intensity_in_photons), filename)
            if data is not None:
                data = np.hstack(
                    [
                        data,
                        np.vstack([sum_intensity_in_photons, z_data, image_filename]),
                    ]
                )
            else:
                data = np.vstack([sum_intensity_in_photons, z_data, image_filename])
        database = self.H_F.clean_database(
            pl.DataFrame(data=data, schema=columns), columns
        )
        spot_numbers = self.A_F.count_spots(database)
        assert np.all(n_spots == spot_numbers["n_spots"].to_numpy())
        n_spots = np.full_like(z_planes, 500)
        data = None
        for i, z in enumerate(z_planes):
            sum_intensity_in_photons = np.hstack([np.full(250, 1), np.full(250, 2)])
            z_data = np.full_like(sum_intensity_in_photons, z)
            image_filename = np.full_like(sum_intensity_in_photons, filename)
            if data is not None:
                data = np.hstack(
                    [
                        data,
                        np.vstack([sum_intensity_in_photons, z_data, image_filename]),
                    ]
                )
            else:
                data = np.vstack([sum_intensity_in_photons, z_data, image_filename])
        database = self.H_F.clean_database(
            pl.DataFrame(data=data, schema=columns), columns
        )
        spot_numbers = self.A_F.count_spots(database, 1)
        assert np.all(250 == spot_numbers["n_spots_above"].to_numpy())
        assert np.all(250 == spot_numbers["n_spots_below"].to_numpy())

    def test_generate_indices(self):
        image_size = (10, 10)
        x = np.arange(3)
        y = np.arange(3)
        coords = np.array([x, y]).T
        assert np.all(
            np.array([0, 11, 22]) == self.A_F.generate_indices(coords, image_size)
        )
        mask = np.zeros([10, 10])
        for ind in x:
            mask[ind, ind] = 1
        assert np.all(
            np.array([0, 11, 22])
            == self.A_F.generate_indices(mask, image_size, is_mask=True)
        )
        large_object = ski.morphology.octagon(1, 1)
        image_size = large_object.shape
        pil, n_lo = self.A_F.generate_indices(
            large_object, image_size, is_mask=True, is_lo=True
        )
        assert n_lo == pytest.approx(1)
        assert np.all(np.array([1, 3, 4, 5, 7]) == np.sort(pil[0]))

    def test_reject_outliers(self):
        from copy import copy

        distribution = np.random.normal(size=10000)
        ks = np.linspace(2, 5, 20)
        q1, q2 = np.percentile(distribution, [25, 75])
        IQR = q2 - q1
        for k in ks:
            filtered_data, q1_F, q2_F, IQR_F = self.A_F.reject_outliers(
                data=distribution, k=k
            )
            assert q1 == pytest.approx(q1_F)
            assert q2 == pytest.approx(q2_F)
            assert IQR == pytest.approx(IQR_F)
            lower_limit = q1 - k * IQR
            upper_limit = q2 + k * IQR
            assert sum(filtered_data < lower_limit) == 0
            assert sum(filtered_data > upper_limit) == 0
            filtered_data_spec, _, _, _ = self.A_F.reject_outliers(
                data=distribution, k=k, q1=q1, q2=q2, IQR=IQR
            )
            assert np.all(filtered_data == filtered_data_spec)
            outlier_indices = self.A_F.reject_outliers(
                data=distribution, k=k, q1=q1, q2=q2, IQR=IQR, return_indices=True
            )
            f_dist = copy(distribution)
            f_dist[outlier_indices] = np.NAN
            f_dist = f_dist[~np.isnan(f_dist)]
            assert np.all(f_dist == filtered_data_spec)
