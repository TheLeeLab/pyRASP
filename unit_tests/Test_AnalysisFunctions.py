#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2025/01/10 13:31

@author: jbeckwith
"""
import sys
import os

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
