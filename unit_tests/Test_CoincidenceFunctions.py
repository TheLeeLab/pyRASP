#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2025/01/09 10:20

@author: jbeckwith
"""
import sys
import os

module_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(module_dir)
from src import CoincidenceFunctions
import pytest
import numpy as np

""" run from the command line with pytest Test_CoincidenceFunctions.py """
""" Checking gives expected results """


class TestClass:

    @classmethod
    def setup_class(self):
        # setup things
        self.C_F = CoincidenceFunctions.Coincidence_Functions()
        return

    @classmethod
    def teardown_class(self):
        # teardown things
        del self.C_F
        return

    def test_spot_to_mask(self):
        n_iter = 10
        coincidence = np.zeros(n_iter)
        chance_coincidence = np.zeros(n_iter)
        image_size = (10000, 10000)
        potential_indices = np.prod(image_size)
        spot_percentage = 0.001
        mask_percentage = 0.25
        for i in np.arange(n_iter):
            spot_indices = np.unique(
                np.random.randint(
                    low=0,
                    high=potential_indices,
                    size=int(potential_indices * spot_percentage),
                )
            )
            mask_indices = np.arange(int(potential_indices * mask_percentage))
            coincidence[i], chance_coincidence[i], _, _ = (
                self.C_F.calculate_coincidence(
                    spot_indices=spot_indices,
                    mask_indices=mask_indices,
                    image_size=image_size,
                    analysis_type="spot_to_mask",
                )
            )
        assert np.mean(coincidence) == pytest.approx(mask_percentage, abs=1e-3)
        assert np.mean(chance_coincidence) == pytest.approx(mask_percentage, abs=1e-3)
        coincidence = np.zeros(n_iter)
        chance_coincidence = np.zeros(n_iter)
        for i in np.arange(n_iter):
            spot_indices = np.unique(
                np.random.randint(
                    low=0,
                    high=int(potential_indices * mask_percentage),
                    size=int(potential_indices * spot_percentage),
                )
            )
            mask_indices = np.arange(int(potential_indices * mask_percentage))
            coincidence[i], chance_coincidence[i], _, _ = (
                self.C_F.calculate_coincidence(
                    spot_indices=spot_indices,
                    mask_indices=mask_indices,
                    image_size=image_size,
                    analysis_type="spot_to_mask",
                )
            )
        assert np.mean(coincidence) == pytest.approx(1, abs=1e-3)
        assert np.mean(chance_coincidence) == pytest.approx(mask_percentage, abs=1e-3)

    def test_spot_to_spot(self):
        n_iter = 1000
        coincidence_1 = np.zeros(n_iter)
        chance_coincidence_1 = np.zeros(n_iter)
        coincidence_2 = np.zeros(n_iter)
        chance_coincidence_2 = np.zeros(n_iter)
        image_size = (1000, 1000)
        potential_indices = np.prod(image_size)
        spot_percentage = 0.01
        for i in np.arange(n_iter):
            spot_indices = np.unique(
                np.random.randint(
                    low=0,
                    high=potential_indices,
                    size=int(potential_indices * spot_percentage),
                )
            )
            second_spot_indices = np.unique(
                np.random.randint(
                    low=0,
                    high=potential_indices,
                    size=int(potential_indices * spot_percentage),
                )
            )
            (
                coincidence_1[i],
                chance_coincidence_1[i],
                coincidence_2[i],
                chance_coincidence_2[i],
                _,
                _,
            ) = self.C_F.calculate_coincidence(
                spot_indices=spot_indices,
                mask_indices=None,
                image_size=image_size,
                second_spot_indices=second_spot_indices,
                blur_degree=0,
                analysis_type="spot_to_spot",
            )
        assert np.mean(coincidence_1) == pytest.approx(spot_percentage, abs=1e-3)
        assert np.mean(coincidence_2) == pytest.approx(spot_percentage, abs=1e-3)
        assert np.mean(chance_coincidence_1) == pytest.approx(spot_percentage, abs=1e-3)
        assert np.mean(chance_coincidence_2) == pytest.approx(spot_percentage, abs=1e-3)
        from copy import copy

        second_spot_indices = copy(spot_indices)
        (
            coincidence_1,
            chance_coincidence_1,
            coincidence_2,
            chance_coincidence_2,
            _,
            _,
        ) = self.C_F.calculate_coincidence(
            spot_indices=spot_indices,
            mask_indices=None,
            image_size=image_size,
            second_spot_indices=second_spot_indices,
            blur_degree=0,
            analysis_type="spot_to_spot",
        )
        assert coincidence_1 == pytest.approx(1)
        assert coincidence_2 == pytest.approx(1)

    def test_protein_load(self):
        n_iter = 1000
        olig_cell_ratio = np.zeros(n_iter)
        n_olig_in_cell = np.zeros(n_iter)
        image_size = (1000, 1000)
        potential_indices = np.prod(image_size)
        spot_percentage = 0.001
        mask_percentage = 0.25
        for i in np.arange(n_iter):
            spot_indices = np.unique(
                np.random.randint(
                    low=0,
                    high=potential_indices,
                    size=int(potential_indices * spot_percentage),
                )
            )
            mask_indices = np.arange(int(potential_indices * mask_percentage))
            spot_intensities = np.random.normal(loc=1, size=len(spot_indices))
            median = np.median(spot_intensities)
            olig_cell_ratio[i], n_olig_in_cell[i], _ = self.C_F.calculate_coincidence(
                spot_indices=spot_indices,
                mask_indices=mask_indices,
                image_size=image_size,
                spot_intensities=spot_intensities,
                median_intensity=median,
                analysis_type="protein_load",
                blur_degree=0,
            )
        assert np.mean(olig_cell_ratio) == pytest.approx(1, abs=1e-2)
        assert np.mean(n_olig_in_cell) == pytest.approx(250, abs=1)

    def test_spot_to_cell(self):
        n_iter = 1000
        olig_cell_ratio = np.zeros(n_iter)
        n_olig_in_cell = np.zeros(n_iter)
        image_size = (1000, 1000)
        potential_indices = np.prod(image_size)
        spot_percentage = 0.001
        mask_percentage = 0.25
        for i in np.arange(n_iter):
            spot_indices = np.unique(
                np.random.randint(
                    low=0,
                    high=potential_indices,
                    size=int(potential_indices * spot_percentage),
                )
            )
            mask_indices = np.arange(int(potential_indices * mask_percentage))
            olig_cell_ratio[i], n_olig_in_cell[i], _ = self.C_F.calculate_coincidence(
                spot_indices=spot_indices,
                mask_indices=mask_indices,
                image_size=image_size,
                analysis_type="spot_to_cell",
                blur_degree=0,
            )
        assert np.mean(olig_cell_ratio) == pytest.approx(1, abs=1e-2)
        assert np.mean(n_olig_in_cell) == pytest.approx(250, abs=1)

    def test_colocalisation_likelihood(self):
        n_iter = 100
        colocalisation_likelihood_ratio = np.zeros(n_iter)
        norm_CSR = np.zeros(n_iter)
        expected_spots_iter = np.zeros(n_iter)
        coincidence = np.zeros(n_iter)
        chance_coincidence = np.zeros(n_iter)
        image_size = (1000, 1000)
        potential_indices = np.prod(image_size)
        spot_percentage = 0.001
        mask_percentage = 0.25
        for i in np.arange(n_iter):
            spot_indices = np.unique(
                np.random.randint(
                    low=0,
                    high=potential_indices,
                    size=int(potential_indices * spot_percentage),
                )
            )
            mask_indices = np.arange(int(potential_indices * mask_percentage))
            (
                colocalisation_likelihood_ratio[i],
                _,
                norm_CSR[i],
                expected_spots_iter[i],
                coincidence[i],
                chance_coincidence[i],
                _,
                _,
            ) = self.C_F.calculate_coincidence(
                spot_indices=spot_indices,
                mask_indices=mask_indices,
                image_size=image_size,
                blur_degree=0,
                analysis_type="colocalisation_likelihood",
            )
        assert np.mean(colocalisation_likelihood_ratio) == pytest.approx(1, abs=1e-2)
        assert np.mean(norm_CSR) == pytest.approx(1, abs=1e-2)
        assert np.median(expected_spots_iter) == pytest.approx(250, abs=5)
        assert np.median(coincidence) == pytest.approx(mask_percentage, abs=1e-2)
        assert np.median(chance_coincidence) == pytest.approx(mask_percentage, abs=1e-2)
        colocalisation_likelihood_ratio = np.zeros(n_iter)
        coincidence = np.zeros(n_iter)
        chance_coincidence = np.zeros(n_iter)
        for i in np.arange(n_iter):
            spot_indices = np.unique(
                np.random.randint(
                    low=0,
                    high=int(potential_indices * mask_percentage),
                    size=int(potential_indices * spot_percentage),
                )
            )
            mask_indices = np.arange(int(potential_indices * mask_percentage))
            (
                colocalisation_likelihood_ratio[i],
                _,
                _,
                _,
                coincidence[i],
                chance_coincidence[i],
                _,
                _,
            ) = self.C_F.calculate_coincidence(
                spot_indices=spot_indices,
                mask_indices=mask_indices,
                image_size=image_size,
                blur_degree=0,
                analysis_type="colocalisation_likelihood",
            )
        assert np.median(colocalisation_likelihood_ratio) == pytest.approx(
            1 / mask_percentage
        )
        assert np.median(coincidence) == pytest.approx(1)
        assert np.median(chance_coincidence) == pytest.approx(mask_percentage, 1e-2)

    def test_largeobj(self):
        n_iter = 100
        coincidence = np.zeros(n_iter)
        image_size = (1000, 1000)
        potential_indices = np.prod(image_size)
        lo_perc = 0.0001
        n_largeobjects = int(potential_indices * lo_perc)
        mask_percentage = 0.25
        for i in np.arange(n_iter):
            lo_indices = self.C_F._apply_blur(
                np.random.randint(
                    low=0,
                    high=int(potential_indices * mask_percentage),
                    size=n_largeobjects,
                ),
                image_size,
                3,
            )
            lo_indices = lo_indices.reshape(
                n_largeobjects, int(len(lo_indices) / n_largeobjects)
            ).tolist()
            mask_indices = np.arange(int(potential_indices * mask_percentage))
            coincidence[i], _, _, _ = self.C_F.calculate_coincidence(
                spot_indices=None,
                largeobj_indices=lo_indices,
                n_largeobjs=n_largeobjects,
                mask_indices=mask_indices,
                image_size=image_size,
                analysis_type="largeobj",
            )
        assert np.median(coincidence) == pytest.approx(1)
        coincidence = np.zeros(n_iter)
        for i in np.arange(n_iter):
            lo_indices = self.C_F._apply_blur(
                np.random.randint(
                    low=0, high=int(potential_indices), size=n_largeobjects
                ),
                image_size,
                3,
            )
            lo_indices = lo_indices.reshape(
                n_largeobjects, int(len(lo_indices) / n_largeobjects)
            ).tolist()
            mask_indices = np.arange(int(potential_indices * mask_percentage))
            coincidence[i], _, _, _ = self.C_F.calculate_coincidence(
                spot_indices=None,
                largeobj_indices=lo_indices,
                n_largeobjs=n_largeobjects,
                mask_indices=mask_indices,
                image_size=image_size,
                analysis_type="largeobj",
            )
        assert np.mean(coincidence) == pytest.approx(0.25, abs=0.05)

    def test_maskfill(self):
        percentages = np.arange(0, 1, 0.01)
        image_size = (1000, 1000)
        potential_indices = np.prod(image_size)
        mask_fill = np.zeros_like(percentages)
        for i, p in enumerate(percentages):
            mask_indices = np.arange(int(potential_indices * p))
            mask_fill[i] = self.C_F.calculate_mask_fill(mask_indices, image_size)
            assert mask_fill[i] == pytest.approx(p)
