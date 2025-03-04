#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:59:53 2025

This code heavily based on code from
Hekrdla, M. et al. Optimized molecule detection in 
localization microscopy with selected false positive probability.
Nat Commun 16, 601 (2025).
"""
import sys
import os
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import square
from scipy.stats import norm


module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
import PSFFunctions

PSF = PSFFunctions.PSF_Functions()


class SpotDetection_Functions:
    def __init__(self):
        self = self
        return

    def detect_puncta_in_image(
        self,
        image: np.ndarray,
        psf_fun=None,
        variance: np.ndarray = None,
        pfa: float = 10**-4,
        wavelength: float = 0.6,
        pixel_size: float = 0.11,
        NA: float = 1.45,
        mf_factor: float = 3.0,
        local_factor: float = 3.0,
    ) -> np.ndarray:
        """detect_puncta_in_image: Returns spots from an image supplied

        Args:
            image (np.ndarray): image to analyse
            psf_fun (function): function of psf (if None, uses gauss2d)
            variance (np.ndarray): variance of camera used to record image
            pfa (float): probability of false alarm
            wavelength (float): average fluorescence wavelength
            pixel_size (float): pixel size in microns
            NA (float): numerical aperture of microscope
            multispot_marginfactor (float): multi spot margin factor
            mf_factor (float): match filter factor
            local_factor (float): local max factor

        Returns:
            detected_puncta (np.ndarray): xy coordinates of detected puncta"""
        if variance is not None:
            image_for_detection = np.divide(image, variance)
        else:
            image_for_detection = image
        if psf_fun is None:
            psf_fun = self.gauss2d
        sigma = np.divide(PSF.sigma_PSF(wavelength, NA), pixel_size)

        # one-sided range of matched filter kernel in pixels
        mf_range = int(np.ceil(mf_factor * sigma))
        guard_interval = int(np.ceil(mf_factor * sigma))
        reference_interval = int(np.ceil(mf_factor * sigma))
        local_max_range = int(np.ceil(local_factor * sigma))

        w = self.get_mf(psf_fun, sigma, mf_range)
        filtered_image = self.filter_image(image_for_detection, w)
        square_annulus = self.get_square_annulus(guard_interval, reference_interval)
        detected_puncta = self.get_detection_points(
            filtered_image, self.cacfar, pfa, local_max_range, kernel=square_annulus
        )
        return detected_puncta

    def get_mf(self, psf_fun, mf_sigma: float, mf_range: int) -> np.ndarray:
        """get_mf: Returns matched filter with PSF function given by parameter 'psf_fun'

        Args:
            psf_fun (function): point spread function model, e.g. 'gauss2d' or 'integrated_gauss2d'
            mf_sigma (float): standard deviation of the matched filter psf model
            mf_range (int): one-sided half size of the filter kernel

        Returns:
            mf (np.2darray): matched filter"""
        mf_size = 2 * mf_range + 1

        mf = self.get_single_spot(
            x0=mf_range, y0=mf_range, psf_fun=psf_fun, sigma=mf_sigma, a=1, size=mf_size
        )
        return mf

    def get_single_spot(
        self,
        x0: float,
        y0: float,
        psf_fun,
        sigma: float,
        a: float,
        size: int,
        sigma_range: int = 8,
    ) -> np.ndarray:
        """get_single_spot: Returns simulated 2D image with a single fluorescence molecule

        Args:
            x0 (float): x-coordinate of the center of the molecule
            y0 (float): y-coordinate of the center of the molecule
            psf_fun (float): point spread function model, e.g. 'gauss2d' or 'integrated_gauss2d'
            sigma (float): standard deviation of the psf model
            a (float): photon count
            size (int): one-sided size of the output array
            sigma_range (int): integer multiple of sigma where psf is considered as non-zero

        Returns:
            signal (np.2darray): 2d array of simulated signal"""

        x_min = int(max([round(x0 - sigma * sigma_range), 0]))
        x_max = int(min([round(x0 + sigma * sigma_range) + 1, size]))

        y_min = int(max([round(y0 - sigma * sigma_range), 0]))
        y_max = int(min([round(y0 + sigma * sigma_range) + 1, size]))

        signal = np.zeros((size, size))
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                signal[i, j] = psf_fun(i, j, x0, y0, sigma, a)

        return signal

    # TODO: check agrees with my 2D gaussian from PSFFunctions
    def gauss2d(
        self, x: float, y: float, x0: float, y0: float, sigma: float, a: float
    ) -> float:
        """gauss2d: Returns 2D gaussian value

        Args:
            x (float): x-coordinate of the gaussian
            y (float): y-coordinate of the gaussian
            x0 (float): x-coordinate of the center of the gaussian
            y0 (float): y-coordinate of the center of the gaussian
            sigma (float): standard deviation of the gaussian
            a (float): photon count

        Returns:
            signal (float): signal at particular (x, y) location"""
        return (
            a
            * 1
            / (2 * np.pi * sigma**2)
            * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
        )

    def filter_image(self, image: np.ndarray, w: np.ndarray) -> np.ndarray:
        """filter_image: Returns filtered image

        Args:
            image (np.ndarray): image to be filtered
            w (np.ndarray): filter kernel

        Returns:
            T (np.ndarray): filtered image"""
        T = convolve(image.astype("float"), w, mode="mirror")
        return T

    def get_square_annulus(
        self, guard_interval: int, reference_interval: int
    ) -> np.ndarray:
        """get_square_annulus: Returns square annulus kernel shape

        Args:
            guard_interval (int): range of internal hole
            reference_interval (int): width of non-zero band

        Returns:
            kernel (np.ndarray): square annulus"""
        kernel = square(2 * (guard_interval + reference_interval) + 1) - np.pad(
            square(2 * guard_interval + 1), pad_width=reference_interval
        )
        return kernel

    def isf_threshold(self, pfa: float, mu: float, sigma: float) -> float:
        """isf_threshold: Returns inverse survival function (ISF) threshold for a
            Gaussian distribution of filtered data

        Args:
            pfa (float): probability of false alarm
            mu (float): mean
            sigma (float): standard deviation

        Returns:
            isf (float)"""
        return norm.isf(pfa, loc=mu, scale=sigma)

    def cacfar_background_mean_estimate(
        self, r: np.ndarray, kernel: np.ndarray
    ) -> np.ndarray:
        """cacfar_background_mean_estimate: Returns local mean background level
           estimate given by arithmetic mean in neighborhood given by the kernel

        Args:
            r (np.ndarray): received 2D signal from which background mean is estimated
            kernel (np.ndarray): binary filter kernel describing local neighborhood within the
                                local mean is computed
        Returns:
            b_estimate (np.ndarray): returns local mean background level"""
        w = kernel / np.sum(kernel)
        b_estimate = convolve(r.astype("float"), w, mode="mirror")
        return b_estimate

    def cacfar_background_std_estimate(
        self, r: np.ndarray, b: np.ndarray, kernel: np.ndarray
    ) -> np.ndarray:
        """cacfar_background_std_estimate: Returns local standard deviation
           estimate given by arithmetic mean in neighborhood given by the kernel

        Args:
            r (np.ndarray): received 2D signal from which background std is estimated
            b (np.ndarray): background estimate image
            kernel (np.ndarray): binary filter kernel describing local neighborhood within the
                                local mean is computed
        Returns:
            b_std_estimate (np.ndarray): returns local mean std level"""
        b_std_estimate = np.sqrt(
            self.cacfar_background_mean_estimate((r - b) ** 2, kernel)
        )
        return b_std_estimate

    def cacfar_segmentation(
        self, T: np.ndarray, pfa: float, kernel: np.ndarray
    ) -> np.ndarray:
        """cacfar_segmentation: Returns binary mask segmentating pixels which are
           above cell-averaging constant false alarm rate (ca-cfar)
           isf threshold, where the mean and std estimates iare given by local
           arithmetic means

        Args:
            T (np.ndarray): input image
            pfa (float): probability of false alarm
            kernel (np.ndarray): binary filter kernel describing local neighborhood within the
            local mean and std is computed
        Returns:
            mask (np.ndarray): binary mask of pixels above false alarm"""
        b_estimate = self.cacfar_background_mean_estimate(T, kernel)
        b_std_estimate = self.cacfar_background_std_estimate(T, b_estimate, kernel)

        tau = self.isf_threshold(pfa, b_estimate, b_std_estimate)
        mask = T > tau
        return mask

    def cacfar(
        self, T: np.ndarray, pfa: float, local_max_range: int, kernel: np.ndarray
    ) -> np.ndarray:
        """cacfar: Returns binary mask segmentating pixels which are
           above cell-averaging constant false alarm rate (ca-cfar) isf
           threshold and which form local maximum, where the mean and std estimates
           are given by local arithmetic means

        Args:
            T (np.ndarray): input image
            pfa (float): probability of false alarm
            local_max_range (int): range over which local maximum is searched
            kernel (np.ndarray): binary filter kernel describing local neighborhood within the
                                local mean is computed
        Returns:
            mask (np.ndarray): binary mask above false alarm constant"""
        segmentation = self.cacfar_segmentation(T, pfa, kernel)
        mask = self.remove_nonlocal_maxima(segmentation, T, local_max_range)
        return mask

    def neigborhood(self, T: np.ndarray, point: np.ndarray, r: int) -> np.ndarray:
        """neigborhood: Return 2D sub-array centered around 'point' with range 'r'

        Args:
            T (np.ndarray) 2D image usually containing test statistic
            point (np.ndarray): x-y coordinate of the center
            r (int) radius of sub-array

        Returns:
            neigborhood (np.ndarray): neighborhood of pixels"""
        (x_max, y_max) = T.shape

        x0 = max([point[0] - r, 0])
        x1 = min([point[0] + r + 1, x_max])

        y0 = max([point[1] - r, 0])
        y1 = min([point[1] + r + 1, y_max])

        return T[x0:x1, y0:y1]

    def get_local_max_points(
        self, T: np.ndarray, points: np.ndarray, local_max_range: int
    ) -> np.ndarray:
        """get_local_max_points: Returns points of local maximum coordinates
            in 2D image 'T' selected from input points 'points'

        Args:
            T (np.ndarray) 2D image usually containing test statistic
            points (np.ndarray): list of x-y coordinate of the center
            local_max_range (int): radius of local maximum

        Returns:
            local_max_points (np.ndarray): maximum points"""

        local_max_points = np.array(
            [p for p in points if self.is_local_max(T, p, local_max_range)]
        )

        return local_max_points

    def is_local_max(self, T: np.ndarray, point: np.ndarray, r: int) -> bool:
        """is_local_max: Returns true if tested 'point' is a local maximum
           in the neighborhood of radius 'r'

        Args:
            T (np.ndarray): 2D image usually containing test statistic
            point (np.ndarray): x-y coordinate of the center
            r (int): radius of sub-array

        Returns:
            is_local_max (boolean): true if local maximum"""
        return np.max(self.neigborhood(T, point, r)) <= T[point[0], point[1]]

    def remove_nonlocal_maxima(
        self, segmentation: np.ndarray, T: np.ndarray, local_max_range: int
    ) -> np.ndarray:
        """remove_nonlocal_maxima: Returns segmentation masks containing only
           pixels that form local maxima of the radius given by 'local_max_range'

        Args:
            segmentation (np.ndarray): binary segmentation mask containing non-local maximum pixels
            T (np.ndarray): 2D image usually containing test statistic
            local_max_range (int) radius of local maximum

        Returns:
            mask (np.ndarray): segmentation mask with nonlocal maxima removed"""
        points = self.mask2points(segmentation)
        points_local_max = self.get_local_max_points(
            segmentation * T + norm.rvs(loc=0, scale=10**-9, size=T.shape),
            points,
            local_max_range,
        )
        size = T.shape[0]
        mask = self.points2mask(points_local_max, size)
        return mask

    def mask2points(self, mask: np.ndarray) -> np.ndarray:
        """mask2points: Converts binary segmentation mask to a list of pixel
           x-y coordinates

        Args:
            mask (np.ndarray): binary segmentation mask

        Returns:
            coords (np.ndarray): returns list of pixel x-y coordinates"""
        return np.array(np.where(mask), dtype="int32").T

    def points2mask(self, points: np.ndarray, size: int) -> np.ndarray:
        """points2mask: Converts list of pixels to a binary segmentation mask
           x-y coordinates

        Args:
            points (np.ndarray): list of pixels
            size (int): output shape

        Returns:
            mask (np.ndarray): list of pixels as a binary segmentation mask"""
        mask = np.zeros((size, size))
        for p in points:
            mask[p[0], p[1]] = 1
        return mask

    def get_detection_points(
        self,
        T: np.ndarray,
        detector_type,
        pfa: float,
        local_max_range: int,
        kernel: np.ndarray = None,
    ) -> np.ndarray:
        """get_detection_points: Return set x-y coordinates of detected puncta

        Args:
            T (np.ndarray): 2D image usually containing test statistic
            detector_type (function): function used in detection, e.g. cfar, cacfar, oscfar
            pfa (float): probability of false alarm
            local_max_range (int): radius of local maximum
            kernel (np.ndarray): binary filter kernel describing local neighborhood within the
                                local mean is computed
        Returns:
            points (np.ndarray): xy coordinates of detected puncta"""
        if kernel is None:
            mask = detector_type(T, pfa, local_max_range)
        else:
            mask = detector_type(T, pfa, local_max_range, kernel)
        points = self.mask2points(mask)
        return points
