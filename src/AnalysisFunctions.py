# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to analysis of images based on their 
radiality, relating to the RASP concept
jsb92, 2024/01/02
"""
import numpy as np
from scipy.signal import gaussian as gauss
from scipy.signal import fftconvolve
import skimage as ski
from skimage.filters import gaussian
from skimage.measure import label, regionprops_table
import skimage.draw as draw
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
import pandas as pd
import pathos
from pathos.pools import ThreadPool as Pool
cpu_number = int(pathos.helpers.cpu_count()*0.75)

class Analysis_Functions():
    def __init__(self):
        self = self
        return

    def calculate_gradient_field(self, image, kernel):
        """
        Calculate the gradient field of an image and compute focus-related measures.
    
        Args:
        - image (numpy.ndarray): The input image.
        - kernel (numpy.ndarray): The kernel for low-pass filtering.
    
        Returns:
        - filtered_image (numpy.ndarray): Image after low-pass filtering.
        - gradient_x (numpy.ndarray): X-gradient of the filtered image.
        - gradient_y (numpy.ndarray): Y-gradient of the filtered image.
        - focus_score (numpy.ndarray): Focus score of the image.
        - concentration_factor (numpy.ndarray): Concentration factor of the image.
        """
        # Initialize variables
        filtered_image = np.zeros_like(image)
        gradient_x = np.zeros_like(image)
        gradient_y = np.zeros_like(image)
    
        # Low-pass filtering using convolution
        if len(image.shape) > 2:
            for channel in np.arange(image.shape[2]):
                image_padded = np.pad(image[:, :, channel], (
                    (kernel.shape[0] // 2, kernel.shape[0] // 2),
                    (kernel.shape[1] // 2, kernel.shape[1] // 2)),
                                      mode='edge')
                filtered_image[:, :, channel] = fftconvolve(image_padded, kernel,
                                                               mode='valid')
        else:
            image_padded = np.pad(image, (
                (kernel.shape[0] // 2, kernel.shape[0] // 2),
                (kernel.shape[1] // 2, kernel.shape[1] // 2)),
                                  mode='edge')
            filtered_image[:, :] = fftconvolve(image_padded, kernel,
                                                           mode='valid')
        # Gradient calculation
        if len(image.shape) > 2:
            gradient_x[:, :-1, :] = np.diff(filtered_image, axis=1)  # x gradient (right to left)
            gradient_y[:-1, :, :] = np.diff(filtered_image, axis=0)  # y gradient (bottom to top)
        else:
            gradient_x[:, :-1] = np.diff(filtered_image, axis=1)  # x gradient (right to left)
            gradient_y[:-1, :] = np.diff(filtered_image, axis=0)  # y gradient (bottom to top)
           
        gradient_magnitude = np.sqrt(np.add(np.square(gradient_x), np.square(gradient_y)))
        sum_gradient = np.sum(gradient_magnitude, axis=(0, 1))
        concentration_factor = np.divide(sum_gradient, np.max(sum_gradient))
        focus_score = np.log(sum_gradient)
    
        return filtered_image, gradient_x, gradient_y, focus_score, concentration_factor

    def calculate_radiality(self, pil_small, img, gradient_x, gradient_y, d=2):
        """
        Calculate radiality measures based on pixel neighborhoods and gradients.

        Args:
        - pil_small (list): List of pixel indices.
        - img (numpy.ndarray): The input image.
        - gradient_x (numpy.ndarray): X-gradient of the image.
        - gradient_y (numpy.ndarray): Y-gradient of the image.
        - d (integer): pixel ring size
        Returns:
        - radiality (numpy.ndarray): Radiality measures.
        """
        xy = np.zeros([len(pil_small), 2])
        r0 = np.zeros(len(pil_small))
        for index in np.arange(len(pil_small)):
            pil_t = pil_small[index]
            r0[index], mi = np.max(img[pil_t[:,0], pil_t[:,1]]), np.argmax(img[pil_t[:,0], pil_t[:,1]])
            xy[index, :] = pil_t[mi]
            
        xy_default = np.asarray(np.unravel_index(np.unique(np.ravel_multi_index(np.asarray(draw.circle_perimeter(5,5,d)), img.shape, order='F')), img.shape, order='F')).T - 5
        x = np.asarray(np.tile(xy_default[:, 0], (len(pil_small), 1)).T + xy[:,0], dtype=int).T
        y = np.asarray(np.tile(xy_default[:, 1], (len(pil_small), 1)).T + xy[:,1], dtype=int).T
        
        
        g2 = np.sqrt(np.add(np.square(gradient_x[x, y]), np.square(gradient_y[x, y])))

        flatness = np.mean(np.divide(img[x, y].T, r0), axis=0)
        integrated_grad = np.sum(g2, axis=1)
        radiality = np.vstack([flatness, integrated_grad]).T

        return radiality
    
    def generate_mask_and_spot_indices(self, mask, centroids, image_size):
        """
        makes mask and spot indices from xy coordinates
    
        Args:
        - mask (2D array): boolean matrix
        - centroids (2D array): xy centroid coordinates
        - image_size (tuple): Image dimensions (height, width).
        
        Returns:
        mask_indices (1D array): indices of mask
        spot_indices (1D array): indices of spots
        """
        mask_coords = np.transpose((mask>0).nonzero())
        mask_indices = np.ravel_multi_index([mask_coords[:, 0], mask_coords[:, 1]], image_size)
        spot_indices = np.ravel_multi_index(centroids.T, image_size, order='F')
        return mask_indices, spot_indices
    
    def calculate_spot_colocalisation_likelihood_ratio(self, spot_indices, mask_indices, image_size, tol=0.01, n_iter=100):
        """
        gets spot colocalisation likelihood ratio, as well as reporting error
        bounds on the likelihood ratio for one image
    
        Args:
        - spot_indices (1D array): indices of spots
        - mask_indices (1D array): indices of pixels in mask
        - image_size (tuple): Image dimensions (height, width).
        - tol (float): default 0.01; tolerance for convergence
        - n_iter (int): default 100; number of iterations to start with
        
        Returns:
        colocalisation_likelihood_ratio (float): likelihood ratio of spots for mask
        perc_std (float): standard deviation on this CLR based on bootstrapping
        meanCSR (float): mean of randomised spot data
        expected_spots (float): number of spots we expect based on mask % of image
        n_iter (int): how many iterations it took to converge
        """
        n_iter_rec = n_iter
        possible_indices = np.arange(0, np.prod(image_size)) # get list of where is possible to exist in an image
        n_spots = len(spot_indices) # get number of spots
        mask_fill = self.calculate_mask_fill(mask_indices, image_size) # get mask_fill
        expected_spots = np.multiply(mask_fill, n_spots) # get expected number of spots
        if np.isclose(expected_spots, 0., atol=1e-4):
            n_iter_rec = 0
            colocalisation_likelihood_ratio = np.NAN
            norm_CSR = np.NAN
            norm_std = np.NAN
            return colocalisation_likelihood_ratio, norm_std, norm_CSR, expected_spots, n_iter_rec
        else:
            nspots_in_mask = self.test_spot_mask_overlap(spot_indices, mask_indices) # get nspots in mask
            colocalisation_likelihood_ratio = np.divide(nspots_in_mask, expected_spots) # generate colocalisation likelihood ratio
            
            random_spot_locations = np.random.choice(possible_indices, size=(n_iter, n_spots)) # get random spot locations
            CSR = np.zeros([n_iter]) # generate CSR array to fill in
            
            for i in np.arange(n_iter):
                CSR[i] = self.test_spot_mask_overlap(random_spot_locations[i, :], mask_indices)
            
            meanCSR = np.divide(np.nanmean(CSR), expected_spots) # should be close to 1
            CSR_diff = np.abs(meanCSR - 1.)
            while CSR_diff > tol: # do n_iter more tests iteratively until convergence
                n_iter_rec = n_iter_rec + n_iter # add n_iter iterations
                CSR_new = np.zeros([n_iter])
                random_spot_locations = np.random.choice(possible_indices, size=(n_iter, n_spots)) # get random spot locations
                for i in np.arange(n_iter):
                    CSR_new[i] = self.test_spot_mask_overlap(random_spot_locations[i, :], mask_indices)
                CSR = np.hstack([CSR, CSR_new]) # stack
                meanCSR = np.divide(np.nanmean(CSR), expected_spots) # should be close to 1
                CSR_diff = np.abs(meanCSR - 1.)
            if (expected_spots > 0) and (np.mean(CSR) > 0):    
                norm_CSR = np.divide(np.nanmean(CSR), expected_spots) # should be close to 1
                norm_std = np.divide(np.nanstd(CSR), np.nanmean(CSR)) # std dev (normalised)
            else:
                norm_CSR = np.NAN
                norm_std = np.NAN
            return colocalisation_likelihood_ratio, norm_std, norm_CSR, expected_spots, n_iter_rec
    
    def default_spotanalysis_routine(self, image, k1, k2, thres=0.05, 
                                     large_thres=450., areathres=30.,
                                     rdl=[50., 0., 0.], d=2):
        """
        Daisy-chains analyses to get
        basic image properties (centroids, radiality)
        from a single image
    
        Args:
        - image (array). image as numpy array
        - k1 (array). gaussian blur kernel
        - k2 (array). ricker wavelet kernel
        - thres (float). percentage threshold
        - areathres (float). area threshold
        - rdl (array). radiality thresholds
        
        Returns:
        - centroids (2D array): centroid positions per oligomer
        - estimated_intensity (numpy.ndarray): Estimated sum intensity per oligomer.
        - estimated_background (numpy.ndarray): Estimated mean background per oligomer.

        """
        large_mask = self.detect_large_features(image, large_thres)
        img2, Gx, Gy, focusScore, cfactor = self.calculate_gradient_field(image, k1)
        dl_mask, centroids, radiality, idxs = self.small_feature_kernel(image, 
        large_mask, img2, Gx, Gy,
        k2, thres, areathres, rdl, d)
        estimated_intensity, estimated_background = self.estimate_intensity(image, centroids)
        to_keep = ~np.isnan(estimated_intensity)
        estimated_intensity = estimated_intensity[to_keep]
        estimated_background = estimated_background[to_keep]
        centroids = centroids[to_keep, :]    
        return centroids, estimated_intensity, estimated_background
    
    def calculate_mask_fill(self, mask_indices, image_size):
        """
        calculate amount of image filled by mask.
    
        Args:
        - mask_indices (1D array): indices of pixels in mask
        - image_size (tuple): Image dimensions (height, width).
    
        Returns:
        - mask_fill (float): proportion of image filled by mask.
        """

        mask_fill = np.divide(len(mask_indices), np.prod(image_size))
        return mask_fill
    
    def test_spot_mask_overlap(self, spot_indices, mask_indices):
        """
        Tests which spots overlap with a given mask.
    
        Args:
        - spot_indices (1D array): indices of spots
        - mask_indices (1D array): indices of pixels in mask
    
        Returns:
        - n_spots_in_mask (float): number of spots that overlap with the mask.
        """

        n_spots_in_mask = np.sum(np.isin(mask_indices, spot_indices))
        return n_spots_in_mask

    
    def dilate_pixel(self, index, image_size, width=5, edge=1):
        """
        Dilate a pixel index to form a neighborhood.
    
        Args:
        - index (int): Pixel index.
        - image_size (tuple): Image dimensions (height, width).
        - width: width of dilation (default 5)
        - edge: edge of dilation (default 1)
    
        Returns:
        - dilated_indices (numpy.ndarray): Dilated pixel indices forming a neighborhood.
        """
        x,y = np.where(ski.morphology.octagon(width, edge))
        x = x - int(ski.morphology.octagon(width, edge).shape[0]/2)
        y = y - int(ski.morphology.octagon(width, edge).shape[1]/2)
        centroid = np.asarray(np.unravel_index(index, image_size, order='F'), dtype=int)
        x = x + int(centroid[0])
        y = y + int(centroid[1])
                
        dilated_indices = np.ravel_multi_index(np.vstack([x, y]), image_size, order='F', mode='wrap')
        return dilated_indices
    
    def create_kernel(self, background_sigma, wavelet_sigma):
        """
        Create Gaussian and Ricker wavelet kernels.

        Args:
        - background_sigma (float): Standard deviation for Gaussian kernel.
        - wavelet_sigma (float): Standard deviation for Ricker wavelet.

        Returns:
        - gaussian_kernel (numpy.ndarray): Gaussian kernel for background suppression.
        - ricker_kernel (numpy.ndarray): Ricker wavelet for feature enhancement.
        """
        gaussian_kernel = self.create_gaussian_kernel((background_sigma, background_sigma), 
                                                (2 * int(np.ceil(2 * background_sigma)) + 1, 
                                                 2 * int(np.ceil(2 * background_sigma)) + 1))
        ricker_kernel = self.ricker_wavelet(wavelet_sigma)
        return gaussian_kernel, ricker_kernel

    def create_gaussian_kernel(self, sigmas, size):
        """
        Create a 2D Gaussian kernel.

        Args:
        - sigmas (tuple): Standard deviations in X and Y directions.
        - size (tuple): Size of the kernel.

        Returns:
        - kernel (numpy.ndarray): 2D Gaussian kernel.
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
        - sigma (float): Standard deviation for the wavelet.

        Returns:
        - wavelet (numpy.ndarray): 2D Ricker wavelet.
        """
        amplitude = np.divide(2., np.multiply(np.sqrt(np.multiply(3., sigma)), np.power(np.pi, 0.25)))
        length = int(np.ceil(np.multiply(4, sigma)))
        x = np.arange(-length, length + 1)
        y = np.arange(-length, length + 1)
        X, Y = np.meshgrid(x, y)
        sigma_sq = np.square(sigma)
        common_term = np.add(np.divide(np.square(X), np.multiply(2., sigma_sq)), np.divide(np.square(Y), np.multiply(2., sigma_sq)))
        wavelet = np.multiply(amplitude, np.multiply(np.subtract(1, common_term), np.exp(-common_term)))
        return wavelet
    
    def create_filled_region(self, image_size, indices_to_keep):
        """
        Fill a region in a boolean matrix based on specified indices.

        Args:
        - image_size (tuple): Size of the boolean matrix.
        - indices_to_keep (list): List of indices to set as True.

        Returns:
        - boolean_matrix (numpy.ndarray): Boolean matrix with specified indices set to True.
        """
        # Concatenate all indices to keep into a single array
        indices_to_keep = np.concatenate(indices_to_keep)

        # Create a boolean matrix of size image_size
        boolean_matrix = np.zeros(image_size, dtype=bool)

        # Set the elements at indices specified in indices_to_keep to True
        # Utilize tuple unpacking for efficient indexing and assignment
        boolean_matrix[tuple(indices_to_keep.T)] = True

        return boolean_matrix

    def infocus_indices(self, focus_scores, threshold_differential):
        """
        Identify in-focus indices based on focus scores and a threshold differential.
    
        Args:
        - focus_scores (numpy.ndarray): Focus scores for different slices.
        - threshold_differential (float): Threshold for differential focus scores.
    
        Returns:
        - in_focus_indices (list): List containing the first and last in-focus indices.
        """
        # Calculate the Euclidean distance between each slice in focus_scores
        focus_score_diff = np.diff(focus_scores)
        
        # Mask distances less than or equal to 0 as NaN
        focus_score_diff[focus_score_diff <= 0] = np.nan
        
        # Perform DBSCAN from the start
        dist1 = np.concatenate(([0], focus_score_diff > threshold_differential))  # Mark as True if distance exceeds threshold
    
        # Calculate the Euclidean distance from the end
        focus_score_diff_end = np.diff(np.flip(focus_scores))
        
        # Perform DBSCAN from the end
        dist2 = np.concatenate(([0], focus_score_diff_end < threshold_differential))  # Mark as True if distance is below threshold
    
        # Refine the DBSCAN results
        dist1 = np.diff(dist1)
        dist2 = np.diff(dist2)
        
        # Find indices indicating the transition from out-of-focus to in-focus and vice versa
        transition_to_in_focus = np.where(dist1 == -1)[0]
        transition_to_out_focus = np.where(dist2 == 1)[0]
    
        # Determine the first and last slices in focus
        first_in_focus = 0 if len(transition_to_in_focus) == 0 else transition_to_in_focus[0]  # First slice in focus
        last_in_focus = len(focus_scores) if len(transition_to_out_focus) == 0 else len(focus_scores) - transition_to_out_focus[-1] + 1  # Last slice in focus
        
        # Ensure consistency and handle cases where the first in-focus slice comes after the last
        first_in_focus = first_in_focus if first_in_focus <= last_in_focus else 1
        
        # Return indices for in-focus images
        in_focus_indices = [first_in_focus, last_in_focus-1]
        return in_focus_indices
    
    def estimate_intensity(self, image, centroids):
        """
        Estimate intensity values for each centroid in the image.
    
        Args:
        - image (numpy.ndarray): Input image.
        - centroids (numpy.ndarray): Centroid locations.
    
        Returns:
        - estimated_intensity (numpy.ndarray): Estimated sum intensity per oligomer.
        - estimated_background (numpy.ndarray): Estimated mean background per oligomer.
        """
        centroids = np.asarray(centroids, dtype=int)
        image_size = image.shape
        indices = np.ravel_multi_index(centroids.T, image_size, order='F')
        estimated_intensity = np.zeros(len(indices), dtype=float)  # Estimated sum intensity per oligomer
        
        x_in, y_in, x_out, y_out = self.intensity_pixel_indices(centroids, image_size)
        
        estimated_background = np.mean(image[y_out, x_out], axis=0)
        estimated_intensity = np.sum(np.subtract(image[y_in, x_in], estimated_background), axis=0)
        
        estimated_intensity[estimated_intensity < 0] = np.NAN
        estimated_background[estimated_background < 0] = np.NAN
       
        return estimated_intensity, estimated_background
    
    def intensity_pixel_indices(self, centroid_loc, image_size):
        """
        Calculate pixel indices for inner and outer regions around the given index.
    
        Args:
        - centroid_loc (2D array): xy location of the pixel.
        - image_size (tuple): Size of the image.
    
        Returns:
        - inner_indices (numpy.ndarray): Pixel indices for the inner region.
        - outer_indices (numpy.ndarray): Pixel indices for the outer region.
        """
        
        small_oct = ski.morphology.octagon(2, 4)
        outer_ind = ski.morphology.octagon(2, 5)
        inner_ind = np.zeros_like(outer_ind)
        inner_ind[1:-1, 1:-1] = small_oct
        outer_ind = outer_ind-inner_ind
        x_inner, y_inner = np.where(inner_ind)
        x_inner = x_inner-int(inner_ind.shape[0]/2)
        y_inner = y_inner-int(inner_ind.shape[1]/2)
       
        x_outer, y_outer = np.where(outer_ind)
        x_outer = x_outer-int(outer_ind.shape[0]/2)
        y_outer = y_outer-int(outer_ind.shape[1]/2)
                
        x_inner = np.tile(x_inner, (len(centroid_loc[:, 0]),1)).T + centroid_loc[:, 0]
        y_inner = np.tile(y_inner, (len(centroid_loc[:, 1]),1)).T + centroid_loc[:, 1]
        x_outer = np.tile(x_outer, (len(centroid_loc[:, 0]),1)).T + centroid_loc[:, 0]
        y_outer = np.tile(y_outer, (len(centroid_loc[:, 1]),1)).T + centroid_loc[:, 1]
        
        return x_inner, y_inner, x_outer, y_outer
    
    def detect_large_features(self, image, threshold1, threshold2=0, sigma1=2., sigma2=60.):
        """
        Detects large features in an image based on a given threshold.
    
        Args:
        - image (numpy.ndarray): Original image.
        - threshold1 (float): Threshold for determining features. Only this is
        used for the determination of large protein aggregates.
        - threshold2 (float): Threshold for determining cell features. If above
        0, gets used and cellular features are detected.
        - sigma1 (float): first gaussian blur width
        - sigma2 (float): second gaussian blur width
    
        Returns:
        - large_mask (numpy.ndarray): Binary mask for the large features.
        """
        # Apply Gaussian filters with different sigmas and subtract to enhance features
        enhanced_image = gaussian(image, sigma=sigma1, truncate=2.) - gaussian(image, sigma=sigma2, truncate=2.)
    
        # Create a binary mask for large features based on the threshold
        large_mask = enhanced_image > threshold1
        
        if threshold2 > 0:
            large_mask = binary_opening(large_mask, structure=ski.morphology.disk(1))
            large_mask = binary_closing(large_mask, structure=ski.morphology.disk(5))
            pixel_index_list, *rest = self.calculate_region_properties(large_mask)
            idx1 = np.zeros_like(pixel_index_list, dtype=bool)
            imcopy = np.copy(image)
            
            for i in np.arange(len(pixel_index_list)):
                idx1[i] = 1*(np.sum(imcopy[pixel_index_list[i][:,0], 
                        pixel_index_list[i][:,1]] > 
                        threshold2)/len(pixel_index_list[i][:,0]) > 0.1)
            
            if len(idx1) > 0:
                large_mask = self.create_filled_region(image.shape, pixel_index_list[idx1])
                large_mask = binary_fill_holes(large_mask)

        return large_mask.astype(bool)


    def calculate_region_properties(self, binary_mask):
        """
        Calculate properties for labeled regions in a binary mask.
    
        Args:
        - binary_mask (numpy.ndarray): Binary mask of connected components.
    
        Returns:
        - pixel_index_list (list): List containing pixel indices for each labeled object.
        - areas (numpy.ndarray): Array containing areas of each labeled object.
        - centroids (numpy.ndarray): Array containing centroids (x, y) of each labeled object.
        """
        # Find connected components and count the number of objects
        labeled_image, num_objects = label(binary_mask,
                                        connectivity=2, return_num=True)
        # Initialize arrays for storing properties
        centroids = np.zeros((num_objects, 2))
        
        # Get region properties
        props = regionprops_table(labeled_image, properties=('centroid',
                                                 'area', 'coords'))
        centroids[:, 0] = props['centroid-1']
        centroids[:, 1] = props['centroid-0']
        areas = props['area']
        pixel_index_list = props['coords']
        return pixel_index_list, areas, centroids

    def small_feature_kernel(self, img, large_mask, img2, Gx, Gy, k2, thres, area_thres, rdl, d=2):
        """
        Find small features in an image and determine diffraction-limited (dl) and non-diffraction-limited (ndl) features.
    
        Args:
        - img (numpy.ndarray): Original image.
        - large_mask (numpy.ndarray): Binary mask for large features.
        - img2 (numpy.ndarray): Smoothed image for background suppression.
        - Gx (numpy.ndarray): Gradient image in x-direction.
        - Gy (numpy.ndarray): Gradient image in y-direction.
        - k2 (numpy.ndarray): The kernel for blob feature enhancement.
        - thres (float): Converting real-valued image into a binary mask.
        - area_thres (float): The maximum area in pixels a diffraction-limited object can be.
        - rdl (list): Radiality threshold [min_radiality, max_radiality, area].
        - d (integer): pixel radius
    
        Returns:
        - dl_mask (numpy.ndarray): Binary mask for diffraction-limited (dl) features.
        - centroids (numpy.ndarray): Centroids for dl features.
        - radiality (numpy.ndarray): Radiality value for all features (before the filtering based on the radiality).
        - idxs (numpy.ndarray): Indices for objects that satisfy the decision boundary.
        """
        img1 = np.maximum(np.subtract(img, img2), 0)
        pad_size = np.subtract(np.asarray(k2.shape), 1) // 2
        img1 = fftconvolve(np.pad(img1, pad_size, mode='edge'), k2, mode='valid')
    
        BW = np.zeros_like(img1, dtype=bool)
        if thres < 1:
            thres = np.percentile(img1.ravel(), 100*(1-thres))
        BW[img1 > thres] = 1
        BW = np.logical_or(BW, large_mask)
        BW = binary_opening(BW, structure=ski.morphology.disk(1))
        
        imsz = img.shape
        pixel_idx_list, areas, centroids = self.calculate_region_properties(BW)
    
        idxb = np.logical_and(centroids[:, 0] > 10, centroids[:, 0] < imsz[1] - 10)
        idxb = np.logical_and(idxb, np.logical_and(centroids[:, 1] > 10, centroids[:, 1] < imsz[0] - 9))
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

    def compute_spot_and_cell_props(self, image, image_cell, k1, k2, prot_thres=0.05, large_prot_thres=450., 
                           areathres=30., rdl=[50., 0., 0.], z=[0], 
                           cell_threshold1=200., cell_threshold2=200, 
                           cell_sigma1=2., cell_sigma2=40., d=2):
        """
        Gets basic image properties (centroids, radiality)
        from a single image and compare to a cell mask from another image channel
    
        Args:
        - image (array). image of protein stain as numpy array
        - image (array). image of cell stain as numpy array
        - k1 (array). gaussian blur kernel
        - k2 (array). ricker wavelet kernel
        - prot_thres (float). percentage threshold for protein
        - large_prot_thres (float). Protein threshold intensity
        - areathres (float). area threshold
        - rdl (array). radiality thresholds
        - z (array). z planes to image, default 0
        - cell_threshold1 (float). 1st cell intensity threshold
        - cell_threshold2 (float). 2nd cell intensity threshold
        - cell_sigma1 (float). cell blur value 1
        - cell_sigma2 (float). cell blur value 2
        - d (integer). pixel radius value
        """
        
        columns = ['x', 'y', 'z', 'sum_intensity_in_photons', 'bg', 'zi', 'zf']
        columns_cell = ['colocalisation_likelihood_ratio', 'std', 
                            'CSR', 'expected_spots', 'n_iterations', 'z']

        if len(z) > 1:
            z_planes = np.arange(z[0], z[1])
            cell_mask = np.zeros_like(image_cell, dtype=bool)
            clr, norm_std, norm_CSR, expected_spots, n_iter = self.gen_CSRmats(image_cell.shape[2])
            
            centroids = {} # do this so can parallelise
            estimated_intensity = {}
            estimated_background = {}
            def analyse_zplanes(zp):
                img_z = image[:, :, zp]
                img_cell_z = image_cell[:, :, zp]
                centroids[zp], estimated_intensity[zp], estimated_background[zp] = self.default_spotanalysis_routine(img_z,
                k1, k2, prot_thres, large_prot_thres, areathres, rdl, d)
                
                cell_mask[:,:,zp] = self.detect_large_features(img_cell_z,
                    cell_threshold1, cell_threshold2, cell_sigma1, cell_sigma2)
                
                image_size = img_z.shape
                mask_indices, spot_indices = self.generate_mask_and_spot_indices(cell_mask[:,:,zp], centroids, image_size)

                clr[zp], norm_std[zp], norm_CSR[zp], expected_spots[zp], n_iter[zp] = self.calculate_spot_colocalisation_likelihood_ratio(spot_indices, mask_indices, image_size)
                
            pool = Pool(nodes=cpu_number); pool.restart()
            pool.map(analyse_zplanes, z_planes)
            pool.close(); pool.terminate()
            
            to_save = self.make_datarray_spot(centroids, 
            estimated_intensity, estimated_background, columns, z_planes)
            to_save = to_save.reset_index(drop=True)
            
            to_save_cell = self.make_datarray_cell(clr, norm_std,
                        norm_CSR, expected_spots, n_iter, columns_cell, z_planes)
        else:
            centroids, estimated_intensity, estimated_background = self.default_spotanalysis_routine(image,
            k1, k2, prot_thres, large_prot_thres, areathres, rdl, d)
            
            cell_mask = self.detect_large_features(image_cell,
                    cell_threshold1, cell_threshold2, cell_sigma1, cell_sigma2)
            
            image_size = image.shape
            mask_indices, spot_indices = self.generate_mask_and_spot_indices(cell_mask, centroids, image_size)

            clr, norm_std, norm_CSR, expected_spots, n_iter = self.calculate_spot_colocalisation_likelihood_ratio(spot_indices, mask_indices, image_size)

            to_save = self.make_datarray_spot(centroids, 
                estimated_intensity, estimated_background, columns[:-2])
            
            to_save_cell = self.make_datarray_cell(clr, norm_std, norm_CSR, 
                                expected_spots, n_iter, columns_cell[:-1])
        return to_save, to_save_cell, cell_mask 

    def compute_spot_props(self, image, k1, k2, thres=0.05, large_thres=450., 
                           areathres=30., rdl=[50., 0., 0.], z=[0], d=2):
        """
        Gets basic image properties (centroids, radiality)
        from a single image
    
        Args:
        - image (array). image as numpy array
        - k1 (array). gaussian blur kernel
        - k2 (array). ricker wavelet kernel
        - thres (float). percentage threshold
        - areathres (float). area threshold
        - rdl (array). radiality thresholds
        - z (array). z planes to image, default 0
        - d (int). Pixel radius value
        """
        
        columns = ['x', 'y', 'z', 'sum_intensity_in_photons', 'bg', 'zi', 'zf']
        if len(z) > 1:
            z_planes = np.arange(z[0], z[1])
            centroids = {}
            estimated_intensity = {}
            estimated_background = {}
            def analyse_zplanes(zp):
                centroids[zp], estimated_intensity[zp], estimated_background[zp] = self.default_spotanalysis_routine(image[:, :, zp],
                k1, k2, thres, large_thres, areathres, rdl, d)
                
            pool = Pool(nodes=cpu_number); pool.restart()
            pool.map(analyse_zplanes, z_planes)
            pool.close(); pool.terminate()
                
            to_save = self.make_datarray_spot(centroids, 
            estimated_intensity, estimated_background, columns, z_planes)
            to_save = to_save.reset_index(drop=True)
        else:
            centroids, estimated_intensity, estimated_background = self.default_spotanalysis_routine(image,
            k1, k2, thres, large_thres, areathres, rdl, d)
            
            to_save = self.make_datarray_spot(centroids, 
                estimated_intensity, estimated_background, columns[:-2])
            
        return to_save
    
    def compute_image_props(self, image, k1, k2, thres=0.05, large_thres=450., areathres=30., rdl=[50., 0., 0.], d=2, z_planes=[0], calib=False):
        """
        Gets basic image properties (dl_mask, centroids, radiality)
        from a single image
    
        Args:
        - image (array). image as numpy array
        - k1 (array). gaussian blur kernel
        - k2 (array). ricker wavelet kernel
        - thres (float). percentage threshold
        - areathres (float). area threshold
        - rdl (array). radiality thresholds
        - d (int). radiality ring
        - z_planes (array). If multiple z planes, give z planes
        - calib (bool). If True, for radiality calibration
   
        """
        if len(z_planes) > 1:
            radiality = {}
            centroids = {}
            dl_mask = np.zeros_like(image)
            def run_over_z(z):
                if calib == True:
                    large_mask = np.full_like(image[:,:,z], False)
                else:
                    large_mask = self.detect_large_features(image[:,:,z], large_thres)
                img2, Gx, Gy, focusScore, cfactor = self.calculate_gradient_field(image[:,:,z], k1)
                dl_mask[:,:,z], centroids[z], radiality[z], idxs = self.small_feature_kernel(image[:,:,z], 
                large_mask, img2, Gx, Gy,
                k2, thres, areathres, rdl, d)
                
            pool = Pool(nodes=cpu_number); pool.restart()
            pool.map(run_over_z, z_planes)
            pool.close(); pool.terminate()
            
        else:
            if calib == True:
                large_mask = np.full_like(image, False)
            else:
                large_mask = self.detect_large_features(image, large_thres)
            img2, Gx, Gy, focusScore, cfactor = self.calculate_gradient_field(image, k1)
            dl_mask, centroids, radiality, idxs = self.small_feature_kernel(image, 
            large_mask, img2, Gx, Gy,
            k2, thres, areathres, rdl, d)
        
        return dl_mask, centroids, radiality

    def gen_CSRmats(self, image_z_shape):
        """
        Generates empty matrices for the CSR
    
        Args:
        - image_z_shape (int). shape of new array

        Returns:
        - clr (ndarray). empty array
        - norm_std (ndarray). empty array
        - norm_CSR (ndarray). empty array
        - expected_spots (ndarray). empty array
        - n_iter (ndarray). empty array
        
        """

        clr = np.zeros(image_z_shape)
        norm_std = np.zeros(image_z_shape)
        norm_CSR = np.zeros(image_z_shape)
        expected_spots = np.zeros(image_z_shape)
        n_iter = np.zeros(image_z_shape)
        return clr, norm_std, norm_CSR, expected_spots, n_iter
    
    def make_datarray_spot(self, centroids, estimated_intensity, estimated_background, columns, z_planes=0):
        """
        makes a datarray in pandas for spot information
    
        Args:
        - centroids (ndarray): centroid positions
        - estimated_intensity (ndarray): estimated intensities
        - estimated_background (ndarray): estimated backgrounds
        - columns (list of strings): column labels
        - z_planes: z_planes to put in array (if needed); if int, assumes only
        one z-plane
        
        Returns:
        - to_save (pandas DataArray) pandas array to save
        
        """
        if isinstance(z_planes, int):
            dataarray = np.vstack([centroids[:, 0], centroids[:, 1], 
            np.full_like(centroids[:, 0], 1), estimated_intensity, 
            estimated_background])
        else:
            for z in z_planes:
                if z == z_planes[0]:
                    dataarray = np.vstack([centroids[z][:, 0], centroids[z][:, 1], 
                    np.full_like(centroids[z][:, 0], z+1), estimated_intensity[z], 
                    estimated_background[z], np.full_like(centroids[z][:, 0], 1+z_planes[0]),
                    np.full_like(centroids[z][:, 0], 1+z_planes[-1])]) 
                else:
                    da = np.vstack([centroids[z][:, 0], centroids[z][:, 1], 
                    np.full_like(centroids[z][:, 0], z+1), estimated_intensity[z], 
                    estimated_background[z], np.full_like(centroids[z][:, 0], 1+z_planes[0]),
                    np.full_like(centroids[z][:, 0], 1+z_planes[-1])]) 
                    dataarray = np.hstack([dataarray, da])
        return pd.DataFrame(data=dataarray.T, columns=columns)
    
    def make_datarray_cell(self, clr, norm_std, norm_CSR, expected_spots, n_iter, columns, z_planes='none'):
        """
        makes a datarray in pandas for cell information
    
        Args:
        - clr (ndarray): colocalisation likelihood ratios
        - estimated_intensity (ndarray): estimated intensities
        - estimated_background (ndarray): estimated backgrounds
        - columns (list of strings): column labels
        - zp (string or int): if int, gives out z-plane version of datarray
        - z_planes: z_planes to put in array (if needed)
        
        Returns:
        - to_save (pandas DataArray) pandas array to save
        
        """
        if isinstance(z_planes, str):
            dataarray_cell = np.vstack([clr, norm_std, norm_CSR, 
                            expected_spots, n_iter])
        else:
            zps = np.zeros_like(clr)
            zps[z_planes] = z_planes+1
            dataarray_cell = np.vstack([clr, norm_std, norm_CSR, 
                            expected_spots, n_iter, zps])
        return pd.DataFrame(data=dataarray_cell.T, columns=columns)