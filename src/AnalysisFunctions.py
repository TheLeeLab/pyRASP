# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to analysis of images based on their 
radiality, relating to the RASP concept
jsb92, 2024/01/02
"""
import numpy as np
from scipy.signal import gaussian as gauss
from scipy.signal import convolve2d
import skimage as ski
from skimage.filters import gaussian
from skimage.measure import label, regionprops_table
from scipy.ndimage.morphology import binary_opening

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
            for channel in range(image.shape[2]):
                image_padded = np.pad(image[:, :, channel], (
                    (kernel.shape[0] // 2, kernel.shape[0] // 2),
                    (kernel.shape[1] // 2, kernel.shape[1] // 2)),
                                      mode='edge')
                filtered_image[:, :, channel] = convolve2d(image_padded, kernel,
                                                               mode='valid')
        else:
            image_padded = np.pad(image, (
                (kernel.shape[0] // 2, kernel.shape[0] // 2),
                (kernel.shape[1] // 2, kernel.shape[1] // 2)),
                                  mode='edge')
            filtered_image[:, :] = convolve2d(image_padded, kernel,
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

    def calculate_radiality(self, pil_small, img, gradient_x, gradient_y):
        """
        Calculate radiality measures based on pixel neighborhoods and gradients.

        Args:
        - pil_small (list): List of pixel indices.
        - img (numpy.ndarray): The input image.
        - gradient_x (numpy.ndarray): X-gradient of the image.
        - gradient_y (numpy.ndarray): Y-gradient of the image.

        Returns:
        - radiality (numpy.ndarray): Radiality measures.
        """
        radiality = np.zeros((len(pil_small), 2))

        # Function to generate indices of pixels at a specific distance from the center
        def radiality_pixel_indices(xy, d=2):
            x = xy[0]
            y = xy[1]
            add_array_1 = np.arange(-d+1, d)
            add_array_2 = np.full(len(add_array_1), d)
            add_array_3 = np.full(len(add_array_1), -d)
            add_array_x = np.hstack([add_array_1, add_array_2, add_array_3, add_array_1])
            add_array_y = np.hstack([add_array_2, add_array_1, add_array_1, add_array_3])
            
            x2 = add_array_x+x
            y2 = add_array_y+y
            xy_r2 = np.vstack([x2, y2]).T
            return xy_r2

        for k, pil_t in enumerate(pil_small):
            r0, mi = np.max(img[pil_t[:,0], pil_t[:,1]]), np.argmax(img[pil_t[:,0], pil_t[:,1]])
            xy = pil_t[mi]
            xy_r2 = radiality_pixel_indices(xy)
            
            g2 = np.sqrt(np.add(np.square(gradient_x[xy_r2[:,0], xy_r2[:,1]]), np.square(gradient_y[xy_r2[:,0], xy_r2[:,1]])))

            steepness = np.mean(np.divide(img[xy_r2[:,0], xy_r2[:,1]], r0))
            integrated_grad = np.sum(g2)
            radiality[k, :] = [steepness, integrated_grad]

        return radiality
    
    def default_spotanalysis_routine(self, image, k1, k2, thres=0.05, 
                                     large_thres=450., areathres=30.,
                                     rdl=[50., 0., 0.]):
        """
        Dasiy-chains analyses to get
        basic image properties (centroids, radiality)
        from a single image
    
        Args:
        - image (array). image as numpy array
        - k1 (array). gaussian blur kernel
        - k2 (array). ricker wavelet kernel
        - thres (float). percentage threshold
        - areathres (float). area threshold
        - rdl (array). radiality thresholds
        """
        large_mask = self.detect_large_features(image, large_thres)
        img2, Gx, Gy, focusScore, cfactor = self.calculate_gradient_field(image, k1)
        dl_mask, centroids, radiality, idxs = self.small_feature_kernel(image, 
        large_mask, img2, Gx, Gy,
        k2, thres, areathres, rdl)
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

        mask_fill = np.divide(len(mask_indices), np.product(image_size))
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

        n_spots_in_mask = np.sum(np.in1d(spot_indices,mask_indices))
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
        estimated_background = np.zeros(len(indices), dtype=float)  # Estimated background per oligomer
    
        for k, index in enumerate(indices):
            in_pixels, out_pixels = self.intensity_pixel_indices(index, image_size)
            x_in, y_in = np.unravel_index(in_pixels, image_size, order='F')
            x_out, y_out = np.unravel_index(out_pixels, image_size, order='F')
            estimated_background[k] = np.mean(image[y_out, x_out])
            estimated_intensity[k] = np.sum(np.subtract(image[y_in, x_in], estimated_background[k]))
            if estimated_intensity[k] < 0:
                estimated_intensity[k] = np.NAN
                estimated_background[k] = np.NAN
        
        return estimated_intensity, estimated_background
    
    def intensity_pixel_indices(self, index, image_size):
        """
        Calculate pixel indices for inner and outer regions around the given index.
    
        Args:
        - index (int): Index of the pixel.
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
        
        centroid = np.asarray(np.unravel_index(index, image_size, order='F'), dtype=int)
        
        x_inner = x_inner + centroid[0]
        y_inner = y_inner + centroid[1]
        x_outer = x_outer + centroid[0]
        y_outer = y_outer + centroid[1]
        
        inner_indices = np.ravel_multi_index(np.vstack([x_inner, y_inner]), image_size, order='F')
        
        outer_indices = np.ravel_multi_index(np.vstack([x_outer, y_outer]), image_size, order='F')
        
        return inner_indices, outer_indices
    
    def detect_large_features(self, image, threshold, sigma1=2., sigma2=60.):
        """
        Detects large features in an image based on a given threshold.
    
        Args:
        - image (numpy.ndarray): Original image.
        - threshold (float): Threshold for determining features.
        - sigma1 (float): first gaussian blur width
        - sigma2 (float): second gaussian blur width
    
        Returns:
        - large_mask (numpy.ndarray): Binary mask for the large features.
        """
        # Apply Gaussian filters with different sigmas and subtract to enhance features
        enhanced_image = gaussian(image, sigma=sigma1, truncate=2.) - gaussian(image, sigma=sigma2, truncate=2.)
    
        # Create a binary mask for large features based on the threshold
        large_mask = enhanced_image > threshold
        
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

    def small_feature_kernel(self, img, large_mask, img2, Gx, Gy, k2, thres, area_thres, rdl):
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
    
        Returns:
        - dl_mask (numpy.ndarray): Binary mask for diffraction-limited (dl) features.
        - centroids (numpy.ndarray): Centroids for dl features.
        - radiality (numpy.ndarray): Radiality value for all features (before the filtering based on the radiality).
        - idxs (numpy.ndarray): Indices for objects that satisfy the decision boundary.
        """
        img1 = img - img2
        img1 = np.maximum(img1, 0)
        pad_size = [(sz - 1) // 2 for sz in k2.shape]
        img1 = np.pad(img1, pad_size, mode='edge')
        img1 = convolve2d(img1, k2, mode='valid')
    
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
        radiality = self.calculate_radiality(pil_small, img2, Gx, Gy)
    
        idxs = np.logical_and(radiality[:, 0] <= rdl[0], radiality[:, 1] >= rdl[1])
        centroids = np.floor(centroids[idxs])
        centroids = np.asarray(centroids)
        if len(pil_small[idxs]) > 1:
            dl_mask = self.create_filled_region(imsz, pil_small[idxs])
        else:
            dl_mask = np.full_like(img, False)
        return dl_mask, centroids, radiality, idxs


