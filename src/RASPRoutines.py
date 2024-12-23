# -*- coding: utf-8 -*-
"""
This class contains functions that collect analysis routines for RASP.
jsb92, 2024/02/08
"""
import os
import fnmatch
import numpy as np
import polars as pl
import sys
import time


module_dir = os.path.dirname(__file__)
sys.path.append(module_dir)
import IOFunctions

IO = IOFunctions.IO_Functions()
import AnalysisFunctions

A_F = AnalysisFunctions.Analysis_Functions()

import Image_Analysis_Functions

IA_F = Image_Analysis_Functions.ImageAnalysis_Functions()


class RASP_Routines:
    def __init__(
        self,
        defaultfolder=None,
        defaultarea=True,
        defaultd=True,
        defaultrad=True,
        defaultflat=True,
        defaultdfocus=True,
        defaultintfocus=True,
        defaultcellparams=True,
        defaultcameraparams=True,
    ):
        """
        Initialises class with default parameters for analysis.
        """
        self.defaultfolder = defaultfolder or os.path.join(
            os.path.split(module_dir)[0], "default_analysis_parameters"
        )

        # Initialise parameters
        if defaultarea:
            self.areathres = self._load_json_value(
                "areathres.json", "areathres", default=30.0, cast=float
            )

        if defaultd:
            self.d = self._load_json_value("areathres.json", "d", default=2, cast=int)

        if defaultrad:
            self.integratedGrad = self._load_json_value(
                "rad_neg.json", "integratedGrad", default=0.0, cast=float
            )

        if defaultflat:
            self.flatness = self._load_json_value(
                "rad_neg.json", "flatness", default=1.0, cast=float
            )

        if defaultdfocus:
            self.focus_score_diff = self._load_json_value(
                "infocus.json", "focus_score_diff", default=0.2, cast=float
            )

        if defaultintfocus:
            self.focus_score_int = self._load_json_value(
                "infocus.json", "focus_score_int", default=390.0, cast=float
            )

        if defaultcellparams:
            self._initialize_cell_params()

        if defaultcameraparams:
            self._initialize_camera_params()

    def _load_json_value(self, filename, key, default=None, cast=None):
        """
        Loads a value from a JSON file if it exists, otherwise returns a default.
        """
        file_path = os.path.join(self.defaultfolder, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r") as f:
                    data = IO.load_json(f)
                    return cast(data[key]) if cast else data[key]
            except (KeyError, ValueError, TypeError):
                pass
        return default

    def _initialize_cell_params(self):
        """
        Loads or sets default cell parameters.
        """
        default_values = {
            "sigma1": 2.0,
            "sigma2": 40.0,
            "threshold1": 200.0,
            "threshold2": 200.0,
        }
        data = self._load_json("default_cell_params.json", default_values)
        self.cell_sigma1 = data["sigma1"]
        self.cell_sigma2 = data["sigma2"]
        self.cell_threshold1 = data["threshold1"]
        self.cell_threshold2 = data["threshold2"]
        return

    def _initialize_camera_params(self):
        """
        Loads or sets default camera parameters.
        """
        self.QE = self._load_json_value(
            "camera_params.json", "QE", default=0.95, cast=float
        )
        self.gain_map = self._load_tiff_or_default("gain_map.tif", default=1.0)
        self.variance_map = self._load_tiff_or_default("variance_map.tif", default=1.0)
        self.offset_map = self._load_tiff_or_default("offset_map.tif", default=0.0)

        # Validate shapes of gain_map and offset_map
        if not isinstance(self.gain_map, float) and not isinstance(
            self.offset_map, float
        ):
            if self.gain_map.shape != self.offset_map.shape:
                print(
                    "Gain and Offset maps have different shapes. Resetting to default values."
                )
                self.gain_map, self.offset_map = 1.0, 0.0
        return

    def _load_json(self, filename, default):
        """
        Loads a JSON file and returns the data, or returns a default dictionary.
        """
        file_path = os.path.join(self.defaultfolder, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r") as f:
                    return IO.load_json(f)
            except (ValueError, TypeError):
                pass
        return default

    def _load_tiff_or_default(self, filename, default):
        """
        Loads a TIFF file if it exists, otherwise returns a default value.
        """
        file_path = os.path.join(self.defaultfolder, filename)
        if os.path.isfile(file_path):
            try:
                return IO.read_tiff(file_path)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        return default

    def get_infocus_planes(self, image, kernel):
        """
        Gets z planes that area in focus from an image stack

        Args:
            image (array): image as numpy array
            kernel (array): gaussian blur kernel

        Returns:
            z_planes (np.1darray): z_plane range that is in focus
        """

        na, na, na, focus_score, na = A_F.calculate_gradient_field(image, kernel)
        z_planes = A_F.infocus_indices(focus_score, self.focus_score_diff)
        return z_planes

    def calibrate_radiality(
        self,
        folder,
        imtype=".tif",
        protein_string="C0",
        gsigma=1.4,
        rwave=2.0,
        accepted_ratio=1,
    ):
        """
        Calibrates radility parameters. Given a folder of negative controls,
        analyses them and saves the radiality parameter to the .json file, as
        well as writing it to the current class radiality and flatness values

        Args:
            folder (string): Folder containing negative control tifs
            imtype (string): Type of images being analysed, default tif
            protein_string (string): Type of images being analysed, default C0
            gsigma (float): gaussian blurring parameter (default 1.4)
            rwave (float): Ricker wavelent sigma (default 2.)
            accepted_ratio (float): Percentage accepted of false positives
        """
        files_list = self.file_search(
            folder, protein_string, imtype
        )  # first get all files in any subfolders
        files = np.sort([e for e in files_list if imtype in e])

        k1, k2 = A_F.create_kernel(gsigma, rwave)  # create image processing kernels
        rdl = [np.inf, 0.0, 0.0]
        thres = 0.05
        r1_neg_forplot = {}  # generate dictionaries for plotting
        r2_neg_forplot = {}

        start = time.time()

        for i in np.arange(len(files)):
            file_path = os.path.join(folder, files[i])
            image = IO.read_tiff_tophotons(file_path)
            if len(image.shape) < 3:
                dl_mask, centroids, radiality, large_mask = A_F.compute_image_props(
                    image,
                    k1,
                    k2,
                    thres,
                    10000.0,
                    self.areathres,
                    rdl,
                    self.d,
                    calib=True,
                )
                r1_neg_forplot[i] = radiality[:, 0]
                r2_neg_forplot[i] = radiality[:, 1]
            else:
                z_planes = self.get_infocus_planes(image, k1)
                z_planes = np.arange(z_planes[0], z_planes[-1])
                if len(z_planes) != 0:  # if there are images we want to analyse
                    dl_mask, centroids, radiality, large_mask = A_F.compute_image_props(
                        image,
                        k1,
                        k2,
                        thres,
                        10000.0,
                        self.areathres,
                        rdl,
                        self.d,
                        z_planes=z_planes,
                        calib=True,
                    )
                    for z in enumerate(z_planes):
                        r1_neg_forplot[i, z[0]] = radiality[z[1]][:, 0]
                        r2_neg_forplot[i, z[0]] = radiality[z[1]][:, 1]
            print(
                "Analysed image file {}/{}    Time elapsed: {:.3f} s".format(
                    i + 1, len(files), time.time() - start
                ),
                end="\r",
                flush=True,
            )

        ### IQR filtering
        means_rad1 = np.zeros(len(r1_neg_forplot.keys()))
        means_rad2 = np.zeros(len(r2_neg_forplot.keys()))

        keys = list(r1_neg_forplot.keys())

        for i, key in enumerate(keys):
            means_rad1[i] = np.nanmean(r1_neg_forplot[key])
            means_rad2[i] = np.nanmean(r2_neg_forplot[key])

        r1_reject = A_F.rejectoutliers_ind(means_rad1)
        if len(r1_reject) > 0:
            for val in r1_reject:
                r1_neg_forplot.pop(keys[val])

        r2_reject = A_F.rejectoutliers_ind(means_rad2)
        if len(r2_reject) > 0:
            for val in r2_reject:
                r2_neg_forplot.pop(keys[val])
        ### IQR filtering complete

        for i, key in enumerate(r1_neg_forplot.keys()):
            if i == 0:
                r1_neg = r1_neg_forplot[key]
            else:
                r1_neg = np.hstack([r1_neg, r1_neg_forplot[key]])

        for i, key in enumerate(r2_neg_forplot.keys()):
            if i == 0:
                r2_neg = r2_neg_forplot[key]
            else:
                r2_neg = np.hstack([r2_neg, r2_neg_forplot[key]])

        rad_1 = np.percentile(r1_neg, accepted_ratio)
        rad_2 = np.percentile(r2_neg, 100.0 - accepted_ratio)

        to_save = {"flatness": rad_1, "integratedGrad": rad_2}

        IO.make_directory(self.defaultfolder)
        IO.save_as_json(to_save, os.path.join(self.defaultfolder, "rad_neg.json"))
        self.flatness = rad_1
        self.integratedGrad = rad_2
        print(
            "Radiality calibrated using negative control"
            + " images in "
            + str(folder)
            + ". New flatness is "
            + str(np.around(rad_1, 2))
            + " and new integrated gradient is "
            + str(np.around(rad_2, 2))
            + ". Parameters saved in "
            + str(self.defaultfolder)
            + "."
        )

        import PlottingFunctions

        plots = PlottingFunctions.Plotter()
        import matplotlib.pyplot as plt

        fig, axs = plots.two_column_plot(
            nrows=1, ncolumns=2, heightratio=[1], widthratio=[1, 1]
        )

        bins_r1 = np.histogram_bin_edges(r1_neg, bins="fd")
        bins_r2 = np.histogram_bin_edges(r2_neg, bins="fd")
        axs[0] = plots.histogram_plot(axs[0], r1_neg, bins=bins_r1, alpha=0.5)
        axs[1] = plots.histogram_plot(axs[1], r2_neg, bins=bins_r2, alpha=0.5)

        ylim0, ylim1 = axs[0].get_ylim()[0], axs[0].get_ylim()[1]
        axs[0].vlines(rad_1, ylim0, ylim1, color="k", label="threshold", ls="--")

        axs[0].set_ylim([ylim0, ylim1])
        axs[0].set_xlabel("flatness metric")
        axs[0].set_ylabel("probability density")

        ylim0, ylim1 = axs[1].get_ylim()[0], axs[1].get_ylim()[1]
        axs[1].vlines(rad_2, ylim0, ylim1, color="k", label="threshold", ls="--")
        axs[1].set_xlabel("integrated gradient metric")
        axs[1].set_xlim([0, rad_2 * 2.0])
        axs[1].set_ylim([ylim0, ylim1])

        axs[0].legend(loc="best", frameon=False)
        axs[1].legend(loc="best", frameon=False)
        plt.tight_layout()
        plt.show(block=False)

        return

    # TODO: fix
    def calibrate_area(
        self, folder, imtype=".tif", gsigma=1.4, rwave=2.0, large_thres=10000.0
    ):
        """
        Calibrates area threshold. Analyzes bead images in a folder and saves
        the radiality parameter and class radiality/flatness values.

        Args:
            folder (string): Folder containing bead (bright) control tifs.
            imtype (string): Type of images being analyzed, default .tif.
            gsigma (float): Gaussian blurring parameter (default 1.4).
            rwave (float): Ricker wavelet sigma (default 2.0).
        """

        def process_image(
            image, k1, k2, thres, large_thres, areathres, rdl, z_planes=None
        ):
            """
            Processes an image (2D or 3D), calculates region properties, and HWHM.
            """
            if z_planes is None:  # 2D image
                dl_mask, _, _, _ = A_F.compute_image_props(
                    image, k1, k2, thres, large_thres, areathres, rdl, self.d
                )
                pixel_indices, areas, _ = A_F.calculate_region_properties(dl_mask)
                return areas, A_F.Gauss2DFitting(image, pixel_indices)
            else:  # 3D image
                all_areas, all_HWHM = [], []
                for z in z_planes:
                    dl_mask, _, _, _ = A_F.compute_image_props(
                        image[:, :, z],
                        k1,
                        k2,
                        thres,
                        large_thres,
                        areathres,
                        rdl,
                        self.d,
                    )
                    pixel_indices, areas, _ = A_F.calculate_region_properties(dl_mask)
                    all_areas.extend(areas)
                    all_HWHM.extend(A_F.Gauss2DFitting(image[:, :, z], pixel_indices))
                return all_areas, all_HWHM

        # Initialize variables and kernels
        files = sorted(f for f in os.listdir(folder) if imtype in f)
        k1, k2 = A_F.create_kernel(gsigma, rwave)
        accepted_ratio = 95.0
        thres, areathres, rdl = 0.05, 1000.0, [self.flatness, self.integratedGrad, 0.0]

        all_areas, all_HWHM = [], []
        start_time = time.time()

        # Process each file
        for i, file_name in enumerate(files):
            file_path = os.path.join(folder, file_name)
            image = IO.read_tiff(file_path)
            if image.ndim < 3:
                areas, HWHM = process_image(
                    image, k1, k2, thres, large_thres, areathres, rdl
                )
            else:
                areas, HWHM = process_image(
                    image,
                    k1,
                    k2,
                    thres,
                    large_thres,
                    areathres,
                    rdl,
                    z_planes=range(image.shape[2]),
                )
            all_areas.extend(areas)
            all_HWHM.extend(HWHM)

            print(
                f"Processed file {i + 1}/{len(files)} | Time elapsed: {time.time() - start_time:.2f}s",
                end="\r",
                flush=True,
            )

        # Final calculations
        all_HWHM = A_F.rejectoutliers(np.asarray(all_HWHM))
        area_thresh = int(np.ceil(np.percentile(all_areas, accepted_ratio)))
        pixel_d = int(np.round(np.mean(all_HWHM)))

        # Save parameters
        to_save = {"areathres": area_thresh, "d": pixel_d}
        IO.make_directory(self.defaultfolder)
        IO.save_as_json(to_save, os.path.join(self.defaultfolder, "areathres.json"))
        self.areathres = area_thresh
        self.d = pixel_d

        print(
            f"Calibration complete. Area threshold: {area_thresh}, Radiality radius: {pixel_d}. "
            f"Parameters saved to {self.defaultfolder}."
        )

        # Plot results
        import matplotlib.pyplot as plt
        from PlottingFunctions import Plotter

        plots = Plotter()
        fig, axs = plots.two_column_plot(nrows=1, ncolumns=2)

        axs[0].ecdf(all_areas, color="k")
        axs[0].hlines(
            accepted_ratio / 100.0,
            *axs[0].get_xlim(),
            color="k",
            ls="--",
            label="Threshold",
        )
        axs[0].set_ylim([0, 1])
        axs[0].set_xlabel("Puncta area (pixels)")
        axs[0].set_ylabel("Probability")
        axs[0].legend(loc="lower right", frameon=False)

        axs[1] = plots.histogram_plot(axs[1], all_HWHM, bins="fd")
        axs[1].vlines(
            np.mean(all_HWHM), *axs[1].get_ylim(), color="k", ls="--", label="Mean HWHM"
        )
        axs[1].set_xlabel("HWHM (pixels)")
        axs[1].legend(loc="best", frameon=False)

        plt.tight_layout()
        plt.show(block=False)

        return

    def analyse_images(
        self,
        folder,
        imtype=".tif",
        thres=0.05,
        large_thres=200.0,
        gsigma=1.4,
        rwave=2.0,
        protein_string="C1",
        cell_string="C0",
        if_filter=True,
        im_start=0,
        cell_analysis=True,
        one_savefile=True,
        disp=True,
        analyse_clr=False,
        folder_recursive=False,
        error_reduction=False,
    ):
        """
        analyses data from images in a specified folder,
        saves spots, locations, intensities and backgrounds in a folder created
        next to the folder analysed with _analysis string attached
        also writes a folder with _analysisparameters and saves analysis parameters
        used for particular experiment

        Args:
            folder (string): Folder containing images
            imtype (string): Type of images being analysed. Default '.tif'
            thres (float): fraction of bright pixels accepted. Default 0.05.
            large_thres (float): large object intensity threshold. Default 100.
            gisgma (float): gaussian blurring parameter. Default 1.4.
            rwave (float): Ricker wavelent sigma. Default 2.
            protein_string (np.1darray): strings for protein-stained data. Default C1.
            cell_string (np.1darray): strings for cell-containing data. Default C0.
            if_filter (boolean): Filter images for focus. Default True.
            im_start (integer): Images to start from. Default 0.
            cell_analysis (boolean): Parameter where script also analyses cell
                images and computes colocalisation likelihood ratios. Default True.
            one_savefile (boolean): Parameter that, if true, doesn't save a file. Default True.
                per image but amalgamates them into one file. Default True.
            disp (boolean): If true, prints when analysed an image stack. Default True.
            analyse_clr (boolean): If true, calculates the clr. If not, just coincidence. Default True.
            folder_recursion (boolean): If true, recursively finds folders and analyses each separately.
            error_reduction (boolean): If true, reduces error on the oligomer image using Huang's code

        """
        all_files = self.file_search(
            folder, protein_string, imtype
        )  # first get all files in any subfolders

        all_files = np.sort([e for e in all_files if "loMask" not in e])

        folders = np.unique(
            [os.path.split(i)[0] for i in all_files]
        )  # get unique folders in this

        k1, k2 = A_F.create_kernel(gsigma, rwave)  # create image processing kernels
        rdl = [self.flatness, self.integratedGrad, 0.0]

        to_save = {
            "flatness": self.flatness,
            "integratedGrad": self.integratedGrad,
            "areathres": self.areathres,
            "d": self.d,
            "gaussian_sigma": gsigma,
            "ricker_sigma": rwave,
            "thres": thres,
            "large_thres": large_thres,
            "focus_score_diff": self.focus_score_diff,
            "cell_sigma1": self.cell_sigma1,
            "cell_sigma2": self.cell_sigma2,
            "cell_threshold1": self.cell_threshold1,
            "cell_threshold2": self.cell_threshold2,
            "QE": self.QE,
        }

        # save analysis parameters in overall directory
        analysis_p_directory = os.path.abspath(folder) + "_analysisparameters"

        IO.save_analysis_params(
            analysis_p_directory,
            to_save,
            gain_map=self.gain_map,
            offset_map=self.offset_map,
            variance_map=self.variance_map,
        )

        start = time.time()

        if folder_recursive == True:
            for val in folders:
                subfolder = os.path.abspath(val)
                files = self.file_search(
                    subfolder, protein_string, imtype
                )  # first get all files in any subfolders
                if cell_analysis == True:
                    cell_files = self.file_search(
                        subfolder, cell_string, imtype
                    )  # get all files in any subfolders
                # create analysis and analysis parameter directories
                analysis_directory = os.path.abspath(subfolder) + "_analysis"
                IO.make_directory(analysis_directory)

                for i in np.arange(len(files)):
                    if error_reduction == False:
                        img = IO.read_tiff_tophotons(
                            os.path.join(folder, files[i]),
                            QE=self.QE,
                            gain_map=self.gain_map,
                            offset_map=self.offset_map,
                        )[:, :, im_start:]
                    else:
                        img = IO.read_tiff_tophotons(
                            os.path.join(folder, files[i]),
                            QE=self.QE,
                            gain_map=self.gain_map,
                            offset_map=self.offset_map,
                            variance_map=self.variance_map,
                            error_correction=True,
                        )[:, :, im_start:]
                    if cell_analysis == True:
                        img_cell = IO.read_tiff_tophotons(
                            os.path.join(folder, cell_files[i]),
                            QE=self.QE,
                            gain_map=self.gain_map,
                            offset_map=self.offset_map,
                        )[:, :, im_start:]
                    z_test = len(img.shape) > 2

                    if z_test:  # if a z-stack
                        z_planes = self.get_infocus_planes(img, k1)
                        if cell_analysis == False:
                            to_save, to_save_largeobjects, lo_mask = (
                                A_F.compute_spot_props(
                                    img,
                                    k1,
                                    k2,
                                    thres=thres,
                                    large_thres=large_thres,
                                    areathres=self.areathres,
                                    rdl=rdl,
                                    z=z_planes,
                                    d=self.d,
                                )
                            )
                        else:
                            (
                                to_save,
                                to_save_largeobjects,
                                lo_mask,
                                to_save_cell,
                                cell_mask,
                            ) = A_F.compute_spot_and_cell_props(
                                img,
                                img_cell,
                                k1,
                                k2,
                                prot_thres=thres,
                                large_prot_thres=large_thres,
                                areathres=self.areathres,
                                rdl=rdl,
                                z=z_planes,
                                cell_threshold1=self.cell_threshold1,
                                cell_threshold2=self.cell_threshold1,
                                cell_sigma1=self.cell_sigma1,
                                cell_sigma2=self.cell_sigma2,
                                d=self.d,
                                analyse_clr=analyse_clr,
                            )

                        if cell_analysis == True:
                            IO.save_analysis(
                                to_save,
                                to_save_largeobjects,
                                analysis_directory,
                                imtype,
                                protein_string,
                                cell_string,
                                files,
                                i,
                                z_planes,
                                lo_mask,
                                cell_analysis=cell_analysis,
                                cell_mask=cell_mask,
                                to_save_cell=to_save_cell,
                                one_savefile=one_savefile,
                            )
                        else:
                            IO.save_analysis(
                                to_save,
                                to_save_largeobjects,
                                analysis_directory,
                                imtype,
                                protein_string,
                                cell_string,
                                files,
                                i,
                                z_planes,
                                lo_mask,
                                cell_analysis=False,
                                cell_mask=False,
                                to_save_cell=False,
                                one_savefile=one_savefile,
                            )
                    if disp == True:
                        print(
                            "Analysed image file {}/{}    Time elapsed: {:.3f} s".format(
                                i + 1, len(files), time.time() - start
                            ),
                            end="\r",
                            flush=True,
                        )
        else:
            files = all_files
            if cell_analysis == True:
                cell_files = self.file_search(
                    folder, cell_string, imtype
                )  # get all files in any subfolders
            # create analysis and analysis parameter directories
            analysis_directory = os.path.abspath(folder) + "_analysis"
            IO.make_directory(analysis_directory)
            for i in np.arange(len(files)):
                if error_reduction == False:
                    img = IO.read_tiff_tophotons(
                        os.path.join(folder, files[i]),
                        QE=self.QE,
                        gain_map=self.gain_map,
                        offset_map=self.offset_map,
                    )[:, :, im_start:]
                else:
                    img = IO.read_tiff_tophotons(
                        os.path.join(folder, files[i]),
                        QE=self.QE,
                        gain_map=self.gain_map,
                        offset_map=self.offset_map,
                        variance_map=self.variance_map,
                        error_correction=True,
                    )[:, :, im_start:]
                if cell_analysis == True:
                    img_cell = IO.read_tiff_tophotons(
                        os.path.join(folder, cell_files[i]),
                        QE=self.QE,
                        gain_map=self.gain_map,
                        offset_map=self.offset_map,
                    )[:, :, im_start:]
                z_test = len(img.shape) > 2

                if z_test:  # if a z-stack
                    z_planes = self.get_infocus_planes(img, k1)
                    if cell_analysis == False:
                        to_save, to_save_largeobjects, lo_mask = A_F.compute_spot_props(
                            img,
                            k1,
                            k2,
                            thres=thres,
                            large_thres=large_thres,
                            areathres=self.areathres,
                            rdl=rdl,
                            z=z_planes,
                            d=self.d,
                        )
                    else:
                        (
                            to_save,
                            to_save_largeobjects,
                            lo_mask,
                            to_save_cell,
                            cell_mask,
                        ) = A_F.compute_spot_and_cell_props(
                            img,
                            img_cell,
                            k1,
                            k2,
                            prot_thres=thres,
                            large_prot_thres=large_thres,
                            areathres=self.areathres,
                            rdl=rdl,
                            z=z_planes,
                            cell_threshold1=self.cell_threshold1,
                            cell_threshold2=self.cell_threshold1,
                            cell_sigma1=self.cell_sigma1,
                            cell_sigma2=self.cell_sigma2,
                            d=self.d,
                            analyse_clr=analyse_clr,
                        )

                    if cell_analysis == True:
                        IO.save_analysis(
                            to_save,
                            to_save_largeobjects,
                            analysis_directory,
                            imtype,
                            protein_string,
                            cell_string,
                            files,
                            i,
                            z_planes,
                            lo_mask,
                            cell_analysis=cell_analysis,
                            cell_mask=cell_mask,
                            to_save_cell=to_save_cell,
                            one_savefile=one_savefile,
                        )
                    else:
                        IO.save_analysis(
                            to_save,
                            to_save_largeobjects,
                            analysis_directory,
                            imtype,
                            protein_string,
                            cell_string,
                            files,
                            i,
                            z_planes,
                            lo_mask,
                            cell_analysis=False,
                            cell_mask=False,
                            to_save_cell=False,
                            one_savefile=one_savefile,
                        )
                if disp == True:
                    print(
                        "Analysed image file {}/{}    Time elapsed: {:.3f} s".format(
                            i + 1, len(files), time.time() - start
                        ),
                        end="\r",
                        flush=True,
                    )
        return

    def single_image_analysis(
        self,
        protein_file,
        thres=0.05,
        large_thres=100.0,
        gsigma=1.4,
        rwave=2.0,
        image_size=200,
        save_figure=False,
        cell_analysis=False,
        cell_file=None,
    ):
        """
        analyses data from specified image,
        presents spots, locations, intensities in a figure, with the option of
        saving this figure

        Args:
            file (string): image location
            thres (float): fraction of bright pixels accepted
            large_thres (float): large object intensity threshold
            gisgma (float): gaussian blurring parameter (default 1.4)
            rwave (float): Ricker wavelent sigma (default 2.)
            image_size (int): Amount of image to plot---by default plots 100x100
                chunk of an image to give you an idea, can scale up
            save_figure (boolean): save the figure as an svg, default no
            cell_analysis (boolean): Parameter where script also analyses cell
                images and computes colocalisation likelihood ratios.
            cell_file (string): cell image location

        """
        import PlottingFunctions

        plots = PlottingFunctions.Plotter()
        import matplotlib.pyplot as plt

        img = IO.read_tiff_tophotons(
            protein_file, QE=self.QE, gain_map=self.gain_map, offset_map=self.offset_map
        )

        k1, k2 = A_F.create_kernel(gsigma, rwave)  # create image processing kernels
        rdl = [self.flatness, self.integratedGrad, 0.0]

        if cell_analysis == True:
            img_cell = IO.read_tiff_tophotons(
                cell_file,
                QE=self.QE,
                gain_map=self.gain_map,
                offset_map=self.offset_map,
            )

        if len(img.shape) > 2:  # if a z-stack
            z_planes = self.get_infocus_planes(img, k1)

            if cell_analysis == False:
                to_save, to_save_largeobjects, lo_mask = A_F.compute_spot_props(
                    img,
                    k1,
                    k2,
                    thres=thres,
                    large_thres=large_thres,
                    areathres=self.areathres,
                    rdl=rdl,
                    z=z_planes,
                    d=self.d,
                )
            else:
                to_save, to_save_largeobjects, lo_mask, to_save_cell, cell_mask = (
                    A_F.compute_spot_and_cell_props(
                        img,
                        img_cell,
                        k1,
                        k2,
                        prot_thres=thres,
                        large_prot_thres=large_thres,
                        areathres=self.areathres,
                        rdl=rdl,
                        z=z_planes,
                        cell_threshold1=self.cell_threshold1,
                        cell_threshold2=self.cell_threshold1,
                        cell_sigma1=self.cell_sigma1,
                        cell_sigma2=self.cell_sigma2,
                        d=self.d,
                    )
                )

        z_to_plot = np.full_like(np.arange(z_planes[0] + 1, z_planes[-1] + 1), -1)
        for i, val in enumerate(np.arange(z_planes[0] + 1, z_planes[-1] + 1)):
            if (
                len(
                    to_save.filter(pl.col("z") == val)[
                        "sum_intensity_in_photons"
                    ].to_numpy()
                )
                > 1
            ):
                z_to_plot[i] = val
        z_to_plot = z_to_plot[z_to_plot >= 0]

        if cell_analysis == False:

            for i in enumerate(z_to_plot):
                fig, axs = plots.two_column_plot(nrows=1, ncolumns=2, widthratio=[1, 1])
                xpositions = to_save.filter(pl.col("z") == i[1])["x"].to_numpy()
                ypositions = to_save.filter(pl.col("z") == i[1])["y"].to_numpy()

                xpositions_large = to_save_largeobjects.filter(pl.col("z") == i[1])[
                    "x"
                ].to_numpy()
                ypositions_large = to_save_largeobjects.filter(pl.col("z") == i[1])[
                    "y"
                ].to_numpy()
                testvals = (xpositions < image_size) * (ypositions < image_size)
                xpositions = xpositions[testvals]
                ypositions = ypositions[testvals]
                axs[0] = plots.image_scatter_plot(
                    axs[0],
                    img[:image_size, :image_size, i[1] - 1],
                    xdata=xpositions,
                    ydata=ypositions,
                    label="z plane = " + str(int(i[1])),
                )

                axs[1] = plots.image_scatter_plot(
                    axs[1],
                    img[:, :, i[1] - 1],
                    xdata=xpositions_large,
                    ydata=ypositions_large,
                    label="z plane = " + str(int(i[1])),
                )
                plt.tight_layout()
                if save_figure == True:
                    plt.savefig(
                        protein_file.split(".")[0]
                        + "_ExampleFigure_zplane"
                        + str(int(i[1]))
                        + ".svg",
                        format="svg",
                        dpi=600,
                    )
                plt.show()
        else:
            for i in enumerate(z_to_plot):
                fig, axs = plots.two_column_plot(
                    nrows=1, ncolumns=3, widthratio=[1, 1, 1]
                )
                xpositions = to_save.filter(pl.col("z") == i[1])["x"].to_numpy()
                ypositions = to_save.filter(pl.col("z") == i[1])["y"].to_numpy()
                xpositions_large = to_save_largeobjects.filter(pl.col("z") == i[1])[
                    "x"
                ].to_numpy()
                ypositions_large = to_save_largeobjects.filter(pl.col("z") == i[1])[
                    "y"
                ].to_numpy()
                testvals = (xpositions < image_size) * (ypositions < image_size)
                xpositions = xpositions[testvals]
                ypositions = ypositions[testvals]
                axs[0] = plots.image_scatter_plot(
                    axs[0],
                    img[:image_size, :image_size, i[1] - 1],
                    xdata=xpositions,
                    ydata=ypositions,
                    label="puncta, z plane = " + str(int(i[1])),
                )

                axs[1] = plots.image_scatter_plot(
                    axs[1],
                    img[:, :, i[1] - 1],
                    xdata=xpositions_large,
                    ydata=ypositions_large,
                    label="z plane = " + str(int(i[1])),
                )

                axs[2] = plots.image_plot(
                    axs[2],
                    img_cell[:, :, i[1] - 1],
                    label="cell, z plane = " + str(int(i[1])),
                    plotmask=True,
                    mask=cell_mask[:, :, i[1] - 1],
                )

                plt.tight_layout()

                if save_figure == True:
                    plt.savefig(
                        protein_file.split(".")[0]
                        + "_ExampleFigure_zplane"
                        + str(int(i[1]))
                        + ".svg",
                        format="svg",
                        dpi=600,
                    )
                plt.show()
        return

    def calculate_spot_mask_rdf_with_threshold(
        self,
        analysis_file,
        threshold,
        out_cell=True,
        pixel_size=0.11,
        dr=1.0,
        cell_string="C0",
        protein_string="C1",
        imtype=".tif",
    ):
        """
        Does rdf analysis of spots wrt mask from an analysis file.

        Args:
            analysis_file (str): The analysis file to be re-done.
            threshold (float): The photon threshold
            out_cell (boolean): If True will only consider oligomers deemed to be outside of cells
            pixel_size (float): size of pixels
            dr (float): dr of rdf
            n_iter (int): number of CSR iterations
            cell_string (string): will use this to find corresponding mask files
            protein_string (string): protein string
            imtype (string): image type previously analysed

        Returns:
            rdf (pd.DataArray): pandas datarray of the rdf
        """
        if int(threshold) == threshold:
            thesholdsavestr = str(threshold)
        else:
            thesholdsavestr = str(threshold).replace(".", "p")

        rdf_AT = A_F.spot_mask_rdf_with_threshold(
            analysis_file,
            threshold,
            out_cell=out_cell,
            pixel_size=pixel_size,
            dr=dr,
            cell_string=cell_string,
            protein_string=protein_string,
            imtype=imtype,
            aboveT=1,
        )

        rdf_UT = A_F.spot_mask_rdf_with_threshold(
            analysis_file,
            threshold,
            out_cell=out_cell,
            pixel_size=pixel_size,
            dr=dr,
            cell_string=cell_string,
            protein_string=protein_string,
            imtype=imtype,
            aboveT=0,
        )

        to_save_name = os.path.join(
            os.path.split(analysis_file)[0],
            "spot_to_mask_" + cell_string + "_threshold_" + thesholdsavestr,
        )

        if isinstance(rdf_AT, pl.DataFrame):
            rdf_AT.write_csv(to_save_name + "_abovethreshold_rdf.csv")
        if isinstance(rdf_UT, pl.DataFrame):
            rdf_UT.write_csv(to_save_name + "_belowthreshold_rdf.csv")
        return rdf_AT, rdf_UT

    def calculate_two_puncta_channels_rdf_with_thresholds(
        self,
        analysis_file_p1,
        analysis_file_p2,
        threshold_p1,
        threshold_p2,
        pixel_size=0.11,
        dr=1.0,
        protein_string_1="C0",
        protein_string_2="C1",
        imtype=".tif",
    ):
        """
        Does rdf analysis of spots wrt mask from an analysis file.

        Args:
            analysis_file_p1 (str): The analysis file of puncta set 1.
            analysis_file_p2 (str): The analysis file of puncta set 2.
            threshold_p1 (float): The photon threshold for puncta set 1.
            threshold_p2 (float): The photon threshold for puncta set 2.
            pixel_size (float): size of pixels
            dr (float): dr of rdf
            n_iter (int): number of CSR iterations
            protein_string_1 (string): will use this to find corresponding other punctum files
            protein_string_2 (string): will use this to find corresponding other punctum files
            imtype (string): image type previously analysed

        Returns:
            rdf (pl.DataArray): polars datarray of the rdf
        """
        analysis_data_p1 = pl.read_csv(analysis_file_p1)

        analysis_data_p2 = pl.read_csv(analysis_file_p2)

        if int(threshold_p1) == threshold_p1:
            theshold_p1_savestr = str(threshold_p1)
        else:
            theshold_p1_savestr = str(threshold_p1).replace(".", "p")

        if int(threshold_p2) == threshold_p2:
            theshold_p2_savestr = str(threshold_p2)
        else:
            theshold_p2_savestr = str(threshold_p2).replace(".", "p")

        rdf_AT = A_F.two_puncta_channels_rdf_with_thresholds(
            analysis_data_p1,
            analysis_data_p2,
            threshold_p1,
            threshold_p2,
            pixel_size=pixel_size,
            dr=dr,
            protein_string_1=protein_string_1,
            protein_string_2=protein_string_2,
            imtype=imtype,
            aboveT=1,
        )

        rdf_UT = A_F.two_puncta_channels_rdf_with_thresholds(
            analysis_data_p1,
            analysis_data_p2,
            threshold_p1,
            threshold_p2,
            pixel_size=pixel_size,
            dr=dr,
            protein_string_1=protein_string_1,
            protein_string_2=protein_string_2,
            imtype=imtype,
            aboveT=0,
        )

        to_save_name = os.path.join(
            os.path.split(analysis_file_p1)[0],
            "puncta1_"
            + protein_string_1
            + "_to_puncta2_"
            + protein_string_2
            + "_threshold_puncta1_"
            + theshold_p1_savestr
            + "_threshold_puncta2_"
            + theshold_p2_savestr,
        )

        if isinstance(rdf_AT, pl.DataFrame):
            rdf_AT.write_csv(to_save_name + "_rdf_abovethreshold.csv")
        if isinstance(rdf_UT, pl.DataFrame):
            rdf_UT.write_csv(
                to_save_name + "_rdf_belowthreshold.csv",
            )
        return rdf_AT, rdf_UT

    def calculate_spot_rdf_with_threshold(
        self,
        analysis_file,
        threshold,
        pixel_size=0.11,
        dr=1.0,
    ):
        """
        Does rdf analysis of spots above a photon threshold in an
        analysis file.

        Uses code from 10.5281/zenodo.4625675, please cite this software if used in a paper.

        Args:
            analysis_file (str): The analysis file to be re-done.
            threshold (float): The photon threshold
            pixel_size (float): size of pixels
            dr (float): dr of rdf

        Returns:
            rdf (pd.DataArray): pandas datarray of the rdf
        """
        if int(threshold) == threshold:
            thesholdsavestr = str(threshold)
        else:
            thesholdsavestr = str(threshold).replace(".", "p")
        rdf_AT = A_F.single_spot_channel_rdf_with_threshold(
            analysis_file, threshold, pixel_size=pixel_size, dr=dr, aboveT=1
        )
        rdf_UT = A_F.single_spot_channel_rdf_with_threshold(
            analysis_file, threshold, pixel_size=pixel_size, dr=dr, aboveT=0
        )
        to_save_name = os.path.join(
            os.path.split(analysis_file)[0], "spot_to_spot_threshold_" + thesholdsavestr
        )

        if isinstance(rdf_AT, pl.DataFrame):
            rdf_AT.write_csv(to_save_name + "_abovethreshold_rdf.csv")
        if isinstance(rdf_UT, pl.DataFrame):
            rdf_UT.write_csv(to_save_name + "_belowthreshold_rdf.csv")
        return rdf_AT, rdf_UT

    def count_puncta_in_individual_cells_threshold(
        self,
        analysis_file,
        threshold,
        cell_string,
        protein_string,
        lower_cell_size_threshold=2000,
        upper_cell_size_threshold=np.inf,
        imtype=".tif",
        blur_degree=1,
        z_project_first=True,
        replace_files=False,
        q1=None,
        q2=None,
        IQR=None,
    ):
        """
        Redo colocalisation analayses of spots above a photon threshold in an
        analysis file.

        Args:
            analysis_file (str): The analysis file to be re-done.
            threshold (float): The photon threshold
            cell_string (str): string of cell to analyse
            protein_string (str): string of analysed protein
            lower_cell_size_threshold (float): lower cell size threshold
            upper_cell_size_threshold (float): upper cell size threshold
            imtype (str): image type
            blur_degree (int): blur degree for colocalisation analysis
            z_project_first (boolean): if True (default), does a z projection before
                                    thresholding cell size. If false, does the opposite.
            replace_files (boolean): if False, looks for files first and if it's already analysed, does nothing
            q1 (float): if float, adds in IQR filter
            q2 (float): if float, adds in IQR filter
            IQR (Float): if float, adds in IQR filter


        Returns:
            cell_punctum_analysis_AT (pl.DataFrame): dataframe of cell analysis above threshold
            cell_punctum_analysis_UT (pl.DataFrame): dataframe of cell analysis below threshold

        """
        if int(threshold) == threshold:
            threshold_str = str(int(threshold))
        else:
            threshold_str = str(np.around(threshold, 1)).replace(".", "p")

        if (
            ~isinstance(q1, type(None))
            and ~isinstance(q2, type(None))
            and ~isinstance(IQR, type(None))
        ):
            threshold_str = threshold_str + "_outliersremoved"

        if int(lower_cell_size_threshold) == lower_cell_size_threshold:
            lc_str = str(int(lower_cell_size_threshold))
        else:
            lc_str = str(np.around(lower_cell_size_threshold, 1)).replace(".", "p")

        if np.isinf(upper_cell_size_threshold):
            savecell_string = os.path.join(
                os.path.split(analysis_file)[0],
                "single_cell_coincidence_"
                + "mincellsize_"
                + lc_str
                + "_photonthreshold_"
                + threshold_str
                + "_photons",
            )
            above_string = savecell_string + "_abovethreshold.csv"
            below_string = savecell_string + "_belowthreshold.csv"
        else:
            if int(upper_cell_size_threshold) == upper_cell_size_threshold:
                uc_str = str(int(upper_cell_size_threshold))
            else:
                uc_str = str(np.around(upper_cell_size_threshold, 1)).replace(".", "p")

            savecell_string = os.path.join(
                os.path.split(analysis_file)[0],
                "single_cell_coincidence_"
                + "mincellsize_"
                + lc_str
                + "_maxcellsize_"
                + uc_str
                + "_photonthreshold_"
                + threshold_str
                + "_photons",
            )
            above_string = savecell_string + "_abovethreshold.csv"
            below_string = savecell_string + "_belowthreshold.csv"

        if replace_files == False:
            if os.path.isfile(above_string) or os.path.isfile(below_string):
                print("Analysis already complete; exiting.")
                return

        cell_punctum_analysis_AT = (
            A_F.number_of_puncta_per_segmented_cell_with_threshold(
                analysis_file,
                threshold,
                lower_cell_size_threshold=lower_cell_size_threshold,
                upper_cell_size_threshold=upper_cell_size_threshold,
                blur_degree=blur_degree,
                cell_string=cell_string,
                protein_string=protein_string,
                imtype=imtype,
                aboveT=1,
                z_project_first=z_project_first,
                q1=q1,
                q2=q2,
                IQR=IQR,
            )
        )

        cell_punctum_analysis_UT = (
            A_F.number_of_puncta_per_segmented_cell_with_threshold(
                analysis_file,
                threshold,
                lower_cell_size_threshold=lower_cell_size_threshold,
                upper_cell_size_threshold=upper_cell_size_threshold,
                blur_degree=blur_degree,
                cell_string=cell_string,
                protein_string=protein_string,
                imtype=imtype,
                aboveT=0,
                z_project_first=z_project_first,
                q1=q1,
                q2=q2,
                IQR=IQR,
            )
        )

        if isinstance(cell_punctum_analysis_AT, pl.DataFrame) and isinstance(
            cell_punctum_analysis_UT, pl.DataFrame
        ):
            cell_punctum_analysis_AT.write_csv(above_string)
            cell_punctum_analysis_UT.write_csv(below_string)
            cell_punctum_analysis = cell_punctum_analysis_AT
        else:
            if isinstance(cell_punctum_analysis_AT, pl.DataFrame):
                cell_punctum_analysis_AT.write_csv(above_string)
                cell_punctum_analysis = cell_punctum_analysis_AT
            if isinstance(cell_punctum_analysis_UT, pl.DataFrame):
                cell_punctum_analysis_UT.write_csv(below_string)
        return cell_punctum_analysis

    def colocalise_with_threshold(
        self,
        analysis_file,
        threshold,
        protein_string,
        lo_string,
        coloc_type=1,
        imtype=".tif",
        blur_degree=1,
        calc_clr=False,
    ):
        """
        Redo colocalisation analyses of spots above a photon threshold in an
        analysis file.

        Args:
            analysis_file (str): The analysis file to be re-done.
            threshold (float): The photon threshold
            protein_string (str): string of analysed protein
            lo_string (str): string of large object to analyse
            coloc_type (boolean): if 1 (default), for cells. if 0, for large protein objects
            imtype (str): image type
            blur_degree (int): blur degree for colocalisation analysis
            calc_clr (boolean): Calculate the clr, yes/no.
        """
        if coloc_type == 1:
            startstr = "cell_"
        else:
            startstr = "lo_"

        if int(threshold) == threshold:
            threshold_str = str(int(threshold))
        else:
            threshold_str = str(threshold).replace(".", "p")

        lo_analysis_AT, spot_analysis_AT = A_F.colocalise_with_threshold(
            analysis_file,
            threshold,
            protein_string,
            lo_string,
            coloc_type=coloc_type,
            imtype=".tif",
            blur_degree=1,
            calc_clr=False,
            aboveT=1,
        )

        lo_analysis_UT, spot_analysis_UT = A_F.colocalise_with_threshold(
            analysis_file,
            threshold,
            protein_string,
            lo_string,
            coloc_type=coloc_type,
            imtype=".tif",
            blur_degree=1,
            calc_clr=False,
            aboveT=0,
        )

        savecell_string = os.path.join(
            os.path.split(analysis_file)[0],
            startstr + "colocalisation_analysis_" + threshold_str,
        )
        if isinstance(lo_analysis_AT, pl.DataFrame) and isinstance(
            lo_analysis_UT, pl.DataFrame
        ):
            above_str = "coincidence_above_" + threshold_str
            above_cc_str = "chance_coincidence_above_" + threshold_str
            niter_str = "n_iter_above_" + threshold_str
            below_str = "coincidence_below_" + threshold_str
            below_cc_str = "chance_coincidence_below_" + threshold_str
            niter_below_str = "n_iter_below_" + threshold_str

            lo_analysis = lo_analysis_AT
            lo_analysis = lo_analysis.rename({"coincidence": above_str})
            lo_analysis = lo_analysis.rename({"chance_coincidence": above_cc_str})
            lo_analysis = lo_analysis.rename({"n_iter": niter_str})

            lo_analysis = lo_analysis.with_columns(
                channelcol=lo_analysis_UT["coincidence"]
            ).rename({"channelcol": below_str})
            lo_analysis = lo_analysis.with_columns(
                channelcol=lo_analysis_UT["chance_coincidence"]
            ).rename({"channelcol": below_cc_str})
            lo_analysis = lo_analysis.with_columns(
                channelcol=lo_analysis_UT["n_iter"]
            ).rename({"channelcol": niter_below_str})

            lo_analysis = lo_analysis[
                above_str,
                above_cc_str,
                niter_str,
                below_str,
                below_cc_str,
                niter_below_str,
                "image_filename",
            ]

            lo_analysis = lo_analysis.filter(
                (lo_analysis["coincidence"] != 0)
                & (lo_analysis["chance_coincidence"] != 0)
            )

            lo_analysis.write_csv(savecell_string + "_photonthreshold.csv")
        else:
            if isinstance(lo_analysis_AT, pl.DataFrame):
                lo_analysis = lo_analysis_AT
                lo_analysis = lo_analysis.filter(
                    (lo_analysis["coincidence"] != 0)
                    & (lo_analysis["chance_coincidence"] != 0)
                )
                lo_analysis_AT.write_csv(savecell_string + "_abovephotonthreshold.csv")
            if isinstance(lo_analysis_UT, pl.DataFrame):
                lo_analysis = lo_analysis_UT
                lo_analysis = lo_analysis.filter(
                    (lo_analysis["coincidence"] != 0)
                    & (lo_analysis["chance_coincidence"] != 0)
                )
                lo_analysis_UT.write_csv(savecell_string + "_belowphotonthreshold.csv")

        if isinstance(spot_analysis_AT, pl.DataFrame):
            spot_analysis_AT.write_csv(
                analysis_file.split(".")[0]
                + "_"
                + threshold_str
                + "_abovephotonthreshold.csv"
            )

        if isinstance(spot_analysis_UT, pl.DataFrame):
            spot_analysis_UT.write_csv(
                analysis_file.split(".")[0]
                + "_"
                + threshold_str
                + "_belowphotonthreshold.csv"
            )
        return lo_analysis, spot_analysis_AT, spot_analysis_UT

    def colocalise_spots_with_threshold(
        self,
        analysis_file_1,
        analysis_file_2,
        threshold_1,
        threshold_2,
        spot_1_string,
        spot_2_string,
        imtype=".tif",
        image_size=(1200, 1200),
        blur_degree=1,
    ):
        """
        Redo colocalisation analayses of spots above a photon threshold in an
        analysis file, to spots above a second threshold in a separate analysis file.

        Args:
            analysis_1_file (str): Analysis file (channel 1) to be re-done.
            analysis_2_file (str): Analysis file (channel 2) to be re-done.
            threshold_1 (float): The photon threshold for channel 1
            threshold_2 (float): The photon threshold for channel 2
            spot_1_string (str): string of spot 1
            spot_2_string (str): string of spot 2
            imtype (str): image type
            image_size (list): original image size
            blur_degree (int): blur degree for colocalisation analysis

        Returns:
            channel_1_analysis (pl.DataFrame): channel 1 type above and below threshold
            channel_2_analysis (pl.DataFrame): channel 2 type above and below threshold
            spot_1_analysis_AT (pl.DataFrame): channel 1 type above threshold
            spot_2_analysis_AT (pl.DataFrame): channel 2 type above threshold
            spot_1_analysis_UT (pl.DataFrame): channel 1 type below threshold
            spot_2_analysis_UT (pl.DataFrame): channel 2 type below threshold
        """
        if int(threshold_1) == threshold_1:
            threshold1_str = str(int(threshold_1))
        else:
            threshold1_str = str(threshold_1).replace(".", "p")

        if int(threshold_2) == threshold_2:
            threshold2_str = str(int(threshold_2))
        else:
            threshold2_str = str(threshold_2).replace(".", "p")

        spots_1_with_intensities = pl.read_csv(analysis_file_1)
        spots_2_with_intensities = pl.read_csv(analysis_file_2)

        (
            channel_1_analysis_AT,
            channel_2_analysis_AT,
            spot_1_analysis_AT,
            spot_2_analysis_AT,
        ) = A_F.colocalise_spots_with_threshold(
            spots_1_with_intensities,
            spots_2_with_intensities,
            threshold_1,
            threshold_2,
            spot_1_string,
            spot_2_string,
            imtype=imtype,
            image_size=image_size,
            blur_degree=blur_degree,
            aboveT=1,
        )

        (
            channel_1_analysis_UT,
            channel_2_analysis_UT,
            spot_1_analysis_UT,
            spot_2_analysis_UT,
        ) = A_F.colocalise_spots_with_threshold(
            spots_1_with_intensities,
            spots_2_with_intensities,
            threshold_1,
            threshold_2,
            spot_1_string,
            spot_2_string,
            imtype=imtype,
            image_size=image_size,
            blur_degree=blur_degree,
            aboveT=0,
        )

        channel_1_analysis, channel_2_analysis = IO.save_abovebelowthresholdcoloc(
            channel_1_analysis_AT,
            channel_2_analysis_AT,
            spot_1_analysis_AT,
            spot_2_analysis_AT,
            channel_1_analysis_UT,
            channel_2_analysis_UT,
            spot_1_analysis_UT,
            spot_2_analysis_UT,
            analysis_file_1,
            analysis_file_2,
            spot_1_string,
            spot_2_string,
            threshold1_str,
            threshold2_str,
        )

        return (
            channel_1_analysis,
            channel_2_analysis,
            spot_1_analysis_AT,
            spot_2_analysis_AT,
            spot_1_analysis_UT,
            spot_2_analysis_UT,
        )

    def file_search(self, folder, string1, string2):
        """
        Search for files containing 'string1' in their names within 'folder',
        and then filter the results to include only those containing 'string2'.

        Args:
            folder (str): The directory to search for files.
            string1 (str): The first string to search for in the filenames.
            string2 (str): The second string to filter the filenames containing string1.

        Returns:
            file_list (list): A sorted list of file paths matching the search criteria.
        """
        # Get a list of all files containing 'string1' in their names within 'folder'
        file_list = [
            os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(folder)
            for f in fnmatch.filter(files, "*" + string1 + "*")
        ]
        file_list = np.sort([e for e in file_list if string2 in e])
        return file_list
