# -*- coding: utf-8 -*-
"""
This class contains functions that collect analysis routines for RASP.
jsb92, 2024/02/08
"""
import os
import fnmatch
import numpy as np
import pandas as pd
import sys
import time


module_dir = os.path.dirname(__file__)
sys.path.append(module_dir)
import IOFunctions

IO = IOFunctions.IO_Functions()
import AnalysisFunctions

A_F = AnalysisFunctions.Analysis_Functions()


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
        Initialises class.

        Args:
            defaultarea (boolean): If True, uses area default for analysis later
            defaultd (boolean): If True, uses default pixel radius for analysis later
            defaultrad (boolean): If True, uses radiality default for analysis later
            defaultflat (boolean): If True, uses flatness default for analysis later
            defaultdfocus (boolean): If True, uses differential infocus default for analysis later
            defaultintfocus (boolean): If True, uses integral infocus default for analysis later
            defaultcameraparams (boolean): If True, uses camera parameters in folder for analysis later
        """
        self = self
        if defaultfolder == None:
            self.defaultfolder = os.path.join(
                os.path.split(module_dir)[0], "default_analysis_parameters"
            )
        else:
            self.defaultfolder = defaultfolder
        if defaultarea == True:
            if os.path.isfile(os.path.join(self.defaultfolder, "areathres.json")):
                data = IO.load_json(os.path.join(self.defaultfolder, "areathres.json"))
                self.areathres = float(data["areathres"])
            else:
                self.areathres = 30.0

        if defaultd == True:
            if os.path.isfile(os.path.join(self.defaultfolder, "areathres.json")):
                data = IO.load_json(os.path.join(self.defaultfolder, "areathres.json"))
                self.d = int(data["d"])
            else:
                self.d = int(2.0)

        if defaultrad == True:
            if os.path.isfile(os.path.join(self.defaultfolder, "rad_neg.json")):
                data = IO.load_json(os.path.join(self.defaultfolder, "rad_neg.json"))
                self.integratedGrad = float(data["integratedGrad"])
            else:
                self.integratedGrad = 0.0

        if defaultflat == True:
            if os.path.isfile(os.path.join(self.defaultfolder, "rad_neg.json")):
                data = IO.load_json(os.path.join(self.defaultfolder, "rad_neg.json"))
                self.flatness = float(data["flatness"])
            else:
                self.flatness = 1.0

        if defaultdfocus == True:
            if os.path.isfile(os.path.join(self.defaultfolder, "infocus.json")):
                data = IO.load_json(os.path.join(self.defaultfolder, "infocus.json"))
                self.focus_score_diff = float(data["focus_score_diff"])
            else:
                self.focus_score_diff = 0.2

        if defaultintfocus == True:
            if os.path.isfile(os.path.join(self.defaultfolder, "infocus.json")):
                data = IO.load_json(os.path.join(self.defaultfolder, "infocus.json"))
                self.focus_score_int = float(data["focus_score_int"])
            else:
                self.focus_score_int = 390.0

        if defaultcellparams == True:
            if os.path.isfile(
                os.path.join(self.defaultfolder, "default_cell_params.json")
            ):
                data = IO.load_json(
                    os.path.join(self.defaultfolder, "default_cell_params.json")
                )
                self.cell_sigma1 = float(data["sigma1"])
                self.cell_sigma2 = float(data["sigma2"])
                self.cell_threshold1 = float(data["threshold1"])
                self.cell_threshold2 = float(data["threshold1"])
            else:
                self.cell_sigma1 = 2.0
                self.cell_sigma2 = 40.0
                self.cell_threshold1 = 200.0
                self.cell_threshold2 = 200.0

        if defaultcameraparams == True:
            if os.path.isfile(os.path.join(self.defaultfolder, "camera_params.json")):
                data = IO.load_json(
                    os.path.join(self.defaultfolder, "camera_params.json")
                )
                self.QE = float(data["QE"])
            else:
                self.QE = 0.95

            if os.path.isfile(os.path.join(self.defaultfolder, "gain_map.tif")):
                self.gain_map = IO.read_tiff(
                    os.path.join(self.defaultfolder, "gain_map.tif")
                )
            else:
                self.gain_map = 1.0

            if os.path.isfile(os.path.join(self.defaultfolder, "offset_map.tif")):
                self.offset_map = IO.read_tiff(
                    os.path.join(self.defaultfolder, "offset_map.tif")
                )
            else:
                self.offset_map = 0.0

            if type(self.gain_map) is not float and type(self.offset_map) is not float:
                if self.gain_map.shape != self.offset_map.shape:
                    print(
                        "Gain and Offset maps are not the same shapes. Defaulting to default gain (1) and offset (0) parameters."
                    )
                    self.gain_map = 1.0
                    self.offset_map = 0.0
        return

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
        self, folder, imtype=".tif", gsigma=1.4, rwave=2.0, accepted_ratio=1
    ):
        """
        Calibrates radility parameters. Given a folder of negative controls,
        analyses them and saves the radiality parameter to the .json file, as
        well as writing it to the current class radiality and flatness values

        Args:
            folder (string): Folder containing negative control tifs
            imtype (string): Type of images being analysed, default tif
            gsigma (float): gaussian blurring parameter (default 1.4)
            rwave (float): Ricker wavelent sigma (default 2.)
            accepted_ratio (float): Percentage accepted of false positives
        """
        files_list = os.listdir(folder)
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

        bins_r1 = A_F.bincalculator(r1_neg)
        bins_r2 = A_F.bincalculator(r2_neg)
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

    def calibrate_area(
        self, folder, imtype=".tif", gsigma=1.4, rwave=2.0, large_thres=10000.0
    ):
        """
        Calibrates area threshold. Given a folder of bead images,
        analyses them and saves the radiality parameter to the .json file, as
        well as writing it to the current class radiality and flatness values

        Args:
            folder (string): Folder containing bead (bright) control tifs
            imtype (string): Type of images being analysed, default tif
            gisgma (float): gaussian blurring parameter (default 1.4)
            rwave (float): Ricker wavelent sigma (default 2.)
        """
        files_list = os.listdir(folder)
        files = np.sort([e for e in files_list if imtype in e])

        k1, k2 = A_F.create_kernel(gsigma, rwave)  # create image processing kernels
        accepted_ratio = 95.0
        # perc. of CDF we'll use
        areathres = 1000.0  # arbitrarily high area threshold for this calibration
        thres = 0.05  # threshold is 0.05
        rdl = [self.flatness, self.integratedGrad, 0.0]

        start = time.time()

        for i in np.arange(len(files)):
            file_path = os.path.join(folder, files[i])
            image = IO.read_tiff(file_path)
            if len(image.shape) < 3:
                dl_mask, centroids, radiality, large_mask = A_F.compute_image_props(
                    image, k1, k2, thres, large_thres, areathres, rdl, self.d
                )
                pixel_index_list, areas, centroids = A_F.calculate_region_properties(
                    dl_mask
                )
                HWHMarray = A_F.Gauss2DFitting(image, pixel_index_list)
                if i == 0:
                    a_neg = areas
                    HWHM = HWHMarray
                else:
                    a_neg = np.hstack([a_neg, areas])
                    HWHM = np.hstack([HWHM, HWHMarray])
            else:
                dl_mask, centroids, radiality, large_mask = A_F.compute_image_props(
                    image,
                    k1,
                    k2,
                    thres,
                    large_thres,
                    areathres,
                    rdl,
                    self.d,
                    z_planes=np.arange(image.shape[2]),
                )
                for j in np.arange(image.shape[2]):
                    pixel_index_list, areas, centroids = (
                        A_F.calculate_region_properties(dl_mask[:, :, j])
                    )
                    HWHMarray = A_F.Gauss2DFitting(image[:, :, j], pixel_index_list)
                    if (i == 0) and (j == 0):
                        a_neg = areas
                        HWHM = HWHMarray
                    else:
                        a_neg = np.hstack([a_neg, areas])
                        HWHM = np.hstack([HWHM, HWHMarray])
            print(
                "Analysed image file {}/{}    Time elapsed: {:.3f} s".format(
                    i + 1, len(files), time.time() - start
                ),
                end="\r",
                flush=True,
            )

        HWHM = A_F.rejectoutliers(HWHM)
        area_thresh = int(np.ceil(np.percentile(a_neg, accepted_ratio)))
        pixel_d = int(np.round(np.mean(HWHM)))

        to_save = {"areathres": area_thresh, "d": pixel_d}

        IO.make_directory(self.defaultfolder)
        IO.save_as_json(to_save, os.path.join(self.defaultfolder, "areathres.json"))
        self.areathres = area_thresh
        self.d = pixel_d

        print(
            "Area threshold using beads"
            + " images in "
            + str(folder)
            + ". New area threshold is "
            + str(np.around(area_thresh, 2))
            + " new radiality radius calibrated and is "
            + str(int(self.d))
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

        axs[0].ecdf(a_neg, color="k")
        xlim0, xlim1 = axs[0].get_xlim()[0], axs[0].get_xlim()[1]
        axs[0].hlines(
            accepted_ratio / 100.0, xlim0, xlim1, color="k", label="threshold", ls="--"
        )
        axs[0].set_ylim([0, 1])
        axs[0].set_xlim([xlim0, xlim1])
        axs[0].set_xlabel("puncta area (pixel)")
        axs[0].set_ylabel("probability ")
        axs[0].legend(loc="lower right", frameon=False)

        axs[1] = plots.histogram_plot(axs[1], HWHM, bins=A_F.bincalculator(HWHM))
        ylim0, ylim1 = axs[1].get_ylim()[0], axs[1].get_ylim()[1]
        axs[1].vlines(
            np.mean(HWHM), ylim0, ylim1, color="k", label="threshold", ls="--"
        )
        axs[1].set_xlabel("HWHM (pixel)")
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

        """
        all_files = self.file_search(
            folder, protein_string, imtype
        )  # first get all files in any subfolders

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
        )

        start = time.time()

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
                img = IO.read_tiff_tophotons(
                    os.path.join(folder, files[i]),
                    QE=self.QE,
                    gain_map=self.gain_map,
                    offset_map=self.offset_map,
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
                        to_save, to_save_largeobjects = A_F.compute_spot_props(
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
                        to_save, to_save_largeobjects, to_save_cell, cell_mask = (
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
                                analyse_clr=analyse_clr,
                            )
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
                to_save, to_save_largeobjects = A_F.compute_spot_props(
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
                to_save, to_save_largeobjects, to_save_cell, cell_mask = (
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
            if len(to_save[to_save.z == val].sum_intensity_in_photons.values) > 1:
                z_to_plot[i] = val
        z_to_plot = z_to_plot[z_to_plot >= 0]

        if cell_analysis == False:

            for i in enumerate(z_to_plot):
                fig, axs = plots.two_column_plot(nrows=1, ncolumns=2, widthratio=[1, 1])
                xpositions = to_save[to_save.z == i[1]].x.values
                ypositions = to_save[to_save.z == i[1]].y.values
                xpositions_large = to_save_largeobjects[
                    to_save_largeobjects.z == i[1]
                ].x.values
                ypositions_large = to_save_largeobjects[
                    to_save_largeobjects.z == i[1]
                ].y.values
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
                xpositions = to_save[to_save.z == i[1]].x.values
                ypositions = to_save[to_save.z == i[1]].y.values
                xpositions_large = to_save_largeobjects[
                    to_save_largeobjects.z == i[1]
                ].x.values
                ypositions_large = to_save_largeobjects[
                    to_save_largeobjects.z == i[1]
                ].y.values
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
        n_iter=5,
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
        analysis_data = pd.read_csv(analysis_file)
        analysis_data = analysis_data[
            analysis_data.sum_intensity_in_photons > threshold
        ]
        analysis_directory = os.path.split(analysis_file)[0]
        if out_cell == True:
            analysis_data = analysis_data[analysis_data.incell == 0]

        start = time.time()

        if len(analysis_data) > 0:
            if int(threshold) == threshold:
                thesholdsavestr = str(threshold)
            else:
                thesholdsavestr = str(threshold).replace(".", "p")

            files = np.unique(analysis_data.image_filename.values)
            z_planes = {}  # make dict where z planes will be stored
            for i, file in enumerate(files):
                z_planes[file] = np.unique(
                    analysis_data[analysis_data.image_filename == file].z.values
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
                subset = analysis_data[analysis_data.image_filename == file]
                image_size = cell_mask[:, :, 0].shape
                for z in zs:
                    uid = str(file) + "___" + str(z)
                    x = subset[subset.z == z].x.values
                    y = subset[subset.z == z].y.values
                    coordinates_spot = np.vstack([x, y]).T
                    xm, ym = np.where(cell_mask[:, :, int(z) - 1])
                    coordinates_mask = np.vstack([xm, ym]).T

                    g_r[uid], radii = A_F.spot_to_mask_rdf(
                        coordinates_spot,
                        coordinates_mask,
                        out_cell=out_cell,
                        pixel_size=pixel_size,
                        dr=dr,
                        min_radius=dr,
                        max_radius=np.divide(
                            np.multiply(np.max(image_size), pixel_size), 2
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

            rdf = pd.DataFrame(
                data=np.vstack([g_r_mean, g_r_std]).T,
                index=radii,
                columns=["g_r_mean", "g_r_std"],
            )
            to_save_name = os.path.join(
                os.path.split(analysis_file)[0],
                "spot_to_mask_"
                + cell_string
                + "_threshold_"
                + thesholdsavestr
                + "_rdf.csv",
            )
            rdf.to_csv(to_save_name)
            return rdf

    def calculate_two_puncta_channels_rdf_with_thresholds(
        self,
        analysis_file_p1,
        analysis_file_p2,
        threshold_p1,
        threshold_p2,
        pixel_size=0.11,
        dr=1.0,
        n_iter=10,
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
            rdf (pd.DataArray): pandas datarray of the rdf
        """
        analysis_data_p1 = pd.read_csv(analysis_file_p1)
        analysis_data_p1 = analysis_data_p1[
            analysis_data_p1.sum_intensity_in_photons > threshold_p1
        ]

        analysis_data_p2 = pd.read_csv(analysis_file_p2)
        analysis_data_p2 = analysis_data_p2[
            analysis_data_p2.sum_intensity_in_photons > threshold_p2
        ]

        start = time.time()

        if (len(analysis_data_p1) > 0) and (len(analysis_data_p2) > 0):
            if int(threshold_p1) == threshold_p1:
                theshold_p1_savestr = str(threshold_p1)
            else:
                theshold_p1_savestr = str(threshold_p1).replace(".", "p")

            if int(threshold_p2) == threshold_p2:
                theshold_p2_savestr = str(threshold_p2)
            else:
                theshold_p2_savestr = str(threshold_p2).replace(".", "p")

            files_p1 = np.unique(
                [
                    file.split(imtype)[0].split(protein_string_1)[0]
                    for file in analysis_data_p1.image_filename.values
                ]
            )
            files_p2 = np.unique(
                [
                    file.split(imtype)[0].split(protein_string_2)[0]
                    for file in analysis_data_p2.image_filename.values
                ]
            )
            files = np.unique(np.hstack([files_p1, files_p2]))

            z_planes = {}  # make dict where z planes will be stored
            for i, file in enumerate(files):
                zp_f1 = np.unique(
                    analysis_data_p1[
                        analysis_data_p1.image_filename
                        == file + str(protein_string_1) + imtype
                    ].z.values
                )
                zp_f2 = np.unique(
                    analysis_data_p2[
                        analysis_data_p2.image_filename
                        == file + str(protein_string_2) + imtype
                    ].z.values
                )
                z_planes[file] = np.unique(np.hstack([zp_f1, zp_f2]))

            g_r = {}

            for i, file in enumerate(files):
                zs = z_planes[file]
                subset_p1 = analysis_data_p1[
                    analysis_data_p1.image_filename
                    == file + str(protein_string_1) + imtype
                ]
                subset_p2 = analysis_data_p1[
                    analysis_data_p2.image_filename
                    == file + str(protein_string_2) + imtype
                ]
                for z in zs:
                    uid = str(file) + "___" + str(z)
                    x_p1 = subset_p1[subset_p1.z == z].x.values
                    y_p1 = subset_p1[subset_p1.z == z].y.values
                    x_p2 = subset_p2[subset_p2.z == z].x.values
                    y_p2 = subset_p2[subset_p2.z == z].y.values
                    coordinates_p1_spot = np.asarray(np.vstack([x_p1, y_p1]).T, dtype=int)
                    coordinates_p2_spot = np.asarray(np.vstack([x_p2, y_p2]).T, dtype=int)

                    g_r[uid], radii = A_F.spot_to_mask_rdf(
                        coordinates_p1_spot,
                        coordinates_p2_spot,
                        out_cell=False,
                        pixel_size=pixel_size,
                        dr=dr,
                        min_radius=dr,
                        max_radius=np.divide(np.multiply(1200, pixel_size), 2),
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

            rdf = pd.DataFrame(
                data=np.vstack([g_r_mean, g_r_std]).T,
                index=radii,
                columns=["g_r_mean", "g_r_std"],
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
                + theshold_p2_savestr
                + "_rdf.csv",
            )
            rdf.to_csv(to_save_name)
            return rdf

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
        analysis_data = pd.read_csv(analysis_file)
        analysis_data = analysis_data[
            analysis_data.sum_intensity_in_photons > threshold
        ]
        if len(analysis_data) > 0:
            if int(threshold) == threshold:
                thesholdsavestr = str(threshold)
            else:
                thesholdsavestr = str(threshold).replace(".", "p")

            files = np.unique(analysis_data.image_filename.values)
            z_planes = {}  # make dict where z planes will be stored
            for i, file in enumerate(files):
                z_planes[file] = np.unique(
                    analysis_data[analysis_data.image_filename == file].z.values
                )

            g_r = {}
            radii = {}

            start = time.time()

            for i, file in enumerate(files):
                zs = z_planes[file]
                subset = analysis_data[analysis_data.image_filename == file]
                for z in zs:
                    uid = str(file) + "___" + str(z)
                    x = subset[subset.z == z].x.values
                    y = subset[subset.z == z].y.values
                    coordinates = np.vstack([x, y]).T
                    g_r[uid], radii[uid] = A_F.spot_to_spot_rdf(
                        coordinates, pixel_size=pixel_size, dr=dr
                    )
                print(
                    "Computing RDF     File {}/{}    Time elapsed: {:.3f} s".format(
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

            rdf = pd.DataFrame(
                data=np.vstack([g_r_mean, g_r_std]).T,
                index=radii_overall,
                columns=["g_r_mean", "g_r_std"],
            )
            to_save_name = os.path.join(
                os.path.split(analysis_file)[0],
                "spot_to_spot_threshold_" + thesholdsavestr + "_rdf.csv",
            )
            rdf.to_csv(to_save_name)
            return rdf

    def colocalise_with_threshold(
        self,
        analysis_file,
        threshold,
        protein_string,
        cell_string,
        imtype=".tif",
        blur_degree=1,
        calc_clr=False,
    ):
        """
        Redo colocalisation analayses of spots above a photon threshold in an
        analysis file.

        Args:
            analysis_file (str): The analysis file to be re-done.
            threshold (float): The photon threshold
            protein_string (str): string of analysed protein
            cell_string (str): string of cell to analyse
            imtype (str): image type
            blur_degree (int): blur degree for colocalisation analysis
            calc_clr (boolean): Calculate the clr, yes/no.
        """
        if int(threshold) == threshold:
            threshold_str = str(int(threshold))
        else:
            threshold_str = str(threshold).replace(".", "p")

        spots_with_intensities = pd.read_csv(analysis_file)
        spots_with_intensities = spots_with_intensities[
            spots_with_intensities.sum_intensity_in_photons > threshold
        ].reset_index(drop=True)
        analysis_directory = os.path.split(analysis_file)[0]
        image_filenames = np.unique(spots_with_intensities.image_filename.values)

        if calc_clr == False:
            columns = ["coincidence", "chance_coincidence", "n_iter", "image_filename"]
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

        for i, image in enumerate(image_filenames):
            cell_mask = IO.read_tiff(
                os.path.join(
                    analysis_directory,
                    os.path.split(image.split(imtype)[0])[-1].split(protein_string)[0]
                    + str(cell_string)
                    + "_cellMask.tiff",
                )
            )
            image_size = cell_mask.shape[:-1]
            image_file = spots_with_intensities[
                spots_with_intensities.image_filename == image
            ].reset_index(drop=True)
            z_planes = np.unique(image_file.z.values)

            dataarray = np.zeros([len(z_planes), len(columns)], dtype="object")

            temp_pd = pd.DataFrame(data=dataarray, columns=columns)

            for j, z_plane in enumerate(z_planes):
                xcoords = image_file[image_file.z == z_plane].x.values
                ycoords = image_file[image_file.z == z_plane].y.values
                mask = cell_mask[:, :, int(z_plane) - 1]
                centroids = np.asarray(np.vstack([xcoords, ycoords]), dtype=int).T
                mask_indices, spot_indices = A_F.generate_mask_and_spot_indices(
                    mask, centroids, image_size
                )
                if calc_clr == False:
                    (
                        temp_pd.values[j, 0],
                        temp_pd.values[j, 1],
                        raw_colocalisation,
                        temp_pd.values[j, 2],
                    ) = A_F.calculate_spot_to_mask_coincidence(
                        spot_indices,
                        mask_indices,
                        image_size,
                        blur_degree=blur_degree,
                    )
                else:
                    (
                        temp_pd.values[j, 0],
                        temp_pd.values[j, 1],
                        temp_pd.values[j, 2],
                        temp_pd.values[j, 3],
                        temp_pd.values[j, 4],
                        temp_pd.values[j, 5],
                        raw_colocalisation,
                        temp_pd.values[j, 6],
                    ) = A_F.calculate_spot_colocalisation_likelihood_ratio(
                        spot_indices, mask_indices, image_size, blur_degree=1
                    )
                if j == 0:
                    rc = raw_colocalisation
                else:
                    rc = np.hstack([rc, raw_colocalisation])
            temp_pd["image_filename"] = np.full_like(z_planes, image, dtype="object")
            image_file["incell"] = rc * 1
            if i == 0:
                cell_analysis = temp_pd
                spot_analysis = image_file
            else:
                cell_analysis = pd.concat([cell_analysis, temp_pd]).reset_index(
                    drop=True
                )
                spot_analysis = pd.concat([spot_analysis, image_file]).reset_index(
                    drop=True
                )

        cell_analysis.to_csv(
            os.path.join(
                analysis_directory,
                "cell_colocalisation_analysis_"
                + threshold_str
                + "_photonthreshold.csv",
            )
        )
        spot_analysis.to_csv(
            analysis_file.split(".")[0] + "_" + threshold_str + "_photonthreshold.csv"
        )
        return cell_analysis, spot_analysis

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
        """
        if int(threshold_1) == threshold_1:
            threshold1_str = str(int(threshold_1))
        else:
            threshold1_str = str(threshold_1).replace(".", "p")

        if int(threshold_2) == threshold_2:
            threshold2_str = str(int(threshold_2))
        else:
            threshold2_str = str(threshold_2).replace(".", "p")

        spots_1_with_intensities = pd.read_csv(analysis_file_1)
        spots_1_with_intensities = spots_1_with_intensities[
            spots_1_with_intensities.sum_intensity_in_photons > threshold_1
        ].reset_index(drop=True)

        spots_2_with_intensities = pd.read_csv(analysis_file_2)
        spots_2_with_intensities = spots_2_with_intensities[
            spots_2_with_intensities.sum_intensity_in_photons > threshold_2
        ].reset_index(drop=True)

        image_1_filenames = np.unique(spots_1_with_intensities.image_filename.values)
        image_2_filenames = np.unique(spots_2_with_intensities.image_filename.values)

        overall_1_filenames = [
            i.split(imtype)[0].split(spot_1_string)[0] for i in image_1_filenames
        ]
        overall_2_filenames = [
            i.split(imtype)[0].split(spot_2_string)[0] for i in image_2_filenames
        ]
        overall_filenames = np.unique(
            np.hstack([overall_1_filenames, overall_2_filenames])
        )

        columns = ["coincidence", "chance_coincidence", "image_filename"]

        for i, image in enumerate(overall_filenames):
            image_1_file = spots_1_with_intensities[
                spots_1_with_intensities.image_filename
                == image + spot_1_string + imtype
            ].reset_index(drop=True)

            image_2_file = spots_2_with_intensities[
                spots_2_with_intensities.image_filename
                == image + spot_2_string + imtype
            ].reset_index(drop=True)
            if (len(image_1_file) > 0) & (len(image_2_file) > 0):
                z_planes = np.intersect1d(image_1_file.z.values, image_2_file.z.values)

                dataarray_1 = np.zeros([len(z_planes), len(columns)], dtype="object")
                dataarray_2 = np.zeros([len(z_planes), len(columns)], dtype="object")

                temp_1_pd = pd.DataFrame(data=dataarray_1, columns=columns)
                temp_2_pd = pd.DataFrame(data=dataarray_2, columns=columns)

                for j, z_plane in enumerate(z_planes):
                    x_1_coords = image_1_file[image_1_file.z == z_plane].x.values
                    y_1_coords = image_1_file[image_1_file.z == z_plane].y.values

                    x_2_coords = image_2_file[image_2_file.z == z_plane].x.values
                    y_2_coords = image_2_file[image_2_file.z == z_plane].y.values

                    centroids1 = np.asarray(
                        np.vstack([x_1_coords, y_1_coords]), dtype=int
                    ).T
                    centroids2 = np.asarray(
                        np.vstack([x_2_coords, y_2_coords]), dtype=int
                    ).T

                    spot_1_indices, spot_2_indices = A_F.generate_spot_and_spot_indices(
                        centroids1, centroids2, image_size
                    )
                    (
                        temp_1_pd.values[j, 0],
                        temp_1_pd.values[j, 1],
                        raw_1_coincidence,
                    ) = A_F.calculate_spot_to_spot_coincidence(
                        spot_1_indices,
                        spot_2_indices,
                        image_size,
                        blur_degree=blur_degree,
                    )

                    (
                        temp_2_pd.values[j, 0],
                        temp_2_pd.values[j, 1],
                        raw_2_coincidence,
                    ) = A_F.calculate_spot_to_spot_coincidence(
                        spot_2_indices,
                        spot_1_indices,
                        image_size,
                        blur_degree=blur_degree,
                    )
                    if j == 0:
                        rc1 = raw_1_coincidence
                        rc2 = raw_2_coincidence
                    else:
                        rc1 = np.hstack([rc1, raw_1_coincidence])
                        rc2 = np.hstack([rc2, raw_2_coincidence])
                image_1_file["coincidence_with_channel_" + spot_2_string] = rc1
                image_2_file["coincidence_with_channel_" + spot_1_string] = rc2
                temp_1_pd["image_filename"] = np.full_like(
                    z_planes, image + spot_1_string + imtype, dtype="object"
                )
                temp_2_pd["image_filename"] = np.full_like(
                    z_planes, image + spot_2_string + imtype, dtype="object"
                )

                if i == 0:
                    plane_1_analysis = temp_1_pd
                    plane_2_analysis = temp_2_pd
                    spot_1_analysis = image_1_file
                    spot_2_analysis = image_2_file
                else:
                    plane_1_analysis = pd.concat(
                        [plane_1_analysis, temp_1_pd]
                    ).reset_index(drop=True)
                    plane_2_analysis = pd.concat(
                        [plane_2_analysis, temp_2_pd]
                    ).reset_index(drop=True)
                    spot_1_analysis = pd.concat([spot_1_analysis, image_1_file])
                    spot_2_analysis = pd.concat([spot_2_analysis, image_2_file])
            plane_1_analysis.to_csv(
                analysis_file_1.split(".")[0]
                + "_colocalisationwith_"
                + spot_2_string
                + "_"
                + threshold1_str
                + "_"
                + spot_1_string
                + "_photonthreshold_"
                + threshold2_str
                + "_"
                + spot_2_string
                + "_photonthreshold.csv"
            )
            plane_2_analysis.to_csv(
                analysis_file_2.split(".")[0]
                + "_colocalisationwith_"
                + spot_1_string
                + "_"
                + threshold2_str
                + "_"
                + spot_2_string
                + "_photonthreshold_"
                + threshold1_str
                + "_"
                + spot_1_string
                + "_photonthreshold.csv"
            )
            spot_1_analysis.to_csv(
                analysis_file_1.split(".")[0]
                + "_rawcolocalisationwith_"
                + spot_2_string
                + "_"
                + threshold1_str
                + "_"
                + spot_1_string
                + "_photonthreshold_"
                + threshold2_str
                + "_"
                + spot_2_string
                + "_photonthreshold.csv"
            )
            spot_2_analysis.to_csv(
                analysis_file_2.split(".")[0]
                + "_rawcolocalisationwith_"
                + spot_1_string
                + "_"
                + threshold2_str
                + "_"
                + spot_2_string
                + "_photonthreshold_"
                + threshold1_str
                + "_"
                + spot_1_string
                + "_photonthreshold.csv"
            )
        return (
            plane_1_analysis,
            plane_2_analysis,
            spot_1_analysis,
            spot_2_analysis,
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
