#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class related to making figure-quality plots.

Probably best to set your default sans-serif font to Helvetica before you make
figures: https://fowlerlab.org/2019/01/03/changing-the-sans-serif-font-to-helvetica/

The maximum published width for a one-column
figure is 3.33 inches (240 pt). The maximum width for a two-column
figure is 6.69 inches (17 cm). The maximum depth of figures should
be 8 ¼ in. (21.1 cm).

panel labels are 8 point font, ticks are 7 point font,
annotations and legends are 6 point font.

"""
import matplotlib  # requires 3.8.0
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, poster=False):
        self.poster = poster
        self = self
        return

    def one_column_plot(self, npanels=1, ratios=None, height=None, width=None):
        """
        Creates a one-column width figure with specified configurations.

        Args:
            npanels (int): Number of panels (rows).
            ratios (list): List of height ratios for the panels.
            height (float): Override for figure height.
            width (float): Override for figure width.

        Returns:
            fig (Figure): Figure object.
            axs (Axes or array of Axes): Axes object(s).
        """
        # Validate ratios
        ratios = ratios or [1] * npanels
        if len(ratios) != npanels:
            raise ValueError("Mismatch between number of panels and ratios length")

        # Configure font size and line width
        fontsz = 12 if self.poster else 7
        lw = 1 if self.poster else 0.5

        # Set default sizes
        xsize = 3.33  # Default one-column width in inches
        ysize = height or npanels * 3.5  # Default height proportional to panels
        if width is not None:
            xsize = min(width, 3.33)
        if height is not None:
            ysize = min(height, 8.25)  # Max allowable height is 8.25 inches
        else:
            ysize = min(npanels * 3.5, 8.25)

        # Apply global plotting configurations
        plt.rcParams.update(
            {
                "figure.figsize": [xsize, ysize],
                "font.size": fontsz,
                "svg.fonttype": "none",
                "axes.linewidth": lw,
            }
        )
        matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

        # Create figure and axes
        fig, axs = plt.subplots(npanels, 1, gridspec_kw={"height_ratios": ratios})

        # Ensure axs is iterable
        axs = np.atleast_1d(axs)

        # Configure tick parameters for each axis
        for ax in axs:
            ax.xaxis.set_tick_params(width=lw, length=lw * 4)
            ax.yaxis.set_tick_params(width=lw, length=lw * 4)
            ax.tick_params(axis="both", pad=1.2)

        return fig, axs

    def two_column_plot(
        self,
        nrows=1,
        ncolumns=1,
        heightratio=None,
        widthratio=None,
        height=0,
        big=False,
    ):
        """
        Creates a two-column width figure with specified configurations.

        Args:
            nrows (int): Number of rows.
            ncolumns (int): Number of columns.
            heightratio (list): List of heights, length must match nrows.
            widthratio (list): List of widths, length must match ncolumns.
            height (float): Overridden height of the figure.
            big (bool): If True, uses larger font sizes.

        Returns:
            fig (Figure): Figure object.
            axs (Axes or array of Axes): Axes object(s).
        """
        # Validate ratios
        heightratio = heightratio or [1] * nrows
        widthratio = widthratio or [1] * ncolumns
        if len(heightratio) != nrows or len(widthratio) != ncolumns:
            raise ValueError("Mismatch between ratios and number of rows/columns")

        # Font size and line width configurations
        fontsz = 12 if self.poster else 7
        lw = 1
        xsize = 5 * ncolumns if big else 6.69  # Adjust column size
        ysize = height if height > 0 else (5 * nrows if big else 3 * nrows)

        # Apply global plotting configurations
        plt.rcParams.update(
            {
                "figure.figsize": [xsize, ysize],
                "font.size": fontsz,
                "svg.fonttype": "none",
                "axes.linewidth": lw,
            }
        )
        matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

        # Create figure and axes
        fig, axs = plt.subplots(
            nrows,
            ncolumns,
            gridspec_kw={"height_ratios": heightratio, "width_ratios": widthratio},
        )

        # Ensure axs is always iterable
        axs = np.array(axs).reshape(-1) if isinstance(axs, np.ndarray) else [axs]

        # Configure tick parameters for each axis
        for ax in axs:
            ax.xaxis.set_tick_params(width=lw, length=lw * 4)
            ax.yaxis.set_tick_params(width=lw, length=lw * 4)
            ax.tick_params(axis="both", pad=1.2)

        axs = axs[0] if isinstance(axs, list) else axs[:]
        return fig, axs

    def line_plot(
        self,
        axs,
        x,
        y,
        xlim=None,
        ylim=None,
        color="k",
        lw=0.75,
        label="",
        xaxislabel="x axis",
        yaxislabel="y axis",
        ls="-",
    ):
        """
        Creates a line plot with specified parameters.

        Args:
            axs (Axes): Matplotlib axis object to plot on.
            x (array-like): x data.
            y (array-like): y data.
            xlim (tuple, optional): x-axis limits (min, max). Defaults to computed range.
            ylim (tuple, optional): y-axis limits (min, max). Defaults to computed range.
            color (str): Line color. Defaults to 'k' (black).
            lw (float): Line width. Defaults to 0.75.
            label (str): Plot label. Defaults to ''.
            xaxislabel (str): Label for the x-axis. Defaults to 'x axis'.
            yaxislabel (str): Label for the y-axis. Defaults to 'y axis'.
            ls (str): Line style. Defaults to '-' (solid line).

        Returns:
            Axes: Updated axis object.
        """
        # Determine font size based on mode
        fontsz = 15 if self.poster else 8

        # Set default limits if not provided
        xlim = xlim or (np.min(x), np.max(x))
        ylim = ylim or (np.min(y), np.max(y))

        # Plot the line
        axs.plot(x, y, lw=lw, color=color, label=label, ls=ls)

        # Configure axis limits and labels
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        axs.set_ylabel(yaxislabel, fontsize=fontsz)

        # Add grid
        axs.grid(True, which="both", ls="--", color="gray", lw=0.25, alpha=0.25)

        return axs

    def cell_punctum_analysis_plot(
        self,
        folder_tosave,
        cell_punctum_analysis_file,
        spot_analysis_file,
        lower_pcl,
        upper_pcl=np.inf,
        z_project=True,
        protein_string="C1",
        cell_string="C0",
    ):
        """
        Plots cells in a specified puncta_cell_likelihood range using analysis data.

        Args:
            folder_tosave (str): Directory to save output figures.
            cell_punctum_analysis_file (str): Path to cell punctum analysis file.
            spot_analysis_file (str): Path to spot analysis file.
            lower_pcl (float): Lower bound for puncta_cell_likelihood.
            upper_pcl (float): Upper bound for puncta_cell_likelihood.
            z_project (bool): Whether to z-project the cell mask.
            protein_string (str): Protein identifier in file names.
            cell_string (str): Cell identifier in file names.
        """
        import os
        import polars as pl
        import IOFunctions, AnalysisFunctions

        # Helper function for formatting likelihood ranges
        def format_pcl_limit(pcl):
            if pcl == np.inf:
                return "none"
            return (
                str(int(pcl))
                if pcl.is_integer()
                else str(np.around(pcl)).replace(".", "p")
            )

        # Parse lower and upper puncta cell likelihoods
        lower_strtosave = format_pcl_limit(lower_pcl)
        upper_strtosave = format_pcl_limit(upper_pcl)

        # Extract cell size thresholds from file name
        def parse_cell_size_thresholds(filename, key, default=np.inf):
            if f"_{key}_" in filename:
                value = filename.split(f"_{key}_")[-1].split("_")[0].replace("p", ".")
                return float(value)
            return default

        lower_cell_size_threshold = parse_cell_size_thresholds(
            cell_punctum_analysis_file, "mincellsize", 0.0
        )
        upper_cell_size_threshold = parse_cell_size_thresholds(
            cell_punctum_analysis_file, "maxcellsize", np.inf
        )

        # Initialize IO and analysis functions
        IO = IOFunctions.IO_Functions()
        A_F = AnalysisFunctions.Analysis_Functions()

        # Load data
        puncta_analysis = pl.read_csv(spot_analysis_file)
        cell_punctum_analysis = pl.read_csv(cell_punctum_analysis_file)

        # Filter data based on puncta_cell_likelihood
        filtered_cells = cell_punctum_analysis.filter(
            (pl.col("puncta_cell_likelihood") > lower_pcl)
            & (pl.col("puncta_cell_likelihood") < upper_pcl)
        )

        # Process each unique file
        for file in filtered_cells["image_filename"].unique():
            cell_analysis = filtered_cells.filter(pl.col("image_filename") == file)
            puncta = puncta_analysis.filter(pl.col("image_filename") == file)

            # Derive file paths
            file_dir, file_name = os.path.split(file)
            analysis_dir = os.path.join(
                "/", os.path.join(*file_dir.split("/")[:-1]), "_analysis"
            )
            cell_mask_path = os.path.join(
                analysis_dir,
                f"{file_name.split('.')[0].split(protein_string)[0]}{cell_string}_cellMask.tiff",
            )
            raw_cell_path = os.path.join(
                file.split(protein_string)[0] + f"{cell_string}.tiff"
            )

            # Read and preprocess data
            cell_mask = IO.read_tiff(cell_mask_path)
            raw_cell_zproject = np.sum(IO.read_tiff(raw_cell_path), axis=-1)

            # Create labeled cell masks
            cell_mask_toplot_analysis, thresholded_cell_mask = (
                A_F.create_labelled_cellmasks(
                    cell_analysis,
                    puncta,
                    cell_mask,
                    lower_cell_size_threshold=lower_cell_size_threshold,
                    upper_cell_size_threshold=upper_cell_size_threshold,
                    z_project=z_project,
                    parameter="puncta_cell_likelihood",
                )
            )

            # Plot data
            fig, axs = self.two_column_plot(ncolumns=3, widthratio=[1, 1, 1])

            self.image_plot(
                axs=axs[0],
                data=raw_cell_zproject,
                cbarlabel="Intensity",
                mask=cell_mask,
                maskcolor="red",
            )
            self.image_scatter_plot(
                axs=axs[1],
                xdata=puncta["x"],
                ydata=puncta["y"],
                data=thresholded_cell_mask,
                cbarlabel="Cell Mask",
                s=0.01,
                lws=0.25,
            )
            self.image_plot(
                axs[axs[2]],
                data=cell_mask_toplot_analysis,
                cbarlabel="Puncta-to-Cell Likelihood",
            )

            # Save figure
            save_filename = os.path.join(
                folder_tosave,
                f"{file_name.split(protein_string)[0]}_lower_likelihood_limit_{lower_strtosave}_"
                f"upper_likelihood_limit_{upper_strtosave}_cell_likelihood_figure.svg",
            )
            plt.tight_layout()
            plt.savefig(save_filename, dpi=600, format="svg")
            plt.close("all")

    def histogram_plot(
        self,
        axs,
        data,
        bins,
        xlim=None,
        ylim=None,
        histcolor="gray",
        xaxislabel="x axis",
        alpha=1.0,
        histtype="bar",
        density=True,
        label="",
    ):
        """
        Creates a histogram on the provided axis.

        Args:
            axs (Axes): Matplotlib axis object.
            data (np.ndarray): Data array for the histogram.
            bins (np.ndarray or int): Bin edges or number of bins.
            xlim (list[float], optional): X-axis limits [min, max]. Defaults to data min/max.
            ylim (list[float], optional): Y-axis limits [min, max]. Defaults to None.
            histcolor (str): Histogram color. Default is 'gray'.
            xaxislabel (str): Label for the X-axis. Default is 'x axis'.
            alpha (float): Transparency of the histogram. Default is 1.0.
            histtype (str): Type of histogram (e.g., 'bar', 'step'). Default is 'bar'.
            density (bool): Normalize to probability density. Default is True.
            label (str): Label for the histogram legend. Default is ''.

        Returns:
            axs (Axes): The modified axis object.
        """
        fontsz = 15 if self.poster else 8

        # Set xlim if not provided
        xlim = xlim or [np.min(data), np.max(data)]

        # Plot the histogram
        axs.hist(
            data,
            bins=bins,
            density=density,
            color=histcolor,
            alpha=alpha,
            histtype=histtype,
            label=label,
        )

        # Configure grid and limits
        axs.grid(
            visible=True,
            which="both",
            linestyle="--",
            color="gray",
            linewidth=0.25,
            alpha=0.25,
        )
        axs.set_xlim(xlim)
        if ylim:
            axs.set_ylim(ylim)

        # Set axis labels
        ylabel = "probability density" if density else "frequency"
        axs.set_ylabel(ylabel, fontsize=fontsz)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)

        return axs

    def scatter_plot(
        self,
        axs,
        x,
        y,
        xlim=None,
        ylim=None,
        label="",
        edgecolor="k",
        facecolor="white",
        s=5,
        lw=0.75,
        xaxislabel="x axis",
        yaxislabel="y axis",
        alpha=1,
    ):
        """scatter_plot function
        takes data and makes a scatter plot
        Args:
            x is x data
            y os y data
            xlim is x limits; default is None (which computes max/min)
            ylim is y limits; default is None (which computes max/min)
            label is label; default is nothing
            edgecolor is edge colour; default is black
            facecolor is face colour; default is white
            s is size of scatter point; default is 5
            lw is line width (default 0.75)
            xaxislabel is x axis label (default is 'x axis')
            yaxislabel is y axis label (default is 'y axis')
        Returns:
            axs is axis object"""
        fontsz = 15 if self.poster else 8

        # Set xlim, ylim if not provided
        xlim = xlim or [np.min(x), np.max(x)]
        ylim = ylim or [np.min(y), np.max(y)]

        axs.scatter(
            x,
            y,
            s=s,
            edgecolors=edgecolor,
            facecolor=facecolor,
            lw=lw,
            label=label,
            alpha=alpha,
        )
        axs.set_xlim(xlim)
        axs.set_ylim(ylim)
        axs.grid(True, which="both", ls="--", c="gray", lw=0.25, alpha=0.25)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        axs.set_ylabel(yaxislabel, fontsize=fontsz)
        return axs

    def image_plot(
        self,
        axs,
        data,
        vmin=None,
        vmax=None,
        cmap="gist_gray",
        cbar="on",
        cbarlabel="photons",
        label="",
        labelcolor="white",
        pixelsize=110,
        scalebarsize=10000,
        scalebarlabel=r"10$\,\mu$m",
        alpha=1,
        plotmask=False,
        mask=None,
        maskcolor="white",
        masklinewidth=0.75,
    ):
        """image_plot function
        takes image data and makes an image plot

        Args:
            axs (axis): axis object
            data (np.2darray): image
            vmin (float): minimum pixel intensity displayed (default 0.1%)
            vmax (float): minimum pixel intensity displayed (default 99.9%)
            cmap (string): colour map used; default gray)
            cbarlabel (string): colour bar label; default 'photons'
            label (string): is any annotation
            labelcolor (string): annotation colour
            pixelsize (float): pixel size in nm for scalebar, default 110
            scalebarsize (float): scalebarsize in nm, default 5000
            scalebarlabel (string): scale bar label, default 5 um

        Returns:
            axs (axis): axis object"""

        fontsz = 15 if self.poster else 8

        # Set xlim, ylim if not provided
        vmin = vmin or np.percentile(data.ravel(), 0.1)
        vmax = vmax or np.percentile(data.ravel(), 99.9)

        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        im = axs.imshow(
            data, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, origin="lower"
        )
        if cbar == "on":
            cbar = plt.colorbar(im, fraction=0.045, pad=0.02, ax=axs, location="left")
            cbar.set_label(cbarlabel, rotation=90, labelpad=1, fontsize=fontsz)
            cbar.ax.tick_params(labelsize=fontsz - 1, pad=0.1, width=0.5, length=2)
        axs.set_xticks([])
        axs.set_yticks([])
        pixvals = scalebarsize / pixelsize
        scalebar = AnchoredSizeBar(
            axs.transData,
            pixvals,
            scalebarlabel,
            "lower right",
            pad=0.1,
            color=labelcolor,
            frameon=False,
            size_vertical=1,
        )

        axs.add_artist(scalebar)
        axs.annotate(
            label,
            xy=(5, 5),
            xytext=(20, 60),
            xycoords="data",
            color=labelcolor,
            fontsize=fontsz - 1,
        )

        if plotmask == True:
            axs.contour(mask, [0.5], linewidths=masklinewidth, colors=maskcolor)

        return axs

    def image_scatter_plot(
        self,
        axs,
        data,
        xdata,
        ydata,
        vmin=None,
        vmax=None,
        cmap="gist_gray",
        cbar="on",
        cbarlabel="photons",
        label="",
        labelcolor="white",
        pixelsize=110,
        scalebarsize=10000,
        scalebarlabel=r"10$\,\mu$m",
        alpha=1,
        scattercolor="red",
        facecolor="None",
        s=20,
        lws=0.75,
        plotmask=False,
        mask=None,
        maskcolor="white",
        masklinewidth=0.75,
        alpha_scatter=1,
    ):
        """image_plot function
        takes image data and makes an image plot

        Args:
            axs (axis): axis object
            data (np.2darray): image
            xdata (np.1darray): scatter points, x
            ydata (np.1darray): scatter points, y
            vmin (float): minimum pixel intensity displayed (default 0.1%)
            vmax (float): minimum pixel intensity displayed (default 99.9%)
            cmap (string): colour map used; default gray)
            cbarlabel (string): colour bar label; default 'photons'
            label (string): is any annotation
            labelcolor (string): annotation colour
            pixelsize (float): pixel size in nm for scalebar, default 110
            scalebarsize (float): scalebarsize in nm, default 5000
            scalebarlabel (string): scale bar label, default 5 um

        Returns:
            axs (axis): axis object"""

        fontsz = 15 if self.poster else 8

        # Set xlim, ylim if not provided
        vmin = vmin or np.percentile(data.ravel(), 0.1)
        vmax = vmax or np.percentile(data.ravel(), 99.9)

        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        im = axs.imshow(
            data, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, origin="lower"
        )
        if cbar == "on":
            cbar = plt.colorbar(im, fraction=0.045, pad=0.02, ax=axs, location="left")
            cbar.set_label(cbarlabel, rotation=90, labelpad=1, fontsize=fontsz)
            cbar.ax.tick_params(labelsize=fontsz - 1, pad=0.1, width=0.5, length=2)
        axs.set_xticks([])
        axs.set_yticks([])
        pixvals = scalebarsize / pixelsize
        scalebar = AnchoredSizeBar(
            axs.transData,
            pixvals,
            scalebarlabel,
            "lower right",
            pad=0.1,
            color=labelcolor,
            frameon=False,
            size_vertical=1,
        )

        axs.add_artist(scalebar)
        axs.annotate(
            label,
            xy=(5, 5),
            xytext=(20, 60),
            xycoords="data",
            color=labelcolor,
            fontsize=fontsz - 1,
        )
        if plotmask == True:
            axs.contour(mask, [0.5], linewidths=masklinewidth, colors=maskcolor)

        axs.scatter(
            ydata,
            xdata,
            lw=lws,
            edgecolor=scattercolor,
            s=s,
            facecolors=facecolor,
            alpha=alpha_scatter,
        )
        return axs
