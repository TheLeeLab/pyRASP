#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:02:24 2023

Class related to making figure-quality plots

Probably best to set your default sans-serif font to Helvetica before you make
figures: https://fowlerlab.org/2019/01/03/changing-the-sans-serif-font-to-helvetica/
 
The maximum published width for a one-column
figure is 3.33 inches (240 pt). The maximum width for a two-column
figure is 6.69 inches (17 cm). The maximum depth of figures should 
be 8 ¼ in. (21.1 cm).

panel labels are 8 point font, ticks are 7 point font,
annotations and legends are 6 point font

@author: jbeckwith
"""
import matplotlib # requires 3.8.0
import matplotlib.pyplot as plt
import numpy as np

class Plotter():
    def __init__(self, poster=True):
        self.poster = poster
        self = self
        return
    
    def two_column_plot(self, nrows=1, ncolumns=1, heightratio=[1], widthratio=[1], height=0, big=True):
        """ two_column_plot function
        takes data and makes a two-column width figure
        ================INPUTS============= 
        npanels is panel matrix
        ratios is size ratios of panels
        ================OUTPUT============= 
        fig, axs are figure objects """
        
        # first, check everything matches
        try:
            if len(heightratio) != nrows:
                raise Exception('Number of height ratios incorrect')
            if len(widthratio) != ncolumns:
                raise Exception('Number of width ratios incorrect')
        except Exception as error:
            print('Caught this error: ' + repr(error))
            return
        
        if self.poster == True:
            fontsz = 12
            lw = 1
        else:
            fontsz = 7
            lw = 1

        if big == True:
            xsize = 5*ncolumns
        else:
            xsize = 6.69 # 3.33 inches for one-column figure
            
        if height == 0:
            if big == True:
                ysize = 5*nrows
            else:
                ysize = 3*nrows
        else:
            ysize = height
        
        plt.rcParams['figure.figsize'] = [xsize, ysize]
        plt.rcParams['font.size'] = fontsz
        plt.rcParams['svg.fonttype'] = 'none'
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        plt.rcParams['axes.linewidth'] = lw # set the value globally
        
        fig, axs = plt.subplots(nrows, ncolumns, height_ratios=heightratio, 
                                width_ratios=widthratio) # create number of panels

        # clean up axes, tick parameters
        if nrows*ncolumns == 1:
            axs.xaxis.set_tick_params(width=lw, length=lw*4)
            axs.yaxis.set_tick_params(width=lw, length=lw*4)
            axs.tick_params(axis='both', pad=1.2)
        elif nrows*ncolumns == 2:
            for i in np.arange(2):
                 axs[i].xaxis.set_tick_params(width=lw, length=lw*4)
                 axs[i].yaxis.set_tick_params(width=lw, length=lw*4)
                 axs[i].tick_params(axis='both', pad=1.2)
        elif nrows*ncolumns == len(widthratio):
            for i in np.arange(len(widthratio)):
                 axs[i].xaxis.set_tick_params(width=lw, length=lw*4)
                 axs[i].yaxis.set_tick_params(width=lw, length=lw*4)
                 axs[i].tick_params(axis='both', pad=1.2)            
        else:
            for i in np.arange(nrows):
                for j in np.arange(ncolumns):
                    axs[i,j].xaxis.set_tick_params(width=0.5, length=lw*4)
                    axs[i,j].yaxis.set_tick_params(width=0.5, length=lw*4)
                    axs[i,j].tick_params(axis='both', pad=1.2)
        return fig, axs

    def histogram_plot(self, axs, data, bins, xlim=None, ylim=None, 
        histcolor='gray', xaxislabel='x axis', alpha=1, histtype='bar', density=True): 
        """ histogram_plot function
        takes data and makes a histogram
        ================INPUTS============= 
        data is data
        bins are bins
        xlim is x limits; default is None (which computes max/min)
        ylim is y limits; default is None (which lets it keep its default)
        histcolor is histogram colour (default is gray)
        xaxislabel is x axis label (default is 'x axis')
        alpha is histogram transparency (default 1)
        density is if to plot as pdf, default yes
        histtype is histogram type, default bar
        ================OUTPUT============= 
        axs is axis object """
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8

        if xlim is None:
            xlim = np.array([np.min(data), np.max(data)])

        axs.hist(data, bins=bins, density=density, color=histcolor, alpha=alpha, histtype=histtype);
        axs.grid(True,which="both",ls="--",c='gray', lw=0.25, alpha=0.25) 
        if density==True:
            axs.set_ylabel('probability density', fontsize=fontsz)
        else:
            axs.set_ylabel('frequency', fontsize=fontsz)    
        axs.set_xlim(xlim)
        if ylim is not None:
            axs.set_ylim(ylim)
        axs.set_xlabel(xaxislabel, fontsize=fontsz)
        return axs

    def image_plot(self, axs, data, vmin=None, vmax=None, cmap='gist_gray', cbar='on', cbarlabel='photons', label='', 
                   labelcolor='white', pixelsize=110, scalebarsize=5000, scalebarlabel=r'5$\,\mu$m', alpha=1, plotmask=False, mask=None):
        """ image_plot function
        takes image data and makes an image plot
        ================INPUTS============= 
        data is image
        vmin is minimum pixel displayed (default 99.9%)
        vmax is maximum pixel displayed (default 0.1%)
        cmap is colour map used; default gray
        cbarlabel is colour bar label; default intensity
        label is any annotation
        labelcolor is annotation colour
        pixelsize is pixel size in nm for scalebar, default 69
        scalebarsize is scalebarsize in nm, default 5000
        scalebarlabel is scale bar label, default 5 um
        ================OUTPUT============= 
        axs is axis object """
        
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8

        if vmin is None:
            vmin = np.percentile(data.ravel(), 0.1)
        if vmax is None:
            vmax = np.percentile(data.ravel(), 99.9)

        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        
        im = axs.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, origin='lower')
        if cbar == 'on':
        	cbar = plt.colorbar(im, fraction=0.045, pad=0.02, ax=axs, location='left')
        	cbar.set_label(cbarlabel, rotation=90, labelpad=1, fontsize=fontsz)
        	cbar.ax.tick_params(labelsize=fontsz-1, pad=0.1, width=0.5, length=2) 
        axs.set_xticks([])
        axs.set_yticks([])
        pixvals = scalebarsize/pixelsize
        scalebar = AnchoredSizeBar(axs.transData,
                                        pixvals, scalebarlabel, 'lower right', 
                                        pad=0.1,
                                        color=labelcolor,
                                        frameon=False,
                                        size_vertical=1)

        axs.add_artist(scalebar)
        axs.annotate(label, xy=(5, 5), xytext=(20, 60),
                        xycoords='data', color=labelcolor, fontsize=fontsz-1)
        
        if plotmask == True:
            axs.contour(mask, [0.5], linewidths=1.5, colors='blue')

        return axs
    
    def image_scatter_plot(self, axs, data, xdata, ydata, vmin=None, vmax=None, cmap='gist_gray', cbar='on', cbarlabel='photons', label='', 
                   labelcolor='white', pixelsize=110, scalebarsize=5000, scalebarlabel=r'5$\,\mu$m', alpha=1, scattercolor='red', s=20, lws=0.75):
        """ image_plot function
        takes image data and makes an image plot
        ================INPUTS============= 
        data is image
        vmin is minimum pixel displayed (default 99.9%)
        vmax is maximum pixel displayed (default 0.1%)
        cmap is colour map used; default gray
        cbarlabel is colour bar label; default intensity
        label is any annotation
        labelcolor is annotation colour
        pixelsize is pixel size in nm for scalebar, default 69
        scalebarsize is scalebarsize in nm, default 5000
        scalebarlabel is scale bar label, default 5 um
        ================OUTPUT============= 
        axs is axis object """
        
        if self.poster == True:
            fontsz = 15
        else:
            fontsz = 8

        if vmin is None:
            vmin = np.percentile(data.ravel(), 0.1)
        if vmax is None:
            vmax = np.percentile(data.ravel(), 99.9)

        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        
        im = axs.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, origin='lower')
        if cbar == 'on':
        	cbar = plt.colorbar(im, fraction=0.045, pad=0.02, ax=axs, location='left')
        	cbar.set_label(cbarlabel, rotation=90, labelpad=1, fontsize=fontsz)
        	cbar.ax.tick_params(labelsize=fontsz-1, pad=0.1, width=0.5, length=2) 
        axs.set_xticks([])
        axs.set_yticks([])
        pixvals = scalebarsize/pixelsize
        scalebar = AnchoredSizeBar(axs.transData,
                                        pixvals, scalebarlabel, 'lower right', 
                                        pad=0.1,
                                        color=labelcolor,
                                        frameon=False,
                                        size_vertical=1)

        axs.add_artist(scalebar)
        axs.annotate(label, xy=(5, 5), xytext=(20, 60),
                        xycoords='data', color=labelcolor, fontsize=fontsz-1)
        axs.scatter(xdata, ydata, lw=lws, edgecolor=scattercolor, s=s, facecolors='None')
        return axs
