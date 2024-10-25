import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from scipy.interpolate import interp1d
import pywt
from matplotlib.colors import Normalize, LogNorm, NoNorm
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time


def cwt_spectrogram(x, fs, nNotes=30, detrend=False, normalize=False):
    
    #length of signal 
    N = len(x)
    #Number of samples pr seconds
    dt = 1.0 / fs
    #sequence of n from 0 to len(x) times dt giving each sequence af time stamp
    times = np.arange(N) * dt

    ###########################################################################
    # detrend and normalize
    #detrend and normalization should not occur as it is for now, since we need all the information as possible for feature extraction
    if detrend:
        x = signal.detrend(x,type='linear')
    if normalize:
        stddev = x.std()
        x = x / stddev


    frequencies_for_scale = np.arange(1, 50.0, 1.0 / nNotes) / fs
    

    scales = pywt.frequency2scale('cmor1.5-1.0', frequencies_for_scale)
    
#     print (scales)

    ###########################################################################
    # cwt and the frequencies used. 
    # Use the complex morlet with bw=1.5 and center frequency of 1.0
    #Choice of center frequency and bandwith: Center frequency determines the center frequency of the wavelet, 
                        #meaning since we have alpha, beta, gamma etc. we might need to choose different, depending on label.
    #Bandwidth controls the width of the frequency band represented on the wavelet. So, since we have intervals of approx 4 hz or so for some major frequencies, let's try use that.
    coef, freqs=pywt.cwt(x,scales,'cmor1.5-1.0', method = 'fft')
    frequencies = pywt.scale2frequency('cmor1.5-1.0', scales) / dt
    
    ###########################################################################
    # power
#     power = np.abs(coef)**2
    #Computes the power spectrum of the wavelet coefficients, by taking the absolute square. np.conj() calculates the complex conjugate.
    power = np.abs(coef * np.conj(coef))
    
    # smooth a bit
    #Choice of smoothing the power spectrum (try urself to note it and see). Looks better, but may be missing some information for features due to "smoothing"
    power = ndimage.gaussian_filter(power, sigma=2)

    ###########################################################################
    # cone of influence in frequency for cmorxx-1.0 wavelet
    #The COI represents the region of the spectrogram where edge effects become important and should be treated with caution. Calculated in wavelength and frequency.
    f0 = 2*np.pi
    cmor_coi = 1.0 / np.sqrt(2)
    cmor_flambda = 4*np.pi / (f0 + np.sqrt(2 + f0**2))
    # cone of influence in terms of wavelength
    coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    coi = cmor_flambda * cmor_coi * dt * coi
    # cone of influence in terms of frequency
    coif = 1.0/coi


    return power, times, frequencies, coif


def spectrogram_plot(z, times, frequencies, coif, cmap, norm, colorbar=True):
    ###########################################################################
    # plot
    
    # set default colormap, if none specified
    if cmap is None:
        cmap = get_cmap('Greys')
    # or if cmap is a string, get the actual object
    elif isinstance(cmap, str):
        cmap = get_cmap(cmap)

    # create the figure
    fig, ax = plt.subplots()

    xx, yy = np.meshgrid(times, frequencies)
    ZZ = z
    
    im = ax.pcolor(xx, yy, ZZ, norm=norm, cmap=cmap)
    # ax.plot(times, coif)
    ax.plot(times)
    # ax.fill_between(times, coif, step="mid", alpha=0.4)
    
    if colorbar:
        cbaxes = inset_axes(ax, width="2%", height="90%", loc=4) 
        fig.colorbar(im, cax=cbaxes, orientation='vertical')

    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(frequencies.min(), frequencies.max())

    # Hide the axes
    ax.axis('off')
    
    return fig


def spectrogram2_plot(z, times, frequencies, coif, cmap=None, norm=Normalize(), ax=None, colorbar=True):
    ###########################################################################
    # plot
    
    # set default colormap, if none specified
    if cmap is None:
        cmap = get_cmap('Greys')
    # or if cmap is a string, get the actual object
    elif isinstance(cmap, str):
        cmap = get_cmap(cmap)

    # create the figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    xx,yy = np.meshgrid(times,frequencies)
    ZZ = z
    
    im = ax.pcolor(xx,yy,ZZ, norm=norm, cmap=cmap)
    #ax.plot(times,coif)
    ax.plot(times)
    # ax.fill_between(times,coif, step="mid", alpha=0.4)
    
    if colorbar:
        cbaxes = inset_axes(ax, width="2%", height="90%", loc=4) 
        fig.colorbar(im,cax=cbaxes, orientation='vertical')

    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(frequencies.min(), frequencies.max())

    return ax