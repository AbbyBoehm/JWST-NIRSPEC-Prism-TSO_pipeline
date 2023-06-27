import os

import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits

from jwst import datamodels as dm

def img(array, aspect=1, title=None, vmin=None, vmax=None, norm=None):
    '''
    Image plotting utility to plot the input 2D array.
    
    :param array: 2D array. Image you want to plot.
    :param aspect: float. Aspect ratio. Useful for visualizing narrow arrays.
    :param title: str. Title to give the plot.
    :param vmin: float. Minimum value for color mapping.
    :param vmax: float. Maximum value for color mapping.
    :param norm: str. Type of normalisation scale to use for this image.
    '''
    fig, ax = plt.subplots(figsize=(20, 25))
    if norm == None:
        im = ax.imshow(array, aspect=aspect, origin="lower", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(array, aspect=aspect, norm=norm, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    return fig, ax, im

def stitch_files(files):
    '''
    Reads all *_calints.fits files given by the filepaths provided
    and stitches them together.

    :param
    '''
    if "postprocessed" in files[0]:
        print("Reading post-processed files, adjusting outputs accordingly...")
        # Reading a postprocessed file, adjust strategy!
        for i, file in enumerate(files):
            print("Attempting to locate file: " + file)
            fitsfile = dm.open(file)
            print("Loaded the file " + file + " successfully.")

            # Need to retrieve the wavelength object, science object, and DQ object from this.
            if i == 0:
                segments = fitsfile.data
                errors = fitsfile.err
                segstarts = [np.shape(fitsfile.data)[0]]
                wavelengths = [fitsfile.wavelength]
                dqflags = fitsfile.dq
                times = fitsfile.int_times["int_mid_MJD_UTC"]
            else:
                segments = np.concatenate([segments, fitsfile.data], 0)
                errors = np.concatenate([errors, fitsfile.err], 0)
                segstarts.append(np.shape(fitsfile.data)[0] + sum(segstarts))
                wavelengths.append(fitsfile.wavelength)
                dqflags = np.concatenate([dqflags, fitsfile.dq], 0)
                times = np.concatenate([times, fitsfile.int_times["int_mid_MJD_UTC"]], 0)
            
            fitsfile.close()
            
            with fits.open(file) as f:
                frames_to_reject = f["REJECT"].data
            
            print("Retrieved segments, wavelengths, DQ flags, and times from file: " + file)
            print("Closing file and moving on to next one...")
        return segments, errors, segstarts, wavelengths, dqflags, times, frames_to_reject
    else:
        for i, file in enumerate(files):
            print("Attempting to locate file: " + file)
            fitsfile = dm.open(file)
            print("Loaded the file " + file + " successfully.")

            # Need to retrieve the wavelength object, science object, and DQ object from this.
            if i == 0:
                segments = fitsfile.data
                errors = fitsfile.err
                segstarts = [np.shape(fitsfile.data)[0]]
                wavelengths = [fitsfile.wavelength]
                dqflags = fitsfile.dq
                times = fitsfile.int_times["int_mid_MJD_UTC"]
            else:
                segments = np.concatenate([segments, fitsfile.data], 0)
                errors = np.concatenate([errors, fitsfile.err], 0)
                segstarts.append(np.shape(fitsfile.data)[0] + sum(segstarts))
                wavelengths.append(fitsfile.wavelength)
                dqflags = np.concatenate([dqflags, fitsfile.dq], 0)
                times = np.concatenate([times, fitsfile.int_times["int_mid_MJD_UTC"]], 0)

            print("Retrieved segments, wavelengths, DQ flags, and times from file: " + file)
            print("Closing file and moving on to next one...")
            fitsfile.close()
        return segments, errors, segstarts, wavelengths, dqflags, times