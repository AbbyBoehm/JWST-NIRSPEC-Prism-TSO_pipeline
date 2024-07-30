import os

import numpy as np
import time

import matplotlib.pyplot as plt
from scipy import signal
from astropy.io import fits

from revised_utils import timer

def check_curvature(outfile, outdir, time_step, show):
    '''
    Checks if the file needs its curvature corrected, and if it does, corrects it.

    :param outfile: str. The name of the file we are checking, sans "_calints.fits".
    :param outdir: str. The directory where the file can be found.
    :param time_step: bool. Whether to time the step.
    :param show: bool. Whether to show diagnostic plots.
    :return: outfile overwritten to outdir with curvature corrected, if necessar.
    '''
    output_file = os.path.join(outdir, outfile+"_calints.fits")
    with fits.open(output_file) as file:
        grating = file[0].header['GRATING']
        if grating in ("G395M","G395H"):
            print("{} grating detected, correcting for trace curvature...".format(grating))
            shifted_data, shifted_wvs = fix_curvature(file['SCI'].data, file['WAVELENGTH'].data, time_step=time_step, show=show)
            write_curve_fixed_file(output_file, shifted_data, shifted_wvs)
    return None

def fix_curvature(data, wvs, time_step, show):
    '''
    Corrects trace curvature in given array. Adapted in part from Eureka!

    :param data: 3D array. An array of *calints.fits data.
    :param wvs: 3D array. An array of wavelength information from *calints.fits which also needs to be rolled to keep the wavelength solution proper.
    :param time_step: bool. Whether to time the step.
    :param show: bool. Whether to show diagnostic plots.
    :return: data and wavelength arrays that have been uncurved.
    '''
    # Time this step if asked.
    if time_step:
        t0 = time.time()
    
    # Use the median frame to determine needed rolls.
    medframe = np.median(data,axis=0)
    medframe[np.isnan(medframe)] = 0 # get rid of nans because they upset the roller.
    rolls = get_rolls(medframe) # get the rolls needed to correct the framese.

    if show:
        plt.plot(rolls)
        plt.xlabel("column position [pixels]")
        plt.ylabel("roll [pixels]")
        plt.title("Rolls needed to correct frames")
        plt.ylim(-13, 13)
        plt.show()
        plt.close()

    # There is only one wavelength frame, so roll it by the median rolls in time.
    shifted_wvs = roll_one_frame(wvs, rolls)

    # Then for each frame in data, need to roll it.
    shifted_data = np.empty_like(data)
    for i in range(data.shape[0]):
        shifted_data[i,:,:] = roll_one_frame(data[i,:,:], rolls)
        if (i == 0 and show):
            plt.imshow(shifted_data[i,:,:])
            plt.title("Rolled frame 0")
            plt.show()
            plt.close()
        if (i%10 == 0 and i != 0 and time_step):
            timer(time.time()-t0,i+1,i,data.shape[0]-i)
    
    return shifted_data, shifted_wvs

def roll_one_frame(frame, rolls):
    '''
    Roll one frame into alignment.

    :param frame: 2D array. One frame from *calints.fits.
    :param rolls: lst of int. How many pixels by which to roll each frame.
    '''
    retain_last_roll = 0
    for j, roll in enumerate(rolls):
        if abs(roll) > 20:
            # If an outlier roll is found, we use the same roll that we used the last time a roll succeeded.
            roll = retain_last_roll
        frame[:,j] = np.roll(frame[:,j], int(roll))
        retain_last_roll = roll
    return frame

def get_rolls(frame):
    '''
    Determine the rolls needed to straighten the trace using the median frame.
    Adapted from Eureka! S3 straighten.py code.

    :param frame: 2D array. Median frame used to measure the rolls.
    :return: lst of int values that will be used to roll traces into place.
    '''
    pix_centers = np.arange(frame.shape[0]) + 0.5
    COMs = signal.medfilt((np.sum(pix_centers[:,np.newaxis]*np.abs(frame),axis=0)/np.sum(np.abs(frame),axis=0)),7)
    integer_COMs = np.around(COMs - 0.5).astype(int)
    new_center = int(frame.shape[0]/2) - 1
    rolls = new_center - integer_COMs
    rolls[COMs<0] = 0
    rolls[COMs>frame.shape[0]] = 0
    rolls = signal.medfilt(rolls,41)
    return rolls

def write_curve_fixed_file(output_file, shifted_data, shifted_wvs):
    '''
    Write curve-corrected fits file.

    :param output_file: str. Name of the *calints.fits file we just rolled and a going to overwrite.
    :return: overwritten output_file. Routine itself returns no callables.
    '''
    with fits.open(output_file, mode="update") as fits_file:
        # Need to update data and wavelength attributes to be rotated arrays.
        fits_file['SCI'].data = shifted_data
        fits_file['WAVELENGTH'].data = shifted_wvs

        # All modified headers get written out.
        fits_file.writeto(output_file, overwrite=True)
    return None