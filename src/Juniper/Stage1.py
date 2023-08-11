import os

import numpy as np
import time

import matplotlib.pyplot as plt
from scipy import signal
from astropy.stats import sigma_clip

from jwst.pipeline import Detector1Pipeline
from .utils import img

def doStage1(filepath, outfile, outdir,
             group_scale={"skip":False},
             dq_init={"skip":False},
             saturation={"skip":False},
             superbias={"skip":False},
             refpix={"skip":False},
             linearity={"skip":False},
             dark_current={"skip":False},
             jump={"skip":True},
             ramp_fit={"skip":False},
             gain_scale={"skip":False},
             one_over_f={"skip":False, "bckg_rows":[1,2,3,4,5,6,-1,-2,-3,-4,-5,-6], "sigma":3.0, "kernel":(5,1), "show":False}
             ):
    '''
    Performs Stage 1 calibration on one file.
    
    :param filepath: str. Location of the file you want to correct. The file must be of type *_uncal.fits.
    :param outfile: str. Name to give to the calibrated file.
    :param outdir: str. Location of where to save the calibrated file to.
    :param group_scale, dq_init, saturation, etc.: dict. These are the dictionaries shown in the Detector1Pipeline() documentation, which control which steps are run and what parameters they are run with. Please consult jwst-pipeline.readthedocs.io for more information on these dictionaries.
    :param one_over_f: dict. Keyword "skip" is a bool that sets whether or not to perform this step. Keyword "bckg_rows" contains list of integers that selects which rows of the array are used as background for 1/f subtraction. Keyword "sigma" sets how aggressively to clean the background region before using it for subtraction. Keyword "kernel" sets the filter shape that will be used in cleaning the background region. Keyword "show" is a bool that sets whether the first group of the first integration is shown as it is cleaned.
    :return: a Stage 1 calibrated file *_rateints.fits saved to the outdir.
    '''
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Report initialization of Stage 1.
    print("Performing Stage 1 JWST/Juniper calibrations on file: " + filepath)
    print("Running JWST Stage 1 pipeline starting at GroupScale and stopping before Jump...")
    
    # Collect timestamp to track how long this takes.
    t0 = time.time()
    with Detector1Pipeline.call(filepath,
                                steps={"group_scale":group_scale,
                                       "dq_init":dq_init,
                                       "saturation":saturation,
                                       "superbias":superbias,
                                       "refpix":refpix,
                                       "linearity":linearity,
                                       "dark_current":dark_current,
                                       "jump":jump,
                                       "ramp_fit": {"skip": True}, "gain_scale": {"skip": True}}) as result:
        print("Stage 1 calibrations up to step Jump resolved in %.3f seconds." % (time.time() - t0))
        
        if not one_over_f["skip"]:
            # Before we ramp_fit, we perform 1/f subtraction.
            print("Performing Juniper pre-RampFit 1/f subtraction...")
            result.data = one_over_f_subtraction(result.data,
                                                 bckg_rows=one_over_f["bckg_rows"],
                                                 bckg_kernel=one_over_f["kernel"],
                                                 bckg_sigma=one_over_f["sigma"],
                                                 show=one_over_f["show"])
        else:
            print("Skipping Juniper pre-RampFit 1/f subtraction...")
        
        # Now we can resume Stage 1 calibration.
        t02 = time.time()
        print("Resuming JWST Stage 1 pipeline calibrations through RampFit and GainScale steps."
              "\nThe RampFit step can take several minutes to hours depending on how big your dataset is,\n"
              "so I suggest you find something else to do in the meantime. Anyways...")
        
        result = Detector1Pipeline.call(result, output_file=outfile, output_dir=outdir,
                                        steps={"group_scale": {"skip": True},
                                               "dq_init": {"skip": True},
                                               "saturation": {"skip": True},
                                               "superbias": {"skip": True},
                                               "refpix": {"skip": True},
                                               "linearity": {"skip": True},
                                               "dark_current": {"skip": True},
                                               "jump": {"skip": True},
                                               "ramp_fit":ramp_fit,
                                               "gain_scale":gain_scale})
        print("Finished final steps of JWST Stage 1 pipeline in %.3f minutes." % ((time.time()-t02)/60))
    print("File calibrated and saved.")
    print("Stage 1 calibrations completed in %.3f minutes." % ((time.time() - t0)/60))

def one_over_f_subtraction(data, bckg_rows, bckg_kernel, bckg_sigma, show):
    '''
    Performs 1/f subtraction on the given array.
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).
    
    :param data: 4D array. Array of integrations x groups x rows x cols.
    :param bckg_rows: list of integers. Indices of the rows to use as the background region.
    :param bckg_kernel: tuple of odd int. Kernel to use when cleaning the background region. Should be a column like (5,1) so that columns do not contaminate adjacent columns.
    :param bckg_sigma: float. Threshold to reject outliers from the background region.
    :param show: bool. Whether to show the cleaned frames. For inspection of whether this this step is working properly.
    :return: 4D array that has undergone 1/f subtraction.
    '''
    # Time this step.
    t0 = time.time()
    for i in range(np.shape(data)[0]): # for each integration
        for g in range(np.shape(data)[1]): # for each group
            # Define the background region.
            background_region = data[i, g, bckg_rows, :]
            if (i == 0 and g == 0 and show):
                fig, ax, im = img(np.log10(np.abs(background_region)), aspect=5, vmin=None, vmax=None, norm=None)
                plt.show()
                plt.close()
            
            # Clean the background region of outliers, so that CRs aren't propagated through the array.
            background_region = clean(background_region, bckg_sigma, bckg_kernel)
            if (i == 0 and g == 0 and show):
                fig, ax, im = img(np.log10(np.abs(background_region)), aspect=5, vmin=None, vmax=None, norm=None)
                plt.show()
                plt.close()
            
            # Define the mean background in each column and extend to a full-size array.
            background = background_region.mean(axis=0)
            background = np.array([background,]*np.shape(data)[2])
            if (i == 0 and g == 0 and show):
                fig, ax, im = img(np.log10(np.abs(background)), aspect=5, vmin=None, vmax=None, norm=None)
                plt.show()
                plt.close()

            if (i == 0 and g == 0 and show):
                fig, ax, im = img(np.log10(np.abs(data[i, g, :, :])), aspect=5, vmin=None, vmax=None, norm=None)
                plt.show()
                plt.close()

            data[i, g, :, :] = data[i, g, :, :] - background

            if (i == 0 and g == 0 and show):
                fig, ax, im = img(np.log10(np.abs(data[i, g, :, :])), aspect=5, vmin=None, vmax=None, norm=None)
                plt.show()
                plt.close()
                
        if (i%1000 == 0 and i != 0):
            # Report every 1000 integrations.
            elapsed_time = time.time()-t0
            iterrate = i/elapsed_time
            iterremain = np.shape(data)[0] - i
            print("On integration %.0f. Elapsed time in this step is %.3f seconds." % (i, elapsed_time))
            print("Average rate of integration processing: %.3f ints/s." % iterrate)
            print("Estimated time remaining: %.3f seconds.\n" % (iterremain/iterrate))
    print("1/f subtraction completed in %.3f seconds." % (time.time()-t0))
    return data

def clean(data, sigma, kernel):
    '''
    Cleans one 2D array with median spatial filtering.
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).
    
    :param data: 2D array. Array that will be median-filtered.
    :param sigma: float. Sigma at which to reject outliers.
    :param kernel: tuple of odd int. Kernel to use for median filtering.
    :return: cleaned 2D array.
    '''
    medfilt = signal.medfilt2d(data, kernel)
    diff = data - medfilt
    temp = sigma_clip(diff, sigma=sigma, axis=0)
    mask = temp.mask
    int_mask = mask.astype(float) * medfilt
    test = (~mask).astype(float)
    return (data*test) + int_mask