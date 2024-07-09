import os

import numpy as np
import time

import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import least_squares
from scipy.stats import linregress
from astropy.stats import sigma_clip

from jwst.pipeline import Detector1Pipeline
from .utils import img

def doStage1(filepaths, outfiles, outdir,
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
    Performs Stage 1 calibration on a list of files.
    
    :param filepaths: lst of str. Location of the files you want to correct. The files must be of type *_uncal.fits.
    :param outfiles: lst of str. Names to give to the calibrated files.
    :param outdir: str. Location of where to save the calibrated files to.
    :param group_scale, dq_init, saturation, etc.: dict. These are the dictionaries shown in the Detector1Pipeline() documentation, which control which steps are run and what parameters they are run with. Please consult jwst-pipeline.readthedocs.io for more information on these dictionaries.
    :param one_over_f: dict. Keyword "skip" is a bool that sets whether or not to perform this step. Keyword "bckg_rows" contains list of integers that selects which rows of the array are used as background for 1/f subtraction. Keyword "sigma" sets how aggressively to clean the background region before using it for subtraction. Keyword "kernel" sets the filter shape that will be used in cleaning the background region. Keyword "show" is a bool that sets whether the first group of the first integration is shown as it is cleaned.
    :return: a Stage 1 calibrated file *_rateints.fits saved to the outdir.
    '''
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Report initialization of Stage 1.
    print("Performing Stage 1 JWST/Juniper calibrations on files: ", filepaths)
    master_t0 = time.time()
    for filepath, outfile in zip(filepaths, outfiles):
        print("Running JWST Stage 1 pipeline on {} starting at GroupScale and stopping before Jump...".format(filepath))
        
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
        print("File " + filepath + " calibrated and saved in %.3f minutes." % ((time.time() - t0)/60))
    print("All files calibrated. Stage 1 completed in %.3f minutes." % ((time.time() - master_t0)/60))

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

    # If the background region is too small, this section and setting bckg_rows == 'all'
    # gives you the option to model the background by using Moffats to remove the trace.
    moffat_model = False
    if bckg_rows == 'all':
        print("You have requested that the entire frame be used for background subtraction.\nWe will now try to fit Moffats to every column to remove the trace\nand reconstruct the background with the trace removed.")
        moffat_model = True
        bckg_rows = [0,1,2,3,4,5,-1,-2,-3,-4,-5,-6]
        background_model = np.copy(data)
        for i in range(np.shape(data)[0]): # for each integration
            for g in range(np.shape(data)[1]): # for each group
                t_moffat = time.time()
                background_model[i,g,:,:] = clean(background_model[i,g,:,:], bckg_sigma, bckg_kernel) # remove CRs before performing the fit.
                moffat_of_this_frame = moffat_frame(background_model[i,g,:,:])
                background_model[i,g,:,:] = background_model[i,g,:,:] - moffat_of_this_frame # fit Moffats to every column and then remove them from the image to leave the background behind.

                if (i == 0 and g == 0 and show):
                    plt.imshow(moffat_of_this_frame)
                    plt.title("Moffat model of the 0th frame of the 0th group\nproduced in {:.2f} seconds".format(time.time()-t_moffat))
                    plt.show()
                    plt.close()
            if (i%1000 == 0 and i != 0):
                # Report every 1000 integrations.
                elapsed_time = time.time()-t0
                iterrate = i/elapsed_time
                iterremain = np.shape(data)[0] - i
                print("On integration %.0f. Elapsed time in background modelling step is %.3f seconds." % (i, elapsed_time))
                print("Average rate of integration processing: %.3f ints/s." % iterrate)
                print("Estimated time remaining to background model completion: %.3f seconds.\n" % (iterremain/iterrate))
        print("Built background model. Resuming background subtraction...")

    for i in range(np.shape(data)[0]): # for each integration
        if bckg_rows == 'mask':
            final_group = clean(np.copy(data[i,-1,:,:]),3,(5,1)) # select the very last group, which should have the brightest, strongest trace signal for masking. Remove outliers from it.
            mu = np.median(final_group[final_group<10000])
            sig = np.std(final_group[final_group<10000])
            masked_fg = np.ma.masked_where(np.abs(final_group - mu) > sig, final_group)
            trace_mask = np.ma.getmask(masked_fg) # and obtain the mask that hides the trace.
        else:
            trace_mask = np.ones_like(data[i,-1,:,:]) # copy the final group but as 1s so all is masked
            trace_mask[bckg_rows,:] = 0 # use 0s to open the bckg rows as not masked
        if (i == 0 and show):
            plt.imshow(trace_mask)
            plt.title('1/f trace mask')
            plt.show()
            plt.close()
        #meas = np.empty_like(data[0,:-1,:,:]) # need to make a fake int.
        for g in range(np.shape(data)[1]): # for each group
            # Define the background region and clean it for outliers before masking it.
            background_region = np.ma.masked_array(data=clean(np.copy(data[i, g, :, :]),bckg_sigma,bckg_kernel),
                                                   mask=trace_mask)
            '''
            if g < (np.shape(data)[1]-1):
                meas[g,:,:] = data[i,g,:,:]
            else:
                # We use the last three to project.
                if (i == 0 and show):
                    fig, ax, im = img(np.log10(np.abs(background_region)), aspect=5, vmin=None, vmax=None, norm=None)
                    plt.colorbar(im)
                    plt.show()
                    plt.close()
                new_group = np.empty_like(data[i,g,:,:]) # make an empty group to populate
                for K1 in range(new_group.shape[0]):
                    for K2 in range(new_group.shape[1]):
                        linregress_result = linregress(x=[z for z in range(np.shape(data)[1]-1)],
                                                       y=meas[:,K1,K2])
                        slope, intercept = linregress_result.slope, linregress_result.intercept
                        next_value = slope*(g+1) + intercept
                        new_group[K1,K2] = next_value
                background_region = np.ma.masked_array(data=clean(new_group,bckg_sigma,bckg_kernel),
                                                       mask=trace_mask)
                if (i == 0 and show):
                    fig, ax, im = img(np.log10(np.abs(background_region)), aspect=5, vmin=None, vmax=None, norm=None)
                    plt.colorbar(im)
                    plt.show()
                    plt.close()
            #np.copy(data[i, g, bckg_rows, :])
            '''


            if (i == 0 and g == 0 and show):
                fig, ax, im = img(np.log10(np.abs(background_region)), aspect=5, vmin=None, vmax=None, norm=None)
                plt.colorbar(im)
                plt.show()
                plt.close()
            
            # Clean the background region of outliers, so that CRs aren't propagated through the array.
            #background_region = clean(background_region, bckg_sigma, bckg_kernel)
            #if (i == 0 and g == 0 and show):
            #    fig, ax, im = img(np.log10(np.abs(background_region)), aspect=5, vmin=None, vmax=None, norm=None)
            #    plt.colorbar(im)
            #    plt.show()
            #    plt.close()

            # Define the mean background in each column and extend to a full-size array.
            #background = background_region.mean(axis=0)
            background = np.ma.median(background_region,axis=0)
            background = np.array([background,]*np.shape(data)[2])
            if (i == 0 and g == 0 and show):
                fig, ax, im = img(np.log10(np.abs(background)), aspect=5, vmin=None, vmax=None, norm=None)
                plt.colorbar(im)
                plt.show()
                plt.close()

            if (i == 0 and g == 0 and show):
                fig, ax, im = img(np.log10(np.abs(data[i, g, :, :])), aspect=5, vmin=None, vmax=None, norm=None)
                plt.colorbar(im)
                plt.show()
                plt.close()

            if moffat_model:
                plt.imshow(background, vmin=np.min(background), vmax=np.max(background))
                plt.colorbar()
                plt.title("Conventionally-extracted background using top and bottom row of array")
                plt.show()
                plt.close()

                background_region = background_model[i, g, :, :]
                background_region = clean(background_region, bckg_sigma, bckg_kernel)
                background = background_region.mean(axis=0)
                background = np.array([background,]*np.shape(data)[2])

                plt.imshow(background, vmin=np.min(background), vmax=np.max(background))
                plt.colorbar()
                plt.title("Modelled background")
                plt.show()
                plt.close()

            data[i, g, :, :] = data[i, g, :, :] - background

            if (i == 0 and g == 0 and show):
                fig, ax, im = img(np.log10(np.abs(data[i, g, :, :])), aspect=5, vmin=None, vmax=None, norm=None)
                plt.show()
                plt.close()

            del(background_region) # recover memory
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

def moffat_frame(frame):
    # Initialize frame model as list.
    P = []
    
    # Iterate on columns.
    for i in range(frame.shape[1]):
        # Grab column to fit.
        col = frame[:,i]
        x = np.array([k for k in range(col.shape[0])])
        # Fit the Moffat profile.
        params = (0, max(col), int(col.shape[0]/2), 1.25, -1.25)
        lb = max(col)*0.99
        ub = max(col)*1.01
        if max(col) < 0:
            lb = max(col)*1.01
            ub = max(col)*0.99
        if max(col) == 0:
            lb = -1e6
            ub = 1e6
        lower_bounds = (-1e6, lb, 0, 1.00, -1.50)
        upper_bounds = (1e6, ub, int(col.shape[0]), 1.50, -1.00)
        opt_result = least_squares(leastsq_input_function, params, bounds=(lower_bounds, upper_bounds), args=(x, col))
        C, c, mu, G, alpha = opt_result.x
        C = 0 # C is effectively the background level, so you *don't* want to subtract it out at this stage.
        moffat_profile = moffat(x, C, c, mu, G, alpha)

        # Append Moffat profile.
        P.append(moffat_profile)
    return np.array(P).T

def moffat(x, C, c, mu, G, alpha):
    r = (x-mu)/G
    return c*((1+r**2)**alpha) + C

def leastsq_input_function(params, x, profile):
    C, c, mu, G, alpha = params
    fit = moffat(x, C, c, mu, G, alpha)
    return (fit - profile)