import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.plotting import img

def iterate_fixed(segments, inpt_dict):
    """Iterate a fixed number of times at specified sigmas to remove cosmic rays.

    Args:
        segments (xarray): its segments.data data_vars has the integrations and its dq data_vars contains the data quality flags to be updated.
        inpt_dict (dict): instructions for how to run this step.

    Returns:
        xarray: segments with removed cosmic rays and data quality flags updated.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Iterating fixed number of times to remove cosmic rays...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Track outliers removed and where they were found.
    bad_pix_map = np.zeros_like(segments.data.values)
    
    # Start iterating.
    for sigma in inpt_dict["fixed_sigmas"]:
        # Compute median image and std deviation.
        med = np.ma.median(segments.data, axis=0)
        std = np.ma.std(segments.data, axis=0)

        # Track bad pixels in this sigma.
        bad_pix_this_sigma = 0

        for k in tqdm(range(segments.data.shape[0]),
                      desc='Iterating over sigma=%.2f...'%sigma,
                      disable=(not time_ints)):
            # Look for where outliers are in this frame and flag with 1.
            S = np.where(np.abs(segments.data.values[k,:,:] - med) > sigma*std, 1, 0)

            # Count outliers and locate them.
            S = np.where(bad_pix_map[k,:,:] == 1, 0, S) # if the pixel was already reported as bad from a previous step, then don't double count.
            bad_pix_this_sigma += np.count_nonzero(S)
            bad_pix_map[k,:,:] += S

            # If time_replace is not None, we replace the bad pixels instead of just masking them with the DQ array.
            if inpt_dict["time_replace"]:
                correction = med
                if inpt_dict["time_replace"] != 'all':
                    # Take the median of the frames that are +/- time_replace away from the current frame.
                    l = k - inpt_dict["time_replace"]
                    r = k + inpt_dict["time_replace"]
                    # Cut at edges.
                    if l < 0:
                        l = 0
                    if r > segments.data.shape[0]:
                        r = segments.data.shape[0]
                    correction = np.median(segments.data.values[l:r,:,:],axis=0)
                # And replace with correction.
                segments.data.values[k,:,:] = np.where(S == 1, correction, segments.data.values[k,:,:])
            # Otherwise, just mask the bad pixels.
            else:
                segments.data.values[k,:,:] = np.ma.masked_array(segments.data.values[k,:,:],
                                                                 mask=bad_pix_map[k,:,:])
            
        # Report how many bad pixels this sigma trimmed.
        if inpt_dict["verbose"] == 2:
            print("Bad pixels flagged at sigma=%.2f: %.0f"%(sigma, bad_pix_this_sigma))
    
    # Update data flags.
    segments.dq.values = np.where(bad_pix_map != 0, 1, segments.dq.values)

    if (plot_step or save_step):
        # Create plots of the entire bad_pix_map collapsed in on itself in time.
        bad_pix_alltime = np.sum(bad_pix_map,axis=0)
        bad_pix_alltime[bad_pix_alltime>0] = 1
        fig, ax, im = img(bad_pix_alltime, aspect=5, title='Fixed-iteration DQ flags',
                          vmin=0, vmax=1, norm='linear', verbose=inpt_dict["verbose"])
        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_iterate-fixed_flags.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()
    
    if (plot_ints or save_ints):
        # Create plots of each frame of the bad_pix_map in time.
        for i in range(bad_pix_map.shape[0]):
            fig, ax, im = img(bad_pix_map[i,:,:], aspect=5, title='Fixed-iteration DQ flags',
                              vmin=0, vmax=1, norm='linear', verbose=inpt_dict["verbose"])
            if save_step:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_iterate-fixed_flags_int{}.png".format(i)),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show()
            plt.close()

    # Report outliers found.
    if inpt_dict["verbose"] >= 1:
        print("Iterations complete. Total bad pixels found: %.0f."% np.count_nonzero(bad_pix_map))

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)

    return segments

def iterate_free(segments, inpt_dict):
    """Iterate an unspecified number of times at a fixed sigma to remove cosmic rays.

    Args:
        segments (xarray): its segments.data data_vars has the integrations and its dq data_vars contains the data quality flags to be updated.
        inpt_dict (dict): instructions for how to run this step.

    Returns:
        xarray: segments with removed cosmic rays and data quality flags updated.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Iterating free number of times to remove cosmic rays...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Track outliers removed and where they were found, and open sigma once.
    bad_pix_map = np.zeros_like(segments.data.values)
    sigma = inpt_dict["free_sigma"]

    # Check force stop iteration condition.
    cut_off = np.inf
    if inpt_dict["free_cutoffs"]:
        cut_off = inpt_dict["free_cutoffs"]
    
    # Start iterating.
    for i in tqdm(range(segments.data.shape[1]),
                  desc='Iterating over pixel time series...',
                  disable=(not time_ints)):
        for j in range(segments.data.shape[2]):
            # Claim outlier found and track steps.
            outliers_found = 1
            step = 0
            
            # Then, iterate until no outliers found, or until cut off kicks in.
            while (outliers_found > 0 and step < cut_off):
                # Compute median pixel time series and std deviation.
                med = np.median(segments.data.values[:,i,j])
                std = np.std(segments.data.values[:,i,j])

                # Check for outliers.
                S = np.where(np.abs(segments.data.values[:,i,j]-med) > sigma*std, 1, 0)

                # Count outliers and locate them.
                S = np.where(bad_pix_map[:,i,j] == 1, 0, S) # if the pixel was already reported as bad from a previous step, then don't double count.
                outliers_found = np.count_nonzero(S)
                bad_pix_map[:,i,j] += S

                # If time_replace is not None, we replace the bad pixels instead of just masking them with the DQ array.
                if inpt_dict["time_replace"]:
                    correction = med
                    if inpt_dict["time_replace"] != 'all':
                        # Smooth the pixel's time series over with a running median rather than a median in all time.
                        correction = medfilt(segments.data.values[:,i,j], kernel_size=(2*inpt_dict["time_replace"]+1))
                    # And replace with correction.
                    segments.data.values[:,i,j] = np.where(S == 1, correction, segments.data.values[:,i,j])

                # Advance another step.
                step += 1     

            # Report that the iteration limit was reached.
            if (step > cut_off and inpt_dict["verbose"] == 1):
                print("Pixel {}, {} hit iteration limit.".format(i,j))
    
    # Update data flags.
    segments.dq.values = np.where(bad_pix_map != 0, 1, segments.dq.values)

    if (plot_step or save_step):
        # Create plots of the entire bad_pix_map collapsed in on itself in time.
        bad_pix_alltime = np.sum(bad_pix_map,axis=0)
        bad_pix_alltime[bad_pix_alltime>0] = 1
        fig, ax, im = img(bad_pix_alltime, aspect=5, title='Free-iteration DQ flags',
                          vmin=0, vmax=1, norm='linear', verbose=inpt_dict["verbose"])
        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_iterate-free_flags.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()
    
    if (plot_ints or save_ints):
        # Create plots of each frame of the bad_pix_map in time.
        for i in range(bad_pix_map.shape[0]):
            fig, ax, im = img(bad_pix_map[i,:,:], aspect=5, title='Free-iteration DQ flags',
                              vmin=0, vmax=1, norm='linear', verbose=inpt_dict["verbose"])
            if save_step:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_iterate-free_flags_int{}.png".format(i)),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show()
            plt.close()

    # Report outliers found.
    if inpt_dict["verbose"] >= 1:
        print("Iterations complete. Total bad pixels found: %.0f."% np.count_nonzero(bad_pix_map))

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)

    return segments