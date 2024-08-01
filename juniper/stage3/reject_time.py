from tqdm import tqdm

import numpy as np

from juniper.util.diagnostics import tqdm_translate, plot_translate

def iterate_fixed(segments, inpt_dict):
    """Iterate a fixed number of times at specified sigmas to remove cosmic rays.

    Args:
        segments (xarray): its segments.data data_vars has the integrations and its dq data_vars contains the data quality flags to be updated.
        inpt_dict (dict): instructions for how to run this step.
        sigmas (list): list of floats. Each float is a sigma threshold and the number of sigmas in the list is how many iterations will be performed.
        replacement (int, optional): if None, replace outlier pixels with median in time. If int, replace with median of int values either side in time. Defaults to None.
        verbose (int, optional): from 0 to 2. How much logging to do. Defaults to 0.
        time_ints (bool, optional): whether to report progress with tqdm. Defaults to False.

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

    # Track outliers removed and where they were found.
    bad_pix_map = np.zeros_like(segments.data.values)
    
    # Start iterating.
    for sigma in inpt_dict["sigmas"]:
        # Compute median image and std deviation.
        med = np.median(segments.data, axis=0)
        std = np.std(segments.data, axis=0)

        # Track bad pixels in this sigma.
        bad_pix_this_sigma = 0

        for k in tqdm(range(segments.data.shape[0]),
                      desc='Iterating over sigma=%.2f...'%sigma,
                      disable=(not time_ints)):
            # Look for where outliers are in this frame and flag with 1.
            S = np.where(np.abs(segments.data.values[k,:,:] - med) > sigma*std, 1, 0)

            # Locate outliers in the bad pixel map.
            bad_pix_map[k,:,:] += S
            bad_pix_this_sigma += np.count_nonzero(S)

            # If replacement is not None, custom replacement.
            correction = med
            if inpt_dict["replacement"]:
                # Take the median of the frames that are +/- replacement away from the current frame.
                l = k - inpt_dict["replacement"]
                r = k + inpt_dict["replacement"]
                # Cut at edges.
                if l < 0:
                    l = 0
                if r > segments.data.shape[0]:
                    r = segments.data.shape[0]
                correction = np.median(segments.data.values[l:r,:,:],axis=0)
            # And replace with correction.
            segments.data[k,:,:] = np.where(S == 1, correction, segments.data[k,:,:])

        if inpt_dict["verbose"] == 2:
            print("Bad pixels flagged at sigma=%.2f: %.0f"%(sigma, bad_pix_this_sigma))
    
    # Update data flags.
    segments.dq = np.where(bad_pix_map != 0, 1, segments.dq)

    # Report outliers found.
    if inpt_dict["verbose"] >= 1:
        print("Iterations complete. Total bad pixels found: %.0f."% np.count_nonzero(bad_pix_map))

    return segments

def iterate_free(segments, inpt_dict):
    """Iterate an unspecified number of times at a fixed sigma to remove cosmic rays.

    Args:
        segments (xarray): its segments.data data_vars has the integrations and its dq data_vars contains the data quality flags to be updated.
        inpt_dict (dict): instructions for how to run this step.
        sigma (float): the sigma threshold at which to reject outliers.
        replacement (int, optional): if None, replace outlier pixels with median in time. If int, replace with median of int values either side in time. Defaults to None.
        cut_off (int, optional): number of iterations to cut off at, useful if it gets stuck. Can be None. Defaults to 100.
        verbose (int, optional): from 0 to 2. How much logging to do. Defaults to 0.
        time_ints (bool, optional): whether to report progress with tqdm. Defaults to False.

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

    # Track outliers removed and where they were found, and open sigma once.
    bad_pix_map = np.zeros_like(segments.data.values)
    sigma = inpt_dict["sigma"]

    # Check force stop iteration condition.
    if not cut_off:
        cut_off = np.inf
    
    # Start iterating.
    for i in tqdm(range(segments.data.shape[1]),
                  desc='Iterating over pixel time series...',
                  disable=(not time_ints)):
        for j in range(segments.data.shape[2]):
            # Claim outlier found and track steps.
            outliers_found = 1
            k = 0
            
            # Then, iterate until no outliers found, or until cut off kicks in.
            while (outliers_found > 0 and k < cut_off):
                # Compute median pixel time series and std deviation.
                med = np.median(segments.data[:,i,j])
                std = np.std(segments.data[:,i,j])

                # Check for outliers.
                S = np.where(np.abs(segments.data[:,i,j]-med) > sigma*std, 1, 0)

                # Count outliers and locate them.
                outliers_found = np.count_nonzero(S)
                bad_pix_map[:,i,j] += S

                # If replacement is not None, custom replacement.
                correction = med
                if inpt_dict["replacement"]:
                    # Take the median of the frames that are +/- replacement away from the current frame.
                    l = k - inpt_dict["replacement"]
                    r = k + inpt_dict["replacement"]
                    # Cut at edges.
                    if l < 0:
                        l = 0
                    if r > segments.data.shape[0]:
                        r = segments.data.shape[0]
                    correction = np.median(segments.data.values[l:r,i,j])
                # And replace with correction.
                segments.data[:,i,j] = np.where(S == 1, correction, segments.data[:,i,j])
            if (k > cut_off and inpt_dict["verbose"] == 1):
                print("Pixel {}, {} hit iteration limit.".format(i,j))
    
    # Update data flags.
    segments.dq = np.where(bad_pix_map != 0, 1, segments.dq)

    # Report outliers found.
    if inpt_dict["verbose"] >= 1:
        print("Iterations complete. Total bad pixels found: %.0f."% np.count_nonzero(bad_pix_map))

    return segments