import time
import numpy as np

from juniper.util.cleaning import median_spatial_filter
from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.plotting import img

def mask_flags(segments, inpt_dict):
    """Uses the jwst pipeline data quality flags to mask bad pixels.

    Args:
        segments (xarray): segments.data contains the integrations and segments.dq contains the data quality flags.
        inpt_dict (dict): instructions for how to process this step.

    Returns:
        xarray: segments with the flagged pixels masked.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Masking pixels flagged by the jwst pipeline...")
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # TIme this step, if asked.
    if time_step:
        t0 = time.time()

    # Protect certain flags, if asked.
    if inpt_dict["skip_flags"]:
        for flag in inpt_dict["skip_flags"]:
            # If this flag is not to be considered, 0 it out.
            segments.dq.values = segments.dq.where(segments.dq.values == flag, 0, segments.dq.values)

    # Turn dqflags into mask arrays, and add nan mask.
    dq_mask = np.empty_like(segments.dq.values)
    dq_mask[:, :, :] = np.where(segments.dq.values > 0, 1, 0)
    dq_mask[np.isnan(segments)] = 1

    # The replacement method can be median time, median space, or None to leave as masked.
    if inpt_dict["flag_replace"] == 'time':
        if inpt_dict["verbose"] == 2:
            print("Replacing flagged pixels with the median in time...")
        # One filtered image as the median in time.
        replacement = np.median(segments.data.values,axis=0)
        # And replace.
        segments.data = segments.data.where(dq_mask > 0, replacement, segments.data)

    elif inpt_dict["flag_replace"] == 'space':
        if inpt_dict["verbose"] == 2:
            print("Replacing flagged pixels with spatially median-filtered values...")
        # Produce a median-filtered spatial image for each frame.
        replacement = np.empty_like(segments.data.values)
        for k in range(replacement.shape[0]):
            replacement[k,:,:] = median_spatial_filter(segments.data[k].values,
                                                       inpt_dict["flag_sigma"],
                                                       inpt_dict["flag_kernel"])
        # And replace.
        segments.data = segments.data.where(dq_mask > 0, replacement, segments.data)

    else:
        if inpt_dict["verbose"] == 2:
            print("Masking flagged pixels using np.ma module...")
        # No replacement specified, mask instead.
        segments.data = np.ma.masked(segments.data, mask=dq_mask)

    # Count how many pixels were replaced.
    if inpt_dict["verbose"] >= 1:
        print("{} flagged pixels were masked or replaced.".format(np.count_nonzero(dq_mask)))

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
    return segments