import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

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

    # Time this step, if asked.
    if time_step:
        t0 = time.time()

    # Protect certain flags, if asked.
    if inpt_dict["skip_flags"]:
        for flag in tqdm(inpt_dict["skip_flags"],
                         desc='Removing protected flags...',
                         disable=(not time_ints)):
            # If this flag is not to be considered, 0 it out.
            segments.dq.values = np.where(segments.dq.values == flag, 0, segments.dq.values)

    # Turn dqflags into mask arrays, and add nan mask.
    dq_mask = np.empty_like(segments.dq.values)
    dq_mask[:, :, :] = np.where(segments.dq.values > 0, 1, 0)
    dq_mask[np.isnan(segments.data.values)] = 1

    if (plot_step or save_step):
        # Create plots of the entire dq_mask collapsed in on itself in time.
        dq_alltime = np.sum(dq_mask,axis=0)
        dq_alltime[dq_alltime>0] = 1
        fig, ax, im = img(dq_alltime, aspect=5, title='JWST DQ flags',
                          vmin=0, vmax=1, norm='linear', verbose=inpt_dict["verbose"])
        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_JWST_flags.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()
    
    if (plot_ints or save_ints):
        # Create plots of each frame of the dq_mask in time.
        for i in range(dq_mask.shape[0]):
            fig, ax, im = img(dq_mask[i,:,:], aspect=5, title='JWST DQ flags',
                              vmin=0, vmax=1, norm='linear', verbose=inpt_dict["verbose"])
            if save_step:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_JWST_flags_int{}.png".format(i)),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show()
            plt.close()

    # The replacement method can be median time, median space, or None to leave as masked.
    if inpt_dict["flag_replace"] == 'time':
        if inpt_dict["verbose"] == 2:
            print("Replacing flagged pixels with the median in time...")
        # One filtered image as the median in time.
        replacement = np.median(segments.data.values,axis=0)
        # And replace.
        segments.data.values = np.where(dq_mask > 0, replacement, segments.data.values)

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
        segments.data.values = np.where(dq_mask > 0, replacement, segments.data.values)

    else:
        # No replacement. We ran this step purely to log JWST flags as 1s and 0s, and exclude flags of our choosing.
        pass

    # In the future, having JWST flag information stored in 1s or 0s rather than even integers will be helpful.
    if inpt_dict["verbose"] == 2:
        print("Converting dq array to 1s and 0s for later steps...")

    # Update data flags.
    segments.dq.values = np.where(dq_mask != 0, 1, segments.dq.values)

    # Count how many pixels were replaced.
    if inpt_dict["verbose"] >= 1:
        print("{} flagged pixels were masked or replaced.".format(np.count_nonzero(dq_mask)))

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
    return segments