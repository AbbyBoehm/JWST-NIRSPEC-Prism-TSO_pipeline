from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.cleaning import median_spatial_filter, colbycol_bckg, get_trace_mask
from juniper.util.plotting import img

def subtract_background(segments, inpt_dict):
    """Performs background subtraction on every integration in segments according to the instructions in inpt_dict.
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).

    Args:
        segments (xarray): Its segments.data object will have its background removed.
        inpt_dict (dict): A dictionary containing instructions for performing this step.

    Returns:
        xarray: segments.data with background removed.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Integration-level background subtraction processing...")
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_plots = plot_translate(inpt_dict["save_plots"])

    # Obtain the mask that hides the trace using the median frame.
    trace_mask = np.zeros_like(segments.data[0,:,:])
    if inpt_dict["mask"]:
        trace_mask = get_trace_mask(median_spatial_filter(np.med(segments.data.values,axis=0),
                                                          sigma=inpt_dict["bckg_sigma"],
                                                          kernel=inpt_dict["bckg_kernel"]))

    # Iterate over frames.
    for i in tqdm(range(segments.data.shape[0]),
                  desc = "Removing 1/f noise from calibrated integrations...",
                  disable=(not time_step)): # for each integration
        # Correct 1/f noise with group-level background subtraction for that group.
        segments.data.values[i,:,:], background = colbycol_bckg(segments.data.values[i,:,:],
                                                                inpt_dict["bckg_rows"],
                                                                trace_mask)
    
    if inpt_dict["verbose"] >= 1:
        print("Integration-level background subtraction complete.")
    return segments