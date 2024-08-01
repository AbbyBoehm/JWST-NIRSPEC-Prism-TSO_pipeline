from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from util.diagnostics import tqdm_translate, plot_translate
from util.cleaning import median_spatial_filter, colbycol_bckg, get_trace_mask
from util.plotting import img

def glbs_all(datamodel, inpt_dict):
    """Performs group-level background subtraction on every group in the datamodel according to the instructions in inpt_dict.

    Args:
        datamodel (jwst.datamodel): A datamodel containing attribute .data, which is an np array of shape nints x ngroups x nrows x ncols, produced during wrap_front_end.
        inpt_dict (dict): A dictionary containing instructions for performing this step.

    Returns:
        jwst.datamodel: datamodel with updated cleaned .data attribute.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Group-level background subtraction processing...")
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_plots = plot_translate(inpt_dict["save_plots"])

    # Copy data.
    data = np.copy(datamodel.data)

    # Iterate over frames.
    for i in tqdm(range(data.shape[0]),
                  desc = "Removing 1/f noise from integrations...",
                  disable=(not time_step)): # for each integration
        # Obtain the mask that hides the trace using a cleaned version of the very last group in this integration.
        trace_mask = np.zeros_like(data[i,-1,:,:])
        if inpt_dict["mask"]:
            trace_mask = get_trace_mask(median_spatial_filter(np.copy(data[i,-1,:,:]),
                                                              sigma=inpt_dict["sigma"],
                                                              kernel=inpt_dict["kernel"]))
            
        for g in tqdm(range(data.shape[1]),
                      desc = "Correcing integration {}, group {}...".format(i, g),
                      disable=(not time_ints)): # for each group
            # Correct 1/f noise with group-level background subtraction for that group.
            datamodel.data[i,g,:,:], background = glbs_one(data[i,g,:,:],
                                                           bckg_rows=inpt_dict["rows"],
                                                           trace_mask=trace_mask)
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Group-level background subtraction complete.")
    return datamodel

def glbs_one(data, bckg_rows=[], trace_mask=None):
    """Performs 1/f subtraction on the given array.
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).

    Args:
        data (np.array): 2D row x col array from jwst.datamodel.
        bckg_rows (list, optional): list of integers which defines the background rows. Defaults to [].
        trace_mask (np.ma.masked_array, optional): mask to hide trace pixels with. Defaults to None.

    Returns:
        np.array: data array with column-by-column noise removed, and background of that noise.
    """
    data, background = colbycol_bckg(data, bckg_rows, trace_mask)
    return data, background