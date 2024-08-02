from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.cleaning import median_spatial_filter, colbycol_bckg, get_trace_mask
from juniper.util.plotting import img

def glbs(datamodel, inpt_dict):
    """Performs group-level background subtraction on every group in the datamodel according to the instructions in inpt_dict.
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).

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
                      desc = "Correcing integration {}...".format(i),
                      disable=(not time_ints)): # for each group
            # Correct 1/f noise with group-level background subtraction for that group.
            datamodel.data[i,g,:,:], background = colbycol_bckg(data[i,g,:,:],
                                                                inpt_dict["rows"],
                                                                trace_mask)
    
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Group-level background subtraction complete.")
    return datamodel