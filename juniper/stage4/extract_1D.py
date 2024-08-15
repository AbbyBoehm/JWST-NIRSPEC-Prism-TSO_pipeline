import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer

def box(segments, inpt_dict):
    """The simplest form of spectal extraction, this method sums the flux
    contained in the aperture with no weighting.

    Args:
        segments (xarray): its data DataSet contains the integrations to
        sum across, and its wavelengths DataSet can be used to limit the
        1D extraction to a certain wavelength range.
        inpt_dict (dict): instructions for running this step.

    Returns:
        np.array, np.array, np.array: the one-dimensional spectrum and
        corresponding wavelengths and errors for that spectrum.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Extracting 1D spectrum with standard (box) method...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Initialize arrays.
    oneD_spec = np.empty((segments.data.shape[0],segments.data.shape[2])) # it has shape nints x ncols
    oneD_err = np.empty((segments.data.shape[0],segments.data.shape[2])) # same shape as oneD_spec
    wav_sols = np.empty((segments.data.shape[0],segments.data.shape[2])) # same shape as oneD_spec

    # And populate.
    for i in tqdm(range(oneD_spec.shape[0]),
                  desc = 'Extracting spectra from each integration...',
                  disable=(not time_ints)):
        # Get the integration, wavelengths, and data quality.
        d = segments.data.values[i,:,:]
        e = segments.err.values[i,:,:]
        w = segments.wavelengths.values[i,:,:]
        dq = segments.dq.values[i,:,:]

        # Start building a mask.
        mask = np.zeros_like(d)
        mask = np.where(np.isnan(w),1,mask) # if the wavelength solution is nan, mask the pixel
        mask = np.where(np.isnan(d),1,mask) # and mask any nans in the data itself
        if inpt_dict["wavelengths"]:
            # Mask wavelengths that are too short.
            mask = np.where(w < inpt_dict["wavelengths"][0],1,mask)
            # And wavelengths that are too long.
            mask = np.where(w > inpt_dict["wavelengths"][1],1,mask)

        if inpt_dict["mask_bad_pix"]:
            # Mask pixels that were flagged by the data quality array.
            mask = np.where(dq != 0, 1, mask)
        
        # Mask where is outside of the aperture.
        lower, upper = inpt_dict["aperture"]
        mask[0:lower] = 1
        mask[upper:-1] = 1

        # Now apply the mask to the data and sum it on columns.
        d = np.ma.masked_array(d, mask=mask)
        oneD_spec[i,:] = np.ma.sum(d,axis=0)

        # Apply the mask to the errors, which add in quadrature.
        e = np.ma.masked_array(e, mask=mask)
        oneD_err[i,:] = np.sqrt(np.ma.sum(np.square(e),axis=0))

        # Apply the mask to the wavelengths and median it on columns.
        w = np.ma.masked_array(w, mask=mask)
        wav_sols[i,:] = np.ma.median(w,axis=0)

        if (plot_step or save_step) and i==0:
            plt.imshow(mask)
            plt.title('1D extraction mask')
            if save_step:
                plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_1D_extraction_mask_frame0.png'),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show(block=True)
            plt.close()
        if (plot_ints or save_ints):
            plt.imshow(mask)
            plt.title('1D extraction mask')
            if save_ints:
                plt.savefig(os.path.join(inpt_dict['plot_dir'],'S4_1D_extraction_mask_frame{}.png'.format(i)),
                            dpi=300, bbox_inches='tight')
            if plot_ints:
                plt.show(block=True)
            plt.close()

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
    
    return oneD_spec, oneD_err, wav_sols

def optimum(segments, inpt_dict):
    """Extracts 1D spectra using the methods of Horne 1986.

    Args:
        segments (xarray): its data DataSet contains the integrations to
        sum across, and its wavelengths DataSet can be used to limit the
        1D extraction to a certain wavelength range.
        inpt_dict (dict): instructions for running this step.

    Returns:
        _type_: _description_
    """
    # Placeholder!
    oneD_spec, oneD_err, wav_sols = [], [], []
    return oneD_spec, oneD_err, wav_sols