import os
import time
from tqdm import tqdm

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from scipy import signal
from scipy.optimize import least_squares

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer

def bin_light_curves(spectra, inpt_dict):
    """Bins 1D spectra to return a light curve.

    Args:
        spectra (xarray): its spectrum data_var has shape [detectors, time,
        wavelength], and its waves coord and time coord will be used in binning.
        inpt_dict (_type_): instructions for running this step.

    Returns:
        xarray: light_curves xarray.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Binning 1D spectra to produce broad-band and spectroscopic light curves...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])
    
    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Initialize some lists at the detector level.
    broadband = [] # shape detector x time
    broaderr = [] # shape detector x time
    broadwave = [] # shape detector
    broadbins = [] # shape detector
    spec = [] # shape detector x time x central_wavelength
    specerr = [] # shape detector x time x central_wavelength
    specwave = []  # shape detector x central_wavelength
    specbins = [] # shape detector x wavelength edges

    # And begin processing detectors.
    for d in tqdm(range(spectra.spectrum.shape[0]),
                  desc='Extracting light curves from each detector...',
                  disable=(not time_ints)):
        # Load the spectra and uncertainties and wavelength solutions from this detector.
        spectrum = spectra.spectrum.values[d,:,:] # has shape time x wavelength
        error = spectra.err.values[d,:,:]
        waves = np.median(spectra.waves.values[d,:,:],axis=0) # median wavelength solution on time axis, has shape wavelength/column

        # Set up a mask for NaN and for bad columns.
        column_mask = np.where(np.isnan(spectrum),1,0) # create a nan mask

        # Initialize detector-specific lists.
        spec_det = [] # will have shape time x central_wavelength
        specerr_det = []
        specwave_det = [] # will have shape central_wavelength
        specbins_det = [] # will have shape wavelength edges

        # Check for bad columns.
        if inpt_dict["reject_bad_cols"]:
            if inpt_dict["verbose"] == 2:
                print("Rejecting unstable columns from spectra...")
            
            # Median normalize the 1D spectra over time.
            norm_spec = np.copy(spectrum)/np.median(spectrum,axis=0)
            norm_std = np.std(norm_spec,axis=0)
            compare = signal.medfilt(norm_std,21)
            std_check = np.abs(norm_std - compare)

            # Anywhere that std_check is above the threshold, we kick that column.
            bad_columns = np.argwhere(std_check>inpt_dict["bad_col_thres"])
            column_mask[:,bad_columns] = 1 # update the mask

            if inpt_dict["verbose"] == 2:
                print("{} columns out of {} were masked.".format(len(bad_columns),norm_spec.shape[1]))

            if (plot_ints or save_ints):
                # Create diagnostic plot of how many columns were marked as bad, and what the column std was.
                plt.plot(norm_std, color='k')
                plt.plot(compare, ls='--', color='red')
                plt.title("Standard deviation by column")
                plt.xlabel("column #")
                plt.ylabel("std dev")
                if save_ints:
                    plt.savefig(os.path.join(inpt_dict['plot_dir'],'detector{}_standarddeviation.png'.format(d)),
                                dpi=300, bbox_inches='tight')
                if plot_ints:
                    plt.show(block=True)
                plt.close()

                plt.plot(std_check)
                plt.axhline(inpt_dict["bad_col_thres"], ls='--', color='red')
                plt.title("Measured - expected standard deviation")
                plt.xlabel("column #")
                plt.ylabel("abs(median filtered - true std dev)")
                plt.ylim(0,0.2)
                if save_ints:
                    plt.savefig(os.path.join(inpt_dict['plot_dir'],'detector{}_kickedcolumns.png'.format(d)),
                                dpi=300, bbox_inches='tight')
                if plot_ints:
                    plt.show(block=True)
                plt.close()
        
        # Apply mask whether it is or is not empty.
        spectrum = np.ma.masked_array(spectrum,mask=column_mask)
        error = np.ma.masked_array(error,mask=column_mask)
        waves = np.ma.masked_array(waves,mask=np.median(column_mask,axis=0))

        # The broad-band light curve is trivial.
        broadband_det = np.ma.sum(spectrum,axis=1) # sum on wavelengths to get flux over whole bandpass
        # Broad-band uncertainties sum in quadrature.
        broaderr_det = np.ma.sqrt(np.ma.sum(np.square(error),axis=1)) # sum on wavelengths to get error over whole bandpass
        # Getting the central wavelength is simple.
        broadwave_det = np.ma.median(waves)
        # The wavelength bounds is also straightforward.
        broadbins_det = np.array([np.ma.min(waves),np.ma.max(waves)])
        # Store both the spectrum and each point's uncertainty.
        broadband.append(broadband_det)
        broaderr.append(broaderr_det)
        broadwave.append(broadwave_det)
        broadbins.append(broadbins_det)

        if (plot_step or save_step):
            # Create diagnostic plot of the broad-band light curve.
            plt.errorbar(spectra.time.values[d,:], broadband_det, yerr=broaderr_det, fmt='ko', capsize=3)
            plt.title("Broad-band light curve")
            plt.xlabel("time [mjd]")
            plt.ylabel("flux [a.u.]")
            if save_step:
                plt.savefig(os.path.join(inpt_dict['plot_dir'],'detector{}_broadband_lc.png'.format(d)),
                            dpi=300, bbox_inches='tight')
            if plot_ints:
                plt.show(block=True)
            plt.close()

        # The spec curves are more nuanced.
        if inpt_dict["bin_method"] == "columns":
            # Bin every n_columns columns.
            n_columns = inpt_dict["n_columns"]
            l = 0
            r = n_columns

            # Get expected number of bins.
            expected = int(spectrum.shape[1]/n_columns)
            pbar = tqdm(total = expected,
                        desc='Binning spectroscopic light curves by columns...',
                        disable=(not time_ints))
            while r < spectrum.shape[1]:
                # Get the light curve and uncertainties for this bin.
                bin_spec = np.ma.sum(spectrum[:,l:r],axis=1)
                bin_err = np.ma.sqrt(np.ma.sum(np.square(error[:,l:r]),axis=1))
                bin_wave = np.ma.median(waves[l:r])
                bin_bins = np.array([np.ma.min(waves[l:r]),np.ma.max(waves[l:r])])
        
                # Store both the spectrum and each point's uncertainty.
                spec_det.append(bin_spec)
                specerr_det.append(bin_err)
                specwave_det.append(bin_wave)
                specbins_det.append(bin_bins)

                # Progress bar update.
                pbar.update(1)

                # Advance bins.
                l = r
                r += n_columns
            # Add the final bin.
            bin_spec = np.ma.sum(spectrum[:,l:],axis=1)
            bin_err = np.ma.sqrt(np.ma.sum(np.square(error[:,l:]),axis=1))
            bin_wave = np.ma.median(waves[l:])
            bin_bins = np.array([np.ma.min(waves[l:]),np.ma.max(waves[l:])])
    
            # Store both the spectrum and each point's uncertainty.
            spec_det.append(bin_spec)
            specerr_det.append(bin_err)
            specwave_det.append(bin_wave)
            specbins_det.append(bin_bins)

            # Close the progress bar.
            pbar.close()
        
        elif inpt_dict["bin_method"] == "wavelengths":
            # Bin parts of 1D spec that are in between each wavelength bin.
            wave_bins = inpt_dict["wave_bins"]

            for i in tqdm(range(1,len(wave_bins)),
                          desc='Binning spectroscopic light curves by wavelength...',
                          disable=(not time_ints)):
                wl, wr = wave_bins[i-1],wave_bins[i] # defines edges of this bin
                wave_bin = np.argwhere((waves >= wl and waves <= wr))

                # Get the light curve and uncertainties for this bin.
                bin_spec = np.ma.sum(spectrum[:,wave_bin],axis=1)
                bin_err = np.ma.sqrt(np.ma.sum(np.square(error[:,wave_bin]),axis=1))
                bin_wave = np.ma.median(waves[wave_bin])
                bin_bins = np.array([np.ma.min(waves[wave_bin]),np.ma.max(waves[wave_bin])])
        
                # Store both the spectrum and each point's uncertainty.
                spec_det.append(bin_spec)
                specerr_det.append(bin_err)
                specwave_det.append(bin_wave)
                specbins_det.append(bin_bins)

        # And store.
        spec.append(spec_det)
        specerr.append(specerr_det)
        specwave.append(specwave_det)
        specbins.append(specbins_det)
    
    # FIX: dummy xpos, ypos, widths
    xpos = np.zeros_like(specwave)
    ypos = np.zeros_like(specwave)
    widths = np.zeros_like(specwave)

    # Now create an xarray out of this info.
    light_curves = xr.Dataset(data_vars=dict(
                                    broadband=(["detector", "time"], broadband),
                                    broaderr=(["detector", "time"], broaderr),
                                    broadwave=(["detector",],broadwave),
                                    broadbins=(["detector",],broadbins),
                                    spec=(["detector", "wavelength", "time"], spec),
                                    specerr=(["detector", "wavelength", "time"], specerr),
                                    specwave=(["detector", "wavelength"],specwave),
                                    specbins=(["detector","wavelength",],specbins),
                                    xpos=(["detector", "wavelength"],xpos),
                                    ypos=(["detector", "wavelength"],ypos),
                                    widths=(["detector", "wavelength"],widths),
                                    ),
                        coords=dict(
                               time = (["detector", "time"], spectra.time.values),
                               detector = (["detector",], [i for i in range(spectra.spectrum.shape[0])]),
                               details = (["detector","observation_mode"], spectra.details.values), # this has the form Ndetectors x [[INSTRUMENT, DETECTOR, FILTER, GRATING]]
                               ),
                        attrs=dict(
                              )
                              )
    
    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
    return light_curves