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
        inpt_dict (dict): instructions for running this step.

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

        # Set up a mask for NaN, inf, zero-waves, and bad columns.
        column_mask = np.where(np.isnan(spectrum),1,0) # create a nan mask
        column_mask = np.where(np.isinf(spectrum),1,column_mask) # mask anywhere the spectrum is inf too
        column_mask = np.where(waves==0,1,column_mask) # and ignore places with no wavelengths

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
                    plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_standarddeviation.png'.format(d)),
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
                    plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_kickedcolumns.png'.format(d)),
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
                plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_broadband_lc.png'.format(d)),
                            dpi=300, bbox_inches='tight')
            if plot_step:
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
                if bin_bins[0] == bin_bins[1]:
                    # Should only happen if len(wavs[l:r]) == 1.
                    try:
                        next_wave = np.ma.median(waves[r:r+n_columns])
                        hw = (next_wave-bin_wave)/2
                        bin_bins=np.array([bin_wave-hw,bin_wave+hw])
                    except:
                        last_wave = specwave_det[-1]
                        hw = (bin_wave-last_wave)/2
                        bin_bins=np.array([bin_wave-hw,bin_wave+hw])
        
                # Store both the spectrum and each point's uncertainty.
                spec_det.append(bin_spec)
                specerr_det.append(bin_err)
                specwave_det.append(bin_wave)
                specbins_det.append(bin_bins)

                if (plot_ints or save_ints):
                    # Create diagnostic plot of this spec light curve.
                    plt.errorbar(spectra.time.values[d,:], bin_spec, yerr=bin_err, fmt='ko', capsize=3)
                    plt.title("Spectroscopic light curve")
                    plt.xlabel("time [mjd]")
                    plt.ylabel("flux [a.u.]")
                    if save_ints:
                        plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_spec{}um_lc.png'.format(d,np.round(bin_wave,3))),
                                    dpi=300, bbox_inches='tight')
                    if plot_ints:
                        plt.show(block=True)
                    plt.close()

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
            if bin_bins[0] == bin_bins[1]:
                last_wave = specwave_det[-1]
                hw = (bin_wave-last_wave)/2
                bin_bins=np.array([bin_wave-hw,bin_wave+hw])
    
            # Store both the spectrum and each point's uncertainty.
            spec_det.append(bin_spec)
            specerr_det.append(bin_err)
            specwave_det.append(bin_wave)
            specbins_det.append(bin_bins)

            if (plot_ints or save_ints):
                # Create diagnostic plot of this spec light curve.
                plt.errorbar(spectra.time.values[d,:], bin_spec, yerr=bin_err, fmt='ko', capsize=3)
                plt.title("Spectroscopic light curve")
                plt.xlabel("time [mjd]")
                plt.ylabel("flux [a.u.]")
                if save_ints:
                    plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_spec{}um_lc.png'.format(d,np.round(bin_wave,3))),
                                dpi=300, bbox_inches='tight')
                if plot_ints:
                    plt.show(block=True)
                plt.close()

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
                if bin_bins[0] == bin_bins[1]:
                    # Should only happen if len(wavs[l:r]) == 1.
                    try:
                        next_wave_bin = np.argwhere((waves >= wave_bins[i] and waves <= wave_bins[i+1]))
                        next_wave = np.ma.median(waves[next_wave_bin])
                        hw = (next_wave-bin_wave)/2
                        bin_bins=np.array([bin_wave-hw,bin_wave+hw])
                    except:
                        last_wave = specwave_det[-1]
                        hw = (bin_wave-last_wave)/2
                        bin_bins=np.array([bin_wave-hw,bin_wave+hw])
        
                # Store both the spectrum and each point's uncertainty.
                spec_det.append(bin_spec)
                specerr_det.append(bin_err)
                specwave_det.append(bin_wave)
                specbins_det.append(bin_bins)

                if (plot_ints or save_ints):
                    # Create diagnostic plot of this spec light curve.
                    plt.errorbar(spectra.time.values[d,:], bin_spec, yerr=bin_err, fmt='ko', capsize=3)
                    plt.title("Spectroscopic light curve")
                    plt.xlabel("time [mjd]")
                    plt.ylabel("flux [a.u.]")
                    if save_ints:
                        plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_spec{}um_lc.png'.format(d,np.round(bin_wave,3))),
                                    dpi=300, bbox_inches='tight')
                    if plot_ints:
                        plt.show(block=True)
                    plt.close()

        # And store.
        spec.append(spec_det)
        specerr.append(specerr_det)
        specwave.append(specwave_det)
        specbins.append(specbins_det)

    # Need these in case you don't want to bin in time.
    xpos = spectra.xpos.values
    ypos = spectra.ypos.values
    widths = spectra.widths.values
    t = spectra.time.values

    # If asked, bin in time.
    if inpt_dict["bin_time"]:
        if inpt_dict["verbose"] >= 1:
            print("Binning light curves down in time...")
            
        # Get the bin size once.
        s = inpt_dict["bin_size"]

        # Start by binning time itself.
        t_new = []
        for d in range(t.shape[0]):
            t_new.append(time_bin(t[d,:],s,'median'))

        t = np.array(t_new)

        # We can initialize a lot of empty arrays this way.
        broadband_new = np.empty_like(t)
        broaderr_new = np.empty_like(t)
        xpos_new = np.empty_like(t)
        ypos_new = np.empty_like(t)
        widths_new = np.empty_like(t)

        for d in range(t.shape[0]):
            broadband_new[d,:] = time_bin(broadband[d],s,'mean')
            broaderr_new[d,:] = time_bin(broaderr[d],s,'quadrature')
            xpos_new[d,:] = time_bin(xpos[d],s,'median')
            ypos_new[d,:] = time_bin(ypos[d],s,'median')
            widths_new[d,:] = time_bin(widths[d],s,'median')

        broadband = broadband_new
        broaderr = broaderr_new
        xpos = xpos_new
        ypos = ypos_new
        widths = widths_new

        # The spec and spec_err need slightly special treatment.
        spec_new = np.empty((t.shape[0],len(spec[0]),t.shape[1]))
        specerr_new = np.empty((t.shape[0],len(specerr[0]),t.shape[1]))

        for d in range(t.shape[0]):
            for l in range(spec_new.shape[1]):
                spec_new[d,l,:] = time_bin(spec[d][l],s,'mean')
                specerr_new[d,l,:] = time_bin(specerr[d][l],s,'quadrature')

        spec = spec_new
        specerr = specerr_new

        if (plot_step or save_step):
            # Create diagnostic plot of the binned broad-band light curve.
            for d in range(t.shape[0]):
                plt.errorbar(t[d,:], broadband[d,:], yerr=broaderr[d,:], fmt='ko', capsize=3)
                plt.title("Broad-band light curve")
                plt.xlabel("time [mjd]")
                plt.ylabel("flux [a.u.]")
                if save_step:
                    plt.savefig(os.path.join(inpt_dict['plot_dir'],'S5_detector{}_binnedbroadband_lc.png'.format(d)),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show(block=True)
                plt.close()


    # Now create an xarray out of this info.
    light_curves = xr.Dataset(data_vars=dict(
                                    broadband=(["detector", "time"], broadband),
                                    broaderr=(["detector", "time"], broaderr),
                                    broadwave=(["detector",],broadwave),
                                    broadbins=(["detector", "edge"],broadbins),
                                    spec=(["detector", "wavelength", "time"], spec),
                                    specerr=(["detector", "wavelength", "time"], specerr),
                                    specwave=(["detector", "wavelength"],specwave),
                                    specbins=(["detector", "wavelength", "edge"],specbins),
                                    xpos=(["detector", "time"],xpos),
                                    ypos=(["detector", "time"],ypos),
                                    widths=(["detector", "time"],widths),
                                    ),
                        coords=dict(
                               time = (["detector", "time"], t),
                               detectors = (["detector",], [i for i in range(spectra.spectrum.shape[0])]),
                               details = (["detector","observation_mode"], spectra.details.values), # this has the form Ndetectors x [[INSTRUMENT, DETECTOR, FILTER, GRATING]]
                               ),
                        attrs=dict(
                              )
                              )
    
    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
    return light_curves

def time_bin(array, bin_size, mode='sum'):
    """Simple function to bin an array down in size.

    Args:
        array (np.array): the array to bin down.
        bin_size (int): how many items should go into the bin.
        mode (str, optional): how to combine the values. Options are 'sum',
        'mean', 'median', or 'quadrature'. Defaults to 'sum'.
    
    Returns:
        np.array: the input array reduced in size.
    """
    # Initialize new array as list.
    binned = []

    # Ensure no overflow.
    if bin_size > array.shape[0]:
        bin_size = array.shape[0] - 1

    # Then, iterate.
    ind = 0
    while ind+bin_size < array.shape[0]:
        trim = array[ind:ind+bin_size]
        if mode == 'sum':
            binned.append(np.ma.sum(trim))
        if mode == 'mean':
            binned.append(np.ma.mean(trim))
        if mode == 'median':
            binned.append(np.ma.median(trim))
        if mode == 'quadrature':
            binned.append(np.sqrt(np.ma.sum(np.square(trim)))/len(trim))
        ind += bin_size
    
    trim = array[ind:]
    if len(trim) == 0:
        pass
    else:
        if mode == 'sum':
            binned.append(np.ma.sum(trim))
        if mode == 'mean':
            binned.append(np.ma.mean(trim))
        if mode == 'median':
            binned.append(np.ma.median(trim))
        if mode == 'quadrature':
            binned.append(np.sqrt(np.ma.sum(np.square(trim)))/len(trim))
    
    return binned