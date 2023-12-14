import os
import glob
from copy import deepcopy

import numpy as np
import time

import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from scipy import signal

from .utils import stitch_files

def doStage4(filesdir, outdir,
             trace_aperture={"hcut1":0,
                             "hcut2":-1,
                             "vcut1":0,
                             "vcut2":-1},
             mask_unstable_pixels={"skip":False,
                                   "threshold":1.0},
             extract_light_curves={"skip":False,
                                   "binmode":"wavelength",
                                   "columns":None,
                                   "min_wav":None,
                                   "max_wav":None,
                                   "wavbins":np.linspace(0.6,5.3,70),
                                   "ext_type":"box",
                                   "badcol_threshold":1.5},
             median_normalize_curves={"skip":False,
                                      "event_type":"transit"},
             sigma_clip_curves={"skip":False,
                                "b":100,
                                "clip_at":5},
             fix_mirror_tilts={"skip":False,
                               "threshold":0.01,
                               "known_index":None},
             fix_transit_times={"skip":False,
                                "epoch":None},
             plot_light_curves={"skip":False,
                                "event_type":"transit"},
             save_light_curves={"skip":False}
            ):
    '''
    Performs Stage 4 extractions on the files located at filepaths.
    
    :param filesdir: str. Folder holding the postprocessed_*.fits files you want to extract spectra from.
    :param outdir: str. Location where you want output images and text files to be saved to.
    :param trace_aperture: dict. Keywords are "hcut1", "hcut2", "vcut1", "vcut2", all integers denoting the rows and columns respectively that define the edges of the aperture bounding the trace.
    :param mask_unstable_pixels: dict. Keywords are "skip", "threshold". Whether to mask pixels whose normalized standard deviations are higher than the threshold value.
    :param extract_light_curves: dict. Keywords are "skip", "binmode", "columns", "min_wav", "max_wav", "wavbins", "ext_type", "badcol_threshold". Whether to extract light curves using the given wavelength bin edges or column number with wavelength limits, the specified extraction type (options are "box", "polycol", "polyrow", "smooth", "medframe"), and a threshold for cutting out bad columns.
    :param median_normalize_curves. dict. Keywords are "skip", "event_type". Whether to normalize curves by their median value. "event_type" specifies "transit" or "eclipse".
    :param sigma_clip_curves: dict. Keywords are "skip", "b", "clip_at". Whether to sigma-clip the light curves every b points, rejecting outliers at clip_at sigma.
    :param fix_mirror_tilts: dict. Keywords are "skip", "threshold". Whether to look for and fix mirror tilt events.
    :param fix_transit_times: dict. Keywords are "skip", "epoch". Whether to fix the transit times so that epoch becomes the 0-point.
    :param plot_light_curves: dict. Keyword is "skip". Whether to save plots of the light curves.
    :param save_light_curves: dict. Keyword is "skip". Whether to save .txt files of the light curves.
    :return: .txt files of curves extracted from the postprocessed_*.fits files and images related to extraction.
    '''
    t0 = time.time()
    # Get the files to analyze.
    filepaths = sorted(glob.glob(os.path.join(filesdir, "*.fits")))
    print("Performing Stage 4 Juniper extractions of spectra from the data located at: {}".format(filepaths))
    
    # Grab the needed info from the file.
    segments, errors, segstarts, wavelengths, dqflags, times, frames_to_reject = stitch_files(filepaths)
    
    # Build the aperture object.
    aperture = np.ones(np.shape(segments))
    aperture[:,
             trace_aperture["hcut1"]:trace_aperture["hcut2"],
             trace_aperture["vcut1"]:trace_aperture["vcut2"]] = 0
    
    if not mask_unstable_pixels["skip"]:
        # Mask any pixel that has standard deviation normalized greater than 1.
        masked = 0
        print("Masking trace pixels with time variations above the specified value...")
        for y in range(trace_aperture["hcut1"],trace_aperture["hcut2"]):
            for x in range(trace_aperture["vcut1"],trace_aperture["vcut2"]):
                try:
                    if np.std(segments[:,y,x]/np.median(segments[:,y,x])) > mask_unstable_pixels["threshold"]:
                        aperture[:,y,x] = 1
                        masked += 1
                except IndexError:
                    pass
        n_trace_pixels = (trace_aperture["hcut2"]-trace_aperture["hcut1"])*(trace_aperture["vcut2"]-trace_aperture["vcut1"])
        print("Masked {} trace pixels out of {} for standard deviation above the threshold (fraction of {}).".format(masked, n_trace_pixels, np.round(masked/n_trace_pixels, 5)))
    
    # Should not skip extract light curves! Rest of code breaks.
    if not extract_light_curves["skip"]:
        wlc, slc, times, central_lams, wavelength_bin_edges = extract_curves(segments, errors, times, aperture, segstarts, wavelengths, frames_to_reject,
                                                                             binmode=extract_light_curves["binmode"],
                                                                             columns=extract_light_curves["columns"],
                                                                             wavelength_range=(extract_light_curves["min_wav"],extract_light_curves["max_wav"]),
                                                                             wavbins=extract_light_curves["wavbins"],
                                                                             ext_type=extract_light_curves["ext_type"],
                                                                             badcol_threshold=extract_light_curves["badcol_threshold"])

    if not median_normalize_curves["skip"]:
        wlc = median_normalize(wlc,
                               event_type=median_normalize_curves["event_type"])
        for i, lc in enumerate(slc):
            slc[i] = median_normalize(lc,
                                      event_type=median_normalize_curves["event_type"])
    
    if not sigma_clip_curves["skip"]:
        wlc = clip_curve(wlc,
                         b=sigma_clip_curves["b"],
                         clip_at=sigma_clip_curves["clip_at"])
        for i, lc in enumerate(slc):
            slc[i] = clip_curve(lc,
                                b=sigma_clip_curves["b"],
                                clip_at=sigma_clip_curves["clip_at"])
    
    if not fix_mirror_tilts["skip"]:
        wlc, slc = correct_mirror_tilt(wlc,
                                       slc,
                                       threshold=fix_mirror_tilts["threshold"],
                                       known_index=fix_mirror_tilts["known_index"])

            
    if not fix_transit_times["skip"]:
        print("Fixing transit/eclipse timestamps...")
        times=fix_times(times, wlc=wlc,
                        epoch=fix_transit_times["epoch"])
        print("Fixed.")
        
    if not plot_light_curves["skip"]:
        print("Generating output plots of extracted light curves...")
        imgs_outdir = os.path.join(outdir, "output_imgs_extraction")
        if not os.path.exists(imgs_outdir):
            os.makedirs(imgs_outdir)
        plot_curve(times, wlc, plot_light_curves["event_type"], "White light curve", "wlc", imgs_outdir)
        for lc, central_lam in zip(slc, central_lams):
            plot_curve(times, lc, plot_light_curves["event_type"], "Spectroscopic light curve at %.3f micron" % central_lam, "slc_%.3fmu" % central_lam, imgs_outdir)
        print("Plots generated.")
    
    if not save_light_curves["skip"]:
        print("Writing all curves to .txt files...")
        txts_outdir = os.path.join(outdir, "output_txts_extraction")
        if not os.path.exists(txts_outdir):
            os.makedirs(txts_outdir)
        write_light_curve(times, wlc, "wlc", txts_outdir)
        write_wvs_file(wavelength_bin_edges[0][0], wavelength_bin_edges[-1][1],-99,"wlc_wvs", txts_outdir)
        for i, lc in enumerate(slc):
            write_light_curve(times, lc, "slc_%.3fmu_%.3fmu" % (wavelength_bin_edges[i][0],wavelength_bin_edges[i][1]), txts_outdir)
            write_wvs_file(wavelength_bin_edges[i][0],wavelength_bin_edges[i][1],central_lams[i],"slc_%.3fmu_%.3fmu_wvs" % (wavelength_bin_edges[i][0],wavelength_bin_edges[i][1]), txts_outdir)
        print("Files written.")
    print("Stage 4 calibrations completed in %.3f minutes." % ((time.time() - t0)/60))

def extract_curves(segments, errors, times, aperture, segstarts, wavelengths, frames_to_reject, binmode, columns, wavelength_range, wavbins, ext_type, badcol_threshold):    
    '''
    Extract a white light curve and spectroscopic light curves from the trace.
    
    :param segments: 3D array. Integrations x rows x cols of data.
    :param errors: 3D array. Integrations x rows x cols of uncertainties.
    :param times: 1D array. Timestamps of integrations.
    :param aperture: 3D array. Mask that defines where the trace is.
    :param segstarts: list of ints. Defines where new segment files begin.
    :param wavelengths: list of lists of floats. The wavelength solutions for each segment.
    :param frames_to_reject: list of ints. Frames that will not be added into the light curve.
    :param binmode: str. Choices are "columns" or "wavelengths". Whether to bin by wavelength or by pixel column.
    :param columns: int. If binmode is "columns", bin every N columns.
    :param wavelength_range: tup of floats. If binmode is "columns", the min and max wavelength that will be included.
    :param wavbins: list of floats. The edges defining each spectroscopic light curve. The ith bin will count pixels that have wavelength solution wavbins[i] <= wav < wavbins[i+1].
    :param ext_type: str. Choices are "box" or "polycol". The first is a standard extraction, while all the others define types of optimal extraction apertures.
    :param badcol_threshold: float. Threshold at which to toss a column for being too unstable.
    :return: corrected timestamps, median-normalized white light curve, and median-normalized spectroscopic light curve.
    '''
    # Update aperture so that for every frame in it, where wavelength is nan becomes masked (1).
    for i in range(aperture.shape[0]):
        aperture[i,:,:][np.isnan(wavelengths[0])] = 1
    # Get just the trace that you want to sum over.
    trace = np.ma.masked_array(segments, aperture)

    # Update aperture to exclude bad columns.
    test_oneDspec = np.nansum(trace,axis=1)
    test_oneDspec = test_oneDspec/np.median(test_oneDspec,axis=0)
    test_oneDspec = np.std(test_oneDspec,axis=0)
    medfilt_test = signal.medfilt(test_oneDspec, 7)
    test_diff = np.abs(test_oneDspec - medfilt_test)
    threshold = badcol_threshold*np.mean(test_diff)
    plt.plot(test_oneDspec, color='k')
    plt.plot(medfilt_test, ls='--', color='red')
    plt.title("Standard deviation by column")
    plt.xlabel("column #")
    plt.ylabel("std dev")
    plt.show()
    plt.close()
    satisfied = False
    while not satisfied:
        plt.plot(test_diff)
        plt.axhline(threshold, ls='--', color='red')
        plt.title("Where standard deviation is unexpectedly high")
        plt.xlabel("column #")
        plt.ylabel("abs(median filtered - true std dev)")
        plt.ylim(0,0.2)
        plt.show()
        plt.close()
        change = int(input("Change threshold? Yes (1) or no (0): "))
        if change == 0:
            satisfied = True
        elif change == 1:
            threshold = float(input("New threshold: "))
    bad_columns = np.argwhere(test_diff>threshold)
    print("Identified and masked {} bad columns out of {} (fraction of {:.4f}).".format(len(bad_columns),len(test_oneDspec),len(bad_columns)/len(test_oneDspec)))
    for i in range(aperture.shape[0]):
        for col in bad_columns:
            aperture[i,:,col] = 1
    trace = np.ma.masked_array(segments, aperture)
    
    # Initialize 1Dspec objects.
    oneDspec = []
    times_with_skips = []

    t0 = time.time()
    masks_built_yet = 0
    if ext_type != "box":
        print("Building spatial profile for optimal extraction...")
        spatial_profile = get_spatial_profile(trace, ext_type=ext_type)
        print("Built.")
    for k in range(np.shape(segments)[0]):
        if k in frames_to_reject:
            print("Integration %.0f will be skipped." % k)
        else:
            # Not a rejected frame, so proceed.
            print("Gathering 1D spectrum of integration %.0f..." % k)
            times_with_skips.append(times[k])
            
            # When we are at the start of a new segment, we have to rebuild the wavelength masks.
            if (k in segstarts or masks_built_yet == 0):
                masks = []
                central_lams = []
                wavelength_bin_edges = []
                for i in range(1):
                    if (k == segstarts[i] or masks_built_yet == 0):
                        wavelength = wavelengths[i]
                if binmode == "wavelengths":
                    # Build masks that contain certain wavelength ranges.
                    print("Building wavelength masks...")
                    for j, w in enumerate(wavbins):
                        if w == wavbins[-1]:
                            # Don't build a bin at the end of the wavelength range.
                            pass
                        else:
                            central_lams.append((wavbins[j]+wavbins[j+1])/2)
                            wavelength_bin_edges.append((wavbins[j], wavbins[j+1]))
                            mask_step1 = np.where(wavelength <= wavbins[j+1], wavelength, 0)
                            mask_step2 = np.where(mask_step1 >= wavbins[j], mask_step1, 0)
                            mask = np.where(mask_step2 != 0, 1, 0)
                            # Now we want to invert it, setting all 0s to 1s and vice versa. Only 0s will be allowed through the mask.
                            mask = np.where(mask == 1, 0, 1)
                            # Finally, where the wavelength solution is nan, mask it.
                            mask[np.isnan(wavelength)] = 1
                            '''
                            plt.imshow(mask)
                            plt.title("Mask for wavelength {}".format(central_lams[-1]))
                            plt.show()
                            plt.close()
                            '''
                            masks.append([mask])
                elif binmode == "columns":
                    # Build masks that contain a specified number of columns.
                    print("Building column masks...")
                    #for i in range(trace.shape[2]):
                    #    print(i, wavelength[12,i])
                    masked_wavs = np.ma.masked_array(wavelength, aperture[0,:,:])
                    left_edge = 0
                    right_edge = columns
                    while right_edge < trace.shape[2]: # array is of shape time x row x col
                        minimum_wavelength = np.ma.min(masked_wavs[:,left_edge:right_edge])
                        maximum_wavelength = np.ma.max(masked_wavs[:,left_edge:right_edge])
                        if (minimum_wavelength < wavelength_range[0] or maximum_wavelength > wavelength_range[1]):
                            # Do not take masks that are outside of the accepted wavelength range.
                            pass
                        else:
                            # Get the central wavelength of this region.
                            central_lams.append(np.ma.mean(masked_wavs[:,left_edge:right_edge]))
                            wavelength_bin_edges.append((minimum_wavelength,
                                                        maximum_wavelength))
                            # Make a fully opaque mask.
                            mask = np.ones_like(wavelength)
                            # Open it up only in the region you want to take.
                            mask[:,left_edge:right_edge] = 0
                            # Remask so that nan wavelength regions are masked again.
                            mask[np.isnan(wavelength)] = 1
                            '''
                            plt.imshow(mask)
                            plt.title("Mask for wavelength {}, columns {}, {}".format(central_lams[-1], left_edge, right_edge))
                            plt.show()
                            plt.close()
                            '''
                            masks.append([mask])
                        # Advance the column edges up.
                        left_edge += columns
                        right_edge += columns
                    if left_edge < trace.shape[2]:
                        # One last mask is needed to get the last columns.
                        # Get the central wavelength of this region.
                        central_lams.append(np.ma.mean(masked_wavs[:,left_edge:]))
                        # Make a fully opaque mask.
                        mask = np.ones_like(wavelength)
                        # Open it up only in the region you want to take.
                        mask[:,left_edge:] = 0
                        # Remask so that nan wavelength regions are masked again.
                        mask[np.isnan(wavelength)] = 1
                        masks.append([mask])
                masks_built_yet = 1
                print("Masks built.")
            
            if ext_type != "box":
                # Have to perform optimal extraction.
                profile  = spatial_profile[k,:,:]
                errors2  = errors[k,:,:]**2
                errors2 -= trace[k,:,:]
                errors2[errors2<=0] = 10**-8
                f = np.nansum(trace[k,:,:], axis=0)
                V = errors2+np.abs(f[np.newaxis,:]*profile)
                trace[k,:,:] = (profile * trace[k,:,:] / V)/np.sum(profile ** 2 / V, axis=0)
            
            spectrum = []
            for mask in masks:
                spectrum.append(np.nansum(np.ma.masked_array(np.ma.masked_array(np.copy(trace[k, :, :]), mask), aperture[k, :, :])))
            oneDspec.append(spectrum)
            print("Collected spectrum.")
        if (k%1000 == 0 and k != 0):
            elapsed_time = time.time()-t0
            iterrate = k/elapsed_time
            iterremain = np.shape(segments)[0] - k
            print("On integration %.0f. Elapsed time is %.3f seconds." % (k, elapsed_time))
            print("Average rate of integration processing: %.3f ints/s." % iterrate)
            print("Estimated time remaining: %.3f seconds.\n" % (iterremain/iterrate))
    print("Gathered 1D spectra in %.3f minutes." % ((time.time()-t0)/60))
    # We now have the oneDspec object. We're going to sum the oneDspec into a wlc,
    # then reorganize the oneDspec into a bunch of spectroscopic light curves.
    print("Producing wlc from 1D spectra...")
    wlc = []
    for spectra in oneDspec:
        wlc.append(np.nansum(spectra))
    print("Generated wlc.")
    
    slc = []
    print("Reshaping oneDspec into slc...")
    for i, w in enumerate(wavelength_bin_edges):
        lc = []
        for j in range(len(wlc)):
            lc.append(oneDspec[j][i])
        slc.append(np.array(lc))
    # Now each object in slc is a full time series corresponding to just one wavelength bin.
    print("Reshaped. Returning wlc and slc...")
    
    return wlc, slc, times_with_skips, np.round(central_lams, 3), wavelength_bin_edges

def get_spatial_profile(segments, ext_type="polycol"):
    '''
    Builds a spatial profile for optimal extraction.

    :param segments: 3D array. Integrations x nrows x ncols of data, masked to include only the data being extracted.
    :param ext_type: str. Choices are "polycol", "polyrow", "medframe", "smooth". Type of spatial profile to build.
    If ext_type "polycol", profile is an optimum profile.
    '''
    P = np.empty_like(segments)
    # If the type is "medframe", we can build the profile right away.
    if ext_type == "medframe":
        median_frame = medframe(segments)

    # Iterate through frames.
    for k in range(np.shape(P)[0]):
        if ext_type == "polycol":
            P[k,:,:] = polycol(segments[k,:,:], poly_order=4, threshold=3)
        if ext_type == "polyrow":
            P[k,:,:] = polyrow(segments[k,:,:], poly_order=10, threshold=3)
        if ext_type == "medframe":
            P[k,:,:] = median_frame
        if ext_type == "smooth":
            P[k,:,:] = smooth(segments[k,:,:], poly_order=4, threshold=3)
    return P

def polycol(trace, poly_order=4, threshold=3):
    '''
    Computes spatial profile fit to the trace using a polynomial of the specified order, fitting along columns.
    
    :param trace: 2D array. A frame out of the trace[y,x,t] array, form trace[y,x].
    :param order: int. Order of polynomial to fit as the spatial profile.
    :param threshold: float. Sigma threshold at which to mask polynomial fit outliers.
    :return: P[x,y] array. An array profile to use for optimal extraction.
    '''
    # Initialize P as a list.
    P = []
    
    # Iterate on columns.
    for i in range(np.shape(trace)[1]):
        col = deepcopy(trace[:,i])
        j = 0
        while True:
            p_coeff = np.polyfit(range(np.shape(col)[0]),col,deg=poly_order)
            p_col = np.polyval(p_coeff, range(np.shape(col)[0]))

            res = np.array(col-p_col)
            dev = np.abs(res)/np.std(res)
            max_dev_idx = np.argmax(dev)

            j += 1
            if (dev[max_dev_idx] > threshold and j < 20):
                try:
                    col[max_dev_idx] = (col[max_dev_idx-1]+col[max_dev_idx+1])/2
                except IndexError:
                    col[max_dev_idx] = np.median(p_col)
                continue
            else:
                break
        P.append(p_col)
    P = np.array(P).T
    P[P < 0] = 0 # enforce positivity
    P /= np.sum(P,axis=0) # normalize on columns
    return P

def polyrow(trace, poly_order=10, threshold=3):
    '''
    Computes spatial profile fit to the trace using a polynomial of the specified order, fitting along rows.
    
    :param trace: 2D array. A frame out of the trace[t,y,x] array, form trace[y,x].
    :param order: int. Order of polynomial to fit as the dispersion profile.
    :param threshold: float. Sigma threshold at which to mask polynomial fit outliers.
    :return: P[x,y] array. An array profile to use for optimal extraction.
    '''
    # Initialize P as a list.
    P = []
    
    # Iterate on rows.
    for i in range(np.shape(trace)[0]):
        row = deepcopy(trace[i,:])
        j = 0
        while True:
            p_coeff = np.polyfit(range(np.shape(row)[0]),row,deg=poly_order)
            p_row = np.polyval(p_coeff, range(np.shape(row)[0]))

            res = np.array(row-p_row)
            dev = np.abs(res)/np.std(res)
            max_dev_idx = np.argmax(dev)

            j += 1
            if (dev[max_dev_idx] > threshold and j < 20):
                try:
                    row[max_dev_idx] = (row[max_dev_idx-1]+row[max_dev_idx+1])/2
                except IndexError:
                    row[max_dev_idx] = np.median(p_row)
                continue
            else:
                break
        P.append(p_row)
    P = np.array(P)
    P[P < 0] = 0 # enforce positivity
    P /= np.sum(P,axis=0) # normalize on columns
    return P

def medframe(trace):
    median_frame = np.median(trace, axis=0)
    median_frame[median_frame < 0] = 0
    median_frame = median_frame/np.sum(median_frame, axis=0)
    return median_frame

def smooth(trace, threshold=3, window_len=21):
    '''
    Median-filters every row in the frame following the Eureka! implementation, enforces positivity, and normalizes to get a profile for extraction.
    
    :param trace: 2D array. A frame out of the trace[t,y,x] array, form trace[y,x].
    :param threshold: float. Sigma threshold at which to mask smooth fit outliers.
    :param window_len: int. Length in pixels of the window used for smoothing the rows.
    :return: P 2D array. An array P[y,x] for optimal extraction.
    '''
    # Initialize P as empty list.
    P = []
    
    # Iterate on rows.
    for i in range(np.shape(trace)[0]):
        j = 0
        row = deepcopy(trace[i])
        for ind in range(np.shape(row)[0]):
            row[ind] /= np.median(row[np.max(0,ind-10):ind+11])
        while True:
            x = deepcopy(row)
                
            # Smooth row.
            s = np.r_[2*np.median(x[0:window_len//5])-x[window_len:1:-1], x,
                      2*np.median(x[-window_len//5:])-x[-1:window_len:-1]]
            
            w = np.hanning(window_len)
            
            p_row = np.convolve(w/w.sum(),s,mode='same')
            p_row = p_row[window_len-1:window_len+1]

            res = np.array(row-p_row)
            dev = np.abs(res)/np.std(res)
            max_dev_idx = np.argmax(dev)
            
            j += 1
            if (dev[max_dev_idx] > threshold and j < 20):
                try:
                    row[max_dev_idx] = (row[max_dev_idx-1]+row[max_dev_idx+1])/2
                except IndexError:
                    row[max_dev_idx] = np.median(p_row)
                continue
            else:
                break
        P.append(p_row)
    P = np.array(P)
    P[P < 0] = 0 # enforce positivity
    P /= np.sum(P,axis=0) # normalize on columns
    return P

def median_normalize(lc, event_type):
    '''
    Normalizes given light curve by its median value.

    :param lc: 1D array. Flux values for a given light curve.
    :param event_type: str. Specifies "eclipse" or "transit", normalizes by the correct median. If unspecified, normalize by median of entire light curve.
    :return: light curve 1D array divided by its median.
    '''
    if event_type == "transit":
        med = np.median(lc[0:int(0.3*len(lc))])
    elif event_type == "eclipse":
        med = np.median(lc[int(0.3*len(lc)):int(0.6*len(lc))])
    else:
        med = np.median(lc)
    return lc/med

def fix_times(times, wlc=None, epoch=None):
    '''
    Fixes times in the times array so that the mid-transit time is 0.
    
    :param times: 1D array. Times in MJD, not corrected for mid-transit. If both wlc and epoch are None, the mean of this object is used as the epoch.
    :param wlc: 1D array or None. If not None and epoch is None, the epoch is defined as the time when wlc hits its minimum.
    :param epoch: float. If not None, the mid-transit time used to correct the times arrays.
    :return: corrected times array.
    '''
    if epoch is None:
        if wlc is not None:
            minimum_value = min(wlc)
            epoch = times[wlc.index(minimum_value)]
        else:
            epoch = np.mean(times)
        
    for i in range(len(times)):
        times[i] = times[i] - epoch
        
    return times

def clip_curve(lc, b, clip_at):
    '''
    Sigma clip the given light curve.

    :param lc: 1D array. The light curve that you want to clip outliers from.
    :param b: int. The width of the bin that you want to clip outliers in. Should be narrow enough that it doesn't grab the entire transit at once - otherwise, it will sigma clip the transit right out of existence.
    :param clip_at: float. Threshold at which to reject outliers from the light curve.
    :return: 1D array of light curve with outliers clipped out.
    '''
    clipcount = 0
    for i in np.arange(0, len(lc)+b, b):
        try:
            # Sigma-clip this segment of wlc and fill the clipped parts with the median.
            smed = np.median(lc[i:i+b])
            ssig = np.std(lc[i:i+b])
            clipcount += np.count_nonzero(np.where(np.abs(lc[i:i+b]-smed) > clip_at*ssig, 1, 0))
            lc[i:i+b] = np.where(np.abs(lc[i:i+b]-smed) > clip_at*ssig, smed, lc[i:i+b])
        except IndexError:
            if len(lc[i:-1]) == 0:
                pass
            else:
                lc[i:-1] = sigma_clip(lc[i:-1], sigma=clip_at)
                lc[i:-1] = lc[i:-1].filled(fill_value=np.ma.median(lc[i:-1]))
    print("Clipped %.0f values from the given light curve." % clipcount)
    return lc

def correct_mirror_tilt(wlc, slc, threshold=0.01, known_index=None):
    '''
    Iterates over every light curve looking for mirror tilt events to correct.

    :param wlc: 1D array. The white light curve. This curve is used to determine whether a tilt event happened.
    :param slc: lst of 1D array. The spectroscopic light curves.
    :param threshold: float. The change in flux expected of a mirror tilt event. If a jump in flux of this magnitude is detected, claim detection of a mirror tilt event.
    :param known_index: None or int. Supply this if you already know the index of the tilt event.
    :return: wlc and slc arrays with mirror tilt events corrected.
    '''
    if known_index != None:
        tilt_occurred, where = (True, known_index)
    else:
        tilt_occurred, where = detect_mirror_tilt(wlc, threshold=threshold)
    if tilt_occurred:
        print("A mirror tilt event was detected in the white light curve. Correcting all given curves...")
        plt.plot(wlc)
        wlc = fix_mirror_tilt(wlc, where)
        plt.plot(wlc)
        plt.title("WLC before and after correction")
        plt.show()
        plt.close()
        for i, lc in enumerate(slc):
            slc[i] = fix_mirror_tilt(lc, where)
        print("Mirror tilt event corrected. Returning fixed arrays...")
        return wlc, slc
    else:
        print("No mirror tilt event detected.")
        return wlc, slc

def detect_mirror_tilt(lc, threshold=0.01):
    '''
    Checks the pre- and post-transit residuals to see if a mirror tilt event occurred.

    :param lc: 1D array. The light curve that you want to search for a mirror tilt event in.
    :param threshold: float. The change in flux expected of a mirror tilt event. If a jump in flux of or exceeding this value is detected, claim detection of a mirror tilt event.
    :return: bool indicating whether a mirror tilt is believed to have occurred.    
    '''
    n = lc.shape[0]
    pre_med = np.nanmedian(lc[0:int(n*0.20)])
    post_med = np.nanmedian(lc[int(n*0.80):n])
    Delta_tilt = np.abs(pre_med - post_med)
    if Delta_tilt > threshold:
        print("Within the threshold of {}, a mirror tilt event is believed to have occurred.".format(threshold))
        print("Determining index at which tilt event happened...")
        index_found = False
        bad_index = 0
        for i in np.arange(n-1,0,-1):
            if not index_found:
                Delta = np.abs(lc[i] - lc[i-1])
                if Delta >= 0.75*Delta_tilt:
                    print("Possible trip found at index {}?".format(i))
                    # Need this to trip once and not many times.
                    next_Delta = np.abs(lc[i-1] - lc[i-2])
                    if next_Delta >= 0.75*Delta_tilt:
                        print("False alarm, checking other indices...")
                    else:
                        print("Confirmed tilt event happened at that index.")
                        bad_index = i
                        index_found = True
        if bad_index == 0:
            print("Failed to find the index of the mirror tilt?")
            print(1/0)
        return True, bad_index
    else:
        return False, 'NA'

def fix_mirror_tilt(lc, where):
    '''
    Corrects for a detected mirror tilt event.
    
    :param lc: 1D array. The light curve that you want to correct a mirror tilt event in.
    :param where: int. The index at which the mirror tilt event first sets in.
    :return: lc 1D array with the tilt evengt corrected.
    '''
    n = lc.shape[0]
    pre_med = np.nanmedian(lc[0:int(n*0.20)])
    post_med = np.nanmedian(lc[int(n*0.80):n])
    Delta_tilt = pre_med - post_med
    for i in range(where, n):
        lc[i] += Delta_tilt
    return lc
    

def plot_curve(t, lc, event_type, title, outfile, outdir):
    '''
    Creates and saves a plot of the given light curve to the outdir.

    :param t: 1D array. Timestamps of each exposure in the light curve.
    :param lc: 1D array. Flux values in the light curve.
    :param event_type: str. "transit" or "eclipse".
    :param title: str. Title of the plot.
    :param outfile: str. Name of the file you want to save, sans the ".pdf" extension.
    :param outdir: str. Directory where the light curve plot will be saved to.
    :return: light curve plot .pdf saved to outdir/outfile.
    '''
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.scatter(t, lc, s=3)
    ax.set_xlabel("time since mid-{} [days]".format(event_type))
    ax.set_ylabel("relative flux [no units]")
    ax.set_title(title)
    plt.savefig(os.path.join(outdir, outfile + ".pdf"), dpi=300)
    plt.close()

def write_light_curve(t, lc, outfile, outdir):
    '''
    Writes light curve to .txt file for future reading.

    :param t: 1D array. Timestamps of each exposure in the light curve.
    :param lc: 1D array. Flux values in the light curve.
    :param outfile: str. Name of the file you want to save, sans the ".txt" extension.
    :param outdir: str. Directory where the light curve file will be saved to.
    :return: light curve file saved to outdir/outfile.
    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    with open('{}/{}.txt'.format(outdir,outfile), mode='w') as file:
        file.write("#time[MJD]     flux[normalized]\n")
        for ti, lci in zip(t, lc):
            file.write('{:8}   {:8}\n'.format(ti, lci))

def write_wvs_file(wavmin, wavmax, central_lam, outfile, outdir):
    '''
    Writes wavelengths to .txt file for future reading.

    :param wavmin: float. The left edge of the wavelength bin.
    :param wavmax: float. The right edge of the wavelength bin.
    :param central_lam: float. The central wavelength of the bin.
    :param outfile: str. Name of the file you want to save, sans the ".txt" extension.
    :param outdir: str. Directory where the wavelengths file will be saved to.
    :return: wavelengths file saved to outdir/outfile.
    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    with open('{}/{}.txt'.format(outdir,outfile), mode='w') as file:
        file.write("#wavelength [mu]\n")
        for w in (wavmin, wavmax, central_lam):
            file.write('{:8}\n'.format(w))