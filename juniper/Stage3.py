import os
import glob
from copy import deepcopy
import shutil

import numpy as np
import time

import matplotlib.pyplot as plt
from astropy import modeling
from astropy.stats import sigma_clip
from astropy.io import fits
from scipy.ndimage import median_filter

from .utils import img, stitch_files
from .Stage1 import clean

def doStage3(filesdir, outdir,
             trace_aperture={"hcut1":0,
                             "hcut2":-1,
                             "vcut1":0,
                             "vcut2":-1},
             frames_to_reject = [],
             identified_bad_pixels = [],
             loss_stats_step={"skip":False},
             mask_flagged_pixels={"skip":False},
             iteration_outlier_removal={"skip":False, "n":2, "sigma":10},
             spatialfilter_outlier_removal={"skip":False, "sigma":3, "kernel":(1,15)},
             laplacianfilter_outlier_removal={"skip":False, "sigma":50},
             second_bckg_subtract={"skip":False,"bckg_rows":[0,1,2,3,4,5,6,-6,-5,-4,-3,-2,-1], "sigma":3},
             track_source_location={"skip":False,"reject_disper":True,"reject_spatial":True}
            ):
    '''
    Performs custom Stage 3 calibrations on all *_calints.fits files located in the filesdir.
    Can be run on *.fits files that have already been run on this step, if you want only to
    load the data from those *.fits files.
    
    :param filesdir: str. Directory where the *_calints.fits files you want to calibrate are stored.
    :param outdir: str. Directory where you want the additionally-calibrated .fits files to be stored, as well as any output images for reference.
    :param trace_aperture: dict. Keywords are "hcut1", "hcut2", "vcut1", "vcut2", all integers denoting the rows and columns respectively that define the edges of the aperture bounding the trace.
    :param frames_to_reject: list of int. Indices of frames that you want to reject, for reasons like a high-gain antenna move, a satellite crossing, or an anomalous drop/rise in flux.
    :param loss_stats: dict. Keywords are "skip". Whether to track stats of lost trace pixels. Useful if you are worried you are masking or replacing too much of the trace.
    :param mask_flagged_pixels: dict. Keywords are "skip". Whether to mask pixels flagged by the JWST DQ array.
    :param iteration_outlier_removal: dict. Keywords are "skip", "n", "sigma". Whether to iterate n times over each pixel's time series and reject outliers at a given sigma threshold.
    :param spatialfilter_outlier_removal: dict. Keywords are "skip", "sigma", "kernel". Whether to filter spatial outliers in the given kernel, with outliers being flagged at a given sigma threshold.
    :param laplacianfilter_outlier_removal: dict. Keywords are "skip", "sigma". Whether to reject outliers at a given sigma threshold as detected in a Laplacian filtered image.
    :param second_bckg_subtract: dict. Keywords are "skip", "bckg_rows", "sigma". Whether to perform additional background subtraction using the given background row indices, first using the given sigma to clean the background region of outliers.
    :param track_source_location: dict. Keywords are "skip", "reject_disper", "reject_spatial". Whether to track the location of the trace and reject outliers in the dispersion and/or spatial direction.
    :return: calibrated postprocessed_*.fits files in the outdir.
    '''
    print("Running Stage 3 Juniper calibrations on files located in: ", filesdir)
    t0 = time.time()
    
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(os.path.join(outdir, "output_imgs_calibration")):
        os.makedirs(os.path.join(outdir, "output_imgs_calibration"))
    
    files = sorted(glob.glob(os.path.join(filesdir,'*_calints.fits')))
    print("Performing Stage 3 calibrations on the following files:")
    t0 = time.time()
    for file in files:
        with fits.open(file) as file:
            print(file[0].header["FILENAME"])
    
    # First, we need to stitch all the segments together. This step is not optional.
    segments, errors, segstarts, wavelengths, dqflags, times = stitch_files(files)
    
    print("Read all files and collected needed data.")
    print("Creating the aperture for extraction of the data and saving an image of it for reference...")
    aperture = np.ones(np.shape(segments))
    aperture[:,
             trace_aperture["hcut1"]:trace_aperture["hcut2"],
             trace_aperture["vcut1"]:trace_aperture["vcut2"]] = 0
    fig, ax, im = img(aperture[0, :, :], aspect=5)
    plt.savefig(os.path.join(outdir, "output_imgs_calibration/trace_aperture.pdf"), dpi=300)
    plt.close(fig)
    fig, ax, im = img(segments[0,
                               trace_aperture["hcut1"]:trace_aperture["hcut2"],
                               trace_aperture["vcut1"]:trace_aperture["vcut2"]],
                      aspect=5)
    plt.savefig(os.path.join(outdir, "output_imgs_calibration/trace_extracted_region.pdf"), dpi=300)
    plt.close(fig)
    print("Aperture created.")
    
    if not loss_stats_step["skip"]:
        num_pixels_lost_to_DQ_flags = loss_stats(dqflags, trace_aperture, outdir)
        print("%.0f pixels were marked by DQ flags." % num_pixels_lost_to_DQ_flags)
    
    if not mask_flagged_pixels["skip"]:
        segments, num_pixels_lost_by_flagging = mask_flagged(segments, dqflags, trace_aperture)
    else:
        num_pixels_lost_by_flagging = 0
    
    if not iteration_outlier_removal["skip"]:
        segments, num_pixels_lost_by_iteration = iterate_outlier_removal(segments, dqflags, trace_aperture,
                                                                         n=iteration_outlier_removal["n"],
                                                                         sigma=iteration_outlier_removal["sigma"])
    else:
        num_pixels_lost_by_iteration = 0
    
    if not spatialfilter_outlier_removal["skip"]:
        segments, num_pixels_lost_by_filtering = spatial_outlier_removal(segments, trace_aperture,
                                                                         sigma=spatialfilter_outlier_removal["sigma"],
                                                                         kernel=spatialfilter_outlier_removal["kernel"])
    
    else:
        num_pixels_lost_by_filtering = 0
        
    if not laplacianfilter_outlier_removal["skip"]:
        segments, num_pixels_lost_by_laplacian = laplacian_outlier_removal(segments, errors, trace_aperture,
                                                                           sigma=laplacianfilter_outlier_removal["sigma"],
                                                                           verbose=True)
    else:
        num_pixels_lost_by_laplacian = 0
    
    if identified_bad_pixels:
        print("You have flagged {} pixels manually as bad. These will be replaced by local column median.".format(len(identified_bad_pixels)))
        for coords in identified_bad_pixels:
            # lst of tup, each tup is x, y of bad pixel.
            x, y = coords
            for i in range(np.shape(segments)[0]):
                segments[i,y,x] = np.median(segments[i,y-2:y+3,x])
    
    if not second_bckg_subtract["skip"]:
        segments = bckg_subtract(segments,
                                 bckg_rows=second_bckg_subtract["bckg_rows"],
                                 sigma=second_bckg_subtract["sigma"])
    
    # If, after all these corrections, NaN pixels remain, 0 them out.
    num_pixels_masked_NaN = 0
    for i in range(trace_aperture["hcut1"],trace_aperture["hcut2"]):
        for j in range(trace_aperture["vcut1"],trace_aperture["vcut2"]):
            try:
                if np.isnan(np.sum(segments[:,i,j])):
                    segments[:,i,j] = 0
                    num_pixels_masked_NaN += len(segments[:,i,j])
            except IndexError:
                pass
    print("Identified and masked {} NaN pixels in the trace.".format(num_pixels_masked_NaN))

    if not track_source_location["skip"]:
        frames_rejected_by_source_motion = gaussian_source_track(segments,
                                                                 reject_dispersion_direction=track_source_location["reject_disper"],
                                                                 reject_spatial_direction=track_source_location["reject_spatial"])
        for frame in frames_rejected_by_source_motion:
            if frame not in frames_to_reject:
                frames_to_reject.append(frame)
    
    totalmasked = num_pixels_lost_by_flagging + num_pixels_lost_by_iteration + num_pixels_lost_by_filtering + num_pixels_lost_by_laplacian + num_pixels_masked_NaN
    total = np.shape(segments)[0]*(trace_aperture["vcut2"]-trace_aperture["vcut1"])*(trace_aperture["hcut2"]-trace_aperture["hcut1"])
    print("In all, %.0f trace pixels out of %.0f were masked (fraction of %.2f) and %.0f frames will be skipped." % (totalmasked, total, totalmasked/total, len(frames_to_reject)))
    
    print("Stage 3 calibrations completed in %.3f minutes." % ((time.time()-t0)/60))
    
    print("Writing calibrated fits file as several .fits file...")
    for i, file in enumerate(files):
        outfile = os.path.join(outdir, "postprocessed_{0:g}.fits".format(i))
        shutil.copy(file,outfile)
        with fits.open(outfile, mode="update") as fits_file:
            # Need to write new objects "segstarts" and "frames_to_reject" to fits file,
            # and update data, int_times, and wavelength attributes to be concatenated arrays.

            # Create REJECT ImageHDU to contain frames_to_reject object.
            REJECT = fits.ImageHDU(np.array(frames_to_reject), name='REJECT')
            # Append REJECT to the hdulist of fits_file.
            fits_file.append(REJECT)

            # Create SEGSTARTS ImageHDU to contain segstarts object.
            SEGSTARTS = fits.ImageHDU(np.array(segstarts), name='SEGSTARTS')
            # Append SEGSTARTS to the hdulist of fits_file.
            fits_file.append(SEGSTARTS)

            # Write calibrated image data, times, and wavelengths to new fits file.
            if i == 0:
                fits_file['SCI'].data = segments[:segstarts[i],:,:]
            else:
                fits_file['SCI'].data = segments[segstarts[i-1]:segstarts[i],:,:]

            # All modified headers get written out.
            fits_file.writeto(outfile, overwrite=True)
    print("Wrote calibrated postprocessed_#.fits files.")
    print("Stage 3 calibrations completed in %.3f minutes." % ((time.time() - t0)/60))

def loss_stats(dqflags, trace_aperture, outdir):
    '''
    Gets pixel loss statistics for the region inside of the trace.
    
    :param dqflags: 3D array. Integrations x rows x cols of data quality flags.
    :param trace_aperture: dict. Keywords are "hcut1", "hcut2", "vcut1", "vcut2", all integers denoting the rows and columns respectively that define the edges of the aperture bounding the trace.
    :return: int of how many pixels were affected by DQ flags.
    '''
    # Create the aperture for counting loss statistics.
    aperture = np.ones(np.shape(dqflags))
    aperture[:,
             trace_aperture["hcut1"]:trace_aperture["hcut2"],
             trace_aperture["vcut1"]:trace_aperture["vcut2"]] = 0
    
    print("Checking to see how much of the trace data will be lost to dq flags...")
    loss_stats = []
    for k in range(np.shape(dqflags)[0]):
        dqflag_arr = dqflags[k,
                             trace_aperture["hcut1"]:trace_aperture["hcut2"],
                             trace_aperture["vcut1"]:trace_aperture["vcut2"]]
        loss_stats.append(np.count_nonzero(dqflag_arr))
    
    # Report what unique flags appear in this dataset.
    unique = []
    
    for row in np.ma.masked_array(dqflags[k, :, :], aperture[k, :, :]):
        for element in row:
            if element not in unique:
                unique.append(element)
    del(unique[0])
    print("The following JWST DQ flags were reported:")
    print(unique)
    
    # Creates an image of the aperture with flags.
    flagged_arr = np.where(np.ma.masked_array(dqflags[0, :, :], aperture[0, :, :]) > 0, 1, 0)
    fig, ax, im = img(np.ma.masked_array(flagged_arr, aperture[0, :, :]), aspect=5)
    plt.savefig(os.path.join(outdir, "output_imgs_calibration/trace_aperture_with_flags.pdf"), dpi=300)
    plt.close(fig)
    
    # Reports pixel loss stats.
    print("The total number of lost trace pixels is %.3f." % np.sum(loss_stats))
    print("The total percentage of pixels being lost is %.3f." % (100*(np.sum(loss_stats)/(np.shape(dqflags)[0]*(trace_aperture["vcut2"]-trace_aperture["vcut1"])*(trace_aperture["hcut2"]-trace_aperture["hcut1"])))))
    print("The median number of lost trace pixels is %.3f." % np.median(loss_stats))
    print("The median percentage of pixels being lost is %.3f." % (100*(np.median(loss_stats)/((trace_aperture["vcut2"]-trace_aperture["vcut1"])*(trace_aperture["hcut2"]-trace_aperture["hcut1"])))))
    print("Creating a histogram showing pixel loss statistics...")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(loss_stats, density=True, bins=70)
    ax.set_xlabel('number of flagged trace pixels')
    ax.set_ylabel('frequency')
    plt.show()
    plt.close()
    
    return np.sum(loss_stats)

def mask_flagged(segments, dqflags, trace_aperture):
    '''
    Mask all pixels flagged by the data quality array with their medians in time.
    
    :param segments: 3D array. Integrations x rows x cols of data.
    :param dqflags: 3D array. Integrations x rows x cols of data quality flags.
    :param trace_aperture: dict. Keywords are "hcut1", "hcut2", "vcut1", "vcut2", all integers denoting the rows and columns respectively that define the edges of the aperture bounding the trace.
    :return: segments array with flagged pixels masked, and int of how many pixels in the trace were affected by this process.
    '''
    # Get borders to evaluate.
    hcut1, hcut2, vcut1, vcut2 = trace_aperture["hcut1"], trace_aperture["hcut2"], trace_aperture["vcut1"], trace_aperture["vcut2"]
    
    # Turn dqflags into mask arrays.
    dq_mask = np.empty_like(dqflags)
    dq_mask[:, :, :] = np.where(dqflags[:, :, :] > 0, 1, 0)
    
    print("Masking flagged pixels pixels inside and outside of the trace...")
    t0 = time.time()
    masked_flagged = 0
    
    for i in range(np.shape(segments)[1]):
        for j in range(np.shape(segments)[2]):
            # Track temporal variations and replace any flagged pixels with the temporal median
            # that was calculated out of only unmasked values.
            pmed = np.ma.median(np.ma.masked_array(segments[:, i, j], dq_mask[:, i, j]))
            psigma = np.std(np.ma.masked_array(segments[:, i, j], dq_mask[:, i, j]))
            if (i in range(hcut1, hcut2) and j in range(vcut1, vcut2)):
                masked_flagged += np.count_nonzero(dqflags[:, i, j])
            segments[:, i, j] = np.where(dq_mask[:, i, j] == 1, pmed, segments[:, i, j])
    print("Masked %.0f flagged trace pixels in %.3f seconds." % (masked_flagged, time.time() - t0))
    return segments, masked_flagged

def iterate_outlier_removal(segments, dqflags, trace_aperture, n, sigma):
    '''
    Iterate and remove outliers to reject CRs. Does not use masked values to compute the time median.
    
    :param segments: 3D array. Integrations x rows x cols of data.
    :param dqflags: 3D array. Integrations x rows x cols of data quality flags.
    :param trace_aperture: dict. Keywords are "hcut1", "hcut2", "vcut1", "vcut2", all integers denoting the rows and columns respectively that define the edges of the aperture bounding the trace.
    :param n: int. Number of times to iterate.
    :param sigma: float. Sigma at which to reject outliers.
    :return: segments array with CRs rejected, and int of how many pixels in the trace were affected by this process.
    '''
    # Get borders to evaluate.
    hcut1, hcut2, vcut1, vcut2 = trace_aperture["hcut1"], trace_aperture["hcut2"], trace_aperture["vcut1"], trace_aperture["vcut2"]
    
    # Turn dqflags into mask arrays.
    dq_mask = np.empty_like(dqflags)
    dq_mask[:, :, :] = np.where(dqflags[:, :, :] > 0, 1, 0)
    
    print("Masking pixels with substantial time variations...")
    t0 = time.time()
    masked_iter = 0
    
    for iteration in range(n):
        print("On iteration %.0f..." % (iteration+1))
        t02 = time.time()
        for i in range(np.shape(segments)[1]):
            for j in range(np.shape(segments)[2]):
                # Track temporal variation of a single pixel and mask anywhere that pixel is 10sigma
                # deviating from its usual levels. This should help suppress noise.
                pmed = np.ma.median(np.ma.masked_array(segments[:, i, j], dq_mask[:, i, j]))
                psigma = np.std(np.ma.masked_array(segments[:, i, j], dq_mask[:, i, j]))
                if (i in range(hcut1, hcut2) and j in range(vcut1, vcut2)):
                    maskcount = np.where(np.abs(segments[:, i, j] - pmed)>sigma*psigma, 1, 0)
                    masked_iter += np.count_nonzero(maskcount)
                segments[:, i, j] = np.where(np.abs(segments[:, i, j] - pmed)>sigma*psigma, pmed, segments[:, i, j])
        print("Performed round %.0g of %.2f-sigma temporal outlier rejection in %.3f seconds." % (iteration+1, sigma, time.time()-t02))
    print("Masked %.0f trace pixels for significant temporal variations in %.3f seconds." % (masked_iter, time.time()-t0))
    return segments, masked_iter

def spatial_outlier_removal(segments, trace_aperture, sigma, kernel):
    '''
    Median filter the image to remove hot pixels.
    
    :param segments: 3D array. Integrations x rows x cols of data.
    :param trace_aperture: dict. Keywords are "hcut1", "hcut2", "vcut1", "vcut2", all integers denoting the rows and columns respectively that define the edges of the aperture bounding the trace.
    :param sigma: float. Sigma at which to reject outliers.
    :param kernel: tuple of odd ints. Kernel to use for spatial filtering.
    :return: segments(t,y,x) with spatial outliers cleaned, and count int of how many pixels were flagged as outliers.
    '''
    # Get borders to evaluate.
    hcut1, hcut2, vcut1, vcut2 = trace_aperture["hcut1"], trace_aperture["hcut2"], trace_aperture["vcut1"], trace_aperture["vcut2"]
    
    print("Performing hot pixel masking through spatial median filtering...")
    masked_filter = 0
    cleaned_segments = np.zeros_like(segments)
    
    t0 = time.time()
    for i in range(np.shape(segments)[0]):
        # Clean the array.
        cleaned_segments[i, :, :] = clean(segments[i, :, :], sigma, kernel)
        
        # Check where it has been changed.
        maskcount = np.empty((hcut2-hcut1, vcut2-vcut1))
        maskcount = np.where(cleaned_segments[i, hcut1:hcut2, vcut1:vcut2] != segments[i, hcut1:hcut2, vcut1:vcut2], 1, 0)
        masked_filter += np.count_nonzero(maskcount)
    print("Masked %.0f trace pixels for significant spatial variation in %.3f seconds." % (masked_filter, time.time()-t0))
    print("Performed median filtering in %.3f seconds." % (time.time()-t0))
    return cleaned_segments, masked_filter

def laplacian_outlier_removal(segments, errors, trace_aperture, sigma=50, verbose=False):
    '''
    Convolves a Laplacian kernel with the segments array to replace spatial outliers with
    the median of the surrounding 3x3 kernel.
    
    :param segments: 3D array. The segments(t,y,x) array from which outliers will be removed.
    :param errors: 3D array. The errors(t,y,x) array used to build the noise model.
    :param trace_aperture: dict. Defines where the aperture is. Used here for counting pixels lost to Laplacian filtering.
    :param sigma: float. Threshold of deviation from median of Laplacian image, above which a pixel will be flagged as an outlier and masked.
    :param verbose: bool. If True, occasionally prints out a progress report.
    :return: segments(t,x,y) array with spatial outliers masked, and count int of how many pixels were lost to filtering.
    '''
    # Get borders to evaluate.
    hcut1, hcut2, vcut1, vcut2 = trace_aperture["hcut1"], trace_aperture["hcut2"], trace_aperture["vcut1"], trace_aperture["vcut2"]
    
    l = 0.25*np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    segmentsc = deepcopy(segments)
    errorsc = deepcopy(errors)
    
    bad_pix_removed = 0
    t0 = time.time()
    steps_taken = 0
    nsteps = np.shape(segments)[0]
    
    print("Cleaning %.1f-sigma outliers with Laplacian edge detection..." % sigma)
    for k in range(np.shape(segments)[0]):
        # Iterate over frames.
        print("On frame %.0f..." % k)
        # Estimate readnoise.
        errf  = errorsc[k,:,:]**2
        errf -= segmentsc[k,:,:] # remove shot noise from errors to get read noise variance.
        errf[errf < 0] = 0 # enforce positivity.
        errf  = np.sqrt(errf) # turn variance into readnoise.
        rn    = np.mean(errf) # mean readnoise is our estimate.
        
        if (verbose and k == 0):
            print("Estimated readnoise: %.10f" % rn)
        
        # Build noise model.
        NOISE = np.sqrt(median_filter(np.abs(segmentsc[k,:,:]),size=5)+rn**2)
        NOISE[NOISE <= 0] = np.min(NOISE[np.nonzero(NOISE)]) # really want to avoid nans
        if (verbose and k == 0):
            fig, ax, im = img(np.log10(NOISE), aspect=5, title="Noise model for LED")
            plt.show()
            plt.close(fig)
        
        original_shape = np.shape(segments[k,:,:])
        ss_shape = (original_shape[0]*2,original_shape[1]*2) # double subsampling
        subsample = np.empty(ss_shape)
        
        # Subsample the array.
        for i in range(ss_shape[0]):
            for j in range(ss_shape[1]):
                try:
                    subsample[i,j] = segments[int((i+1)/2),int((j+1)/2),k]
                except IndexError:
                    subsample[i,j] = 0
        
        # Convolve subsample with laplacian.
        lap_img = np.convolve(l.flatten(),subsample.flatten(),mode='same').reshape(ss_shape)
        lap_img[lap_img < 0] = 0 # force positivity
        
        # Resample to original size.
        resample = np.empty(original_shape)
        for i in range(original_shape[0]):
            for j in range(original_shape[1]):
                resample[i,j] = 0.25*(lap_img[2*i-1,2*j-1] +
                                      lap_img[2*i-1,2*j] +
                                      lap_img[2*i,2*j-1] +
                                      lap_img[2*i,2*j])
                
        # Divide by subsample factor times noise model.
        scaled_resample = resample/(2*NOISE)
        
        # Spot outliers.
        med = np.median(scaled_resample)
        if (verbose and k == 0):
            print("Median of scaled resampled laplacian image: %.10f" % med)
        scaled_resample[np.abs(scaled_resample-med) < med*sigma] = 0 # any not zero after this are rays.
        scaled_resample[scaled_resample!=0] = 1 # for visualization
        
        if (verbose and k == 0):
            fig, ax, im = img(scaled_resample, aspect=5, title="Where CRs and hot pixels were detected")
            plt.show()
            plt.close()
        
        # Correct frames
        for i, j in zip(np.where(scaled_resample!=0)[0],np.where(scaled_resample!=0)[1]):
            segments[k,i,j] = np.median(segments[k,i-1:i+2,j-1:j+2]) # replace with local median
            if (hcut1 <= i <= hcut2 and vcut1 <= j <= vcut2):
                bad_pix_removed += 1
            
        # Report progress.
        steps_taken += 1
        if (steps_taken % int(nsteps*.1) == 0 and verbose):
            iter_rate = steps_taken/(time.time()-t0)
            print("%.0f-percent done. Time elapsed: %.0f seconds. Estimated time remaining: %.0f seconds." % (steps_taken*100/nsteps, time.time()-t0, (nsteps-steps_taken)/iter_rate))
    print("Iterations complete. Removed %.0f spatial outliers in %.0f seconds." % (bad_pix_removed, time.time()-t0))
    return segments, bad_pix_removed

def bckg_subtract(segments, bckg_rows, sigma):
    '''
    Subtract background signal using the rows defined by bckg_rows as the background.
    
    :param segments: 3D array. Integrations x rows x cols of data.
    :param bckg_rows: list of integers. Indices of the rows to use as the background region.
    :param sigma: float. Sigma at which to clean the background region of outliers. Lower sigma means more aggressive cleaning.
    :return: segments with background subtracted.
    '''
    print("Performing additional background subtraction...")
    t0 = time.time()
    
    for i in range(np.shape(segments)[0]):
        background_region = segments[i, bckg_rows, :]
        background_region = sigma_clip(background_region, sigma=sigma)
        mmed = np.ma.median(background_region)
        background_region = background_region.filled(fill_value=mmed)
        background = background_region.mean(axis=0)
        background = np.array([background,]*np.shape(segments)[1])
        segments[i, :, :] = segments[i, :, :] - background
        if (i%1000 == 0 and i != 0):
            # Report progress every 1,000 integrations.
            elapsed_time = time.time()-t0
            iterrate = i/elapsed_time
            iterremain = np.shape(segments)[0] - i
            print("On integration %.0f. Elapsed time is %.3f seconds." % (i, elapsed_time))
            print("Average rate of integration processing: %.3f ints/s." % iterrate)
            print("Estimated time remaining: %.3f seconds.\n" % (iterremain/iterrate))
    print("Additional background subtraction completed in %.3f seconds." % (time.time()-t0))
    return segments

def gaussian_source_track(segments, reject_dispersion_direction=True, reject_spatial_direction=True):
    '''
    Tracks the location of the trace between frames and reports frame numbers that show
    significant deviations from the usual location.
    
    :param segments: 3D array. Integrations x rows x cols of data.
    :param reject_dispersion_direction: bool. Whether to reject outliers of position in the dispersion direction.
    :param reject_spatial_direction: bool. Whether to reject outliers of position in the spatial direction.
    :return: reject_frames list of ints showing which frames are to get rejected by tracking of source position.
    '''
    reject_frames = []
    t0 = time.time()
    source_pos_disp = []
    source_pos_cros = []
    print("Fitting source position for each integration...")
    for k in range(np.shape(segments)[0]):
        # First find the dispersion axis position.
        profile = np.sum(segments[k, :, :], axis=0)
        profile = profile/np.max(profile) # normalize amplitude to 1 for ease of fit
        fitter = modeling.fitting.LevMarLSQFitter()
        model = modeling.models.Gaussian1D(amplitude=1, mean=100, stddev=1)
        fitted_model = fitter(model, [i for i in range(np.shape(profile)[0])], profile)
        source_pos_disp.append(fitted_model.mean[0])

        # Then find the cross dispersion axis position.
        profile = np.sum(segments[k, :, :], axis=1)
        profile = profile/np.max(profile) # normalize amplitude to 1 for ease of fit
        fitter = modeling.fitting.LevMarLSQFitter()
        model = modeling.models.Gaussian1D(amplitude=1, mean=14, stddev=1)
        fitted_model = fitter(model, [i for i in range(np.shape(profile)[0])], profile)
        source_pos_cros.append(fitted_model.mean[0])

        if (k%500 == 0 and k != 0):
            # Report progress every 500 integrations.
            elapsed_time = time.time()-t0
            iterrate = k/elapsed_time
            iterremain = np.shape(segments)[0] - k
            print("On integration %.0f. Elapsed time is %.3f seconds." % (k, elapsed_time))
            print("Average rate of integration processing: %.3f ints/s." % iterrate)
            print("Estimated time remaining: %.3f seconds.\n" % (iterremain/iterrate))
    print("Fit source positions in %.3f minutes." % ((time.time()-t0)/60))
    
    # Now that we have the positions, build list of integration indices to reject for being too far off.
    mpd = np.median(source_pos_disp)
    spd = np.std(source_pos_disp)
    mpc = np.median(source_pos_cros)
    spc = np.std(source_pos_cros)

    print("Median position: " + str(mpd) + ", " + str(mpc))
    print("Sigma: " + str(spd) + ", " + str(spc))
    skipped = 0
    for k in range(np.shape(segments)[0]):
        if reject_dispersion_direction:
            if np.abs(mpd - source_pos_disp[k]) > 3*spd:
                reject_frames.append(k)
                skipped += 1
        
        if reject_spatial_direction:
            if np.abs(mpc - source_pos_cros[k]) > 3*spc:
                reject_frames.append(k)
                skipped += 1
    print("%.0f integrations had source positions significantly off from the median position and will be rejected." % skipped)
    return reject_frames