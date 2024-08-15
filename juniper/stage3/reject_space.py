import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.plotting import img

def smooth(segments, inpt_dict):
    """Uses median filtering to smooth outliers.

    Args:
        segments (xarray): its segments.data is the data to remove outliers from.
        inpt_dict (dict): instructions for running this step.

    Returns:
        xarray: segments.data with spatial outliers removed by median filtering.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Cleaning threshold=%.1f outliers with spatial filtering..." % inpt_dict["space_sigma"])

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Track outliers removed and where they were found.
    bad_pix_map = np.zeros_like(segments.data.values)
    bad_pix_removed = 0
    bad_pix_per_frame = []

    # Iterate over each frame and smooth.
    for k in tqdm(range(segments.data.shape[0]),
                  desc = 'Smoothing outliers from integrations...',
                  disable=(not time_ints)):
        # Build a smoothed model.
        smooth = median_filter(np.copy(segments.data.values[k,:,:]),
                               size=inpt_dict["space_kernel"])
        
        # Check for outliers and locate them.
        S = np.where(np.abs(segments.data.values[k,:,:]-smooth)>inpt_dict["space_sigma"],1,0)
        bad_pix_map[k,:,:] += S
        bad_pix_this_frame = np.count_nonzero(S)
        bad_pix_removed += bad_pix_this_frame

        # Count outliers.
        bad_pix_per_frame.append(bad_pix_this_frame)

        # And replace if asked.
        if inpt_dict["space_replace"]:
            segments.data.values[k,:,:] = np.where(S == 1, smooth, segments.data.values[k,:,:])
    
    # Update data flags.
    segments.dq.values = np.where(bad_pix_map != 0, 1, segments.dq.values)

    if inpt_dict["verbose"] == 2:
        print("Median outliers found in each frame: {} +/- {}".format(int(np.median(bad_pix_per_frame)),
                                                                      int(np.std((bad_pix_per_frame)))))

    if (plot_step or save_step):
        # Create plots of the entire bad_pix_map collapsed in on itself in time.
        bad_pix_alltime = np.sum(bad_pix_map,axis=0)
        bad_pix_alltime[bad_pix_alltime>0] = 1
        fig, ax, im = img(bad_pix_alltime, aspect=5, title='Spatial smoothing DQ flags',
                          vmin=0, vmax=1, norm='linear', verbose=inpt_dict["verbose"])
        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_spatial-smoothing_flags.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()
    
    if (plot_ints or save_ints):
        # Create plots of each frame of the bad_pix_map in time.
        for i in range(bad_pix_map.shape[0]):
            fig, ax, im = img(bad_pix_map[i,:,:], aspect=5, title='Spatial smoothing DQ flags',
                              vmin=0, vmax=1, norm='linear', verbose=inpt_dict["verbose"])
            if save_step:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_spatial-smoothing_flags_int{}.png".format(i)),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show()
            plt.close()

    # Count outliers and log.
    if inpt_dict["verbose"] >= 1:
        print("Smoothing complete. Outliers found in total: {}".format(bad_pix_removed))

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
        
    return segments


def led(segments, inpt_dict):
    """Convolves a Laplacian kernel with the obs.images to replace spatial outliers with
    the median of the surrounding 3x3 kernel.

    Args:
        segments (xarray): its segments.data is the data to remove outliers from.
        inpt_dict (dict): instructions for running this step.

    Returns:
        xarray: segments.data with spatial outliers removed by Laplacian Edge Detection.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Cleaning threshold=%.1f outliers with Laplacian edge detection..." % inpt_dict["led_sigma"])

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Track outliers removed and where they were found.
    bad_pix_map = np.zeros_like(segments.data.values)

    # Define the Laplacian kernel.
    l = 0.25*np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

    # Iterate over each frame one at a time until the iteration stop condition is met by each frame.
    for k in tqdm(range(segments.data.shape[0]),
                  desc = 'Running LED on integrations...',
                  disable=(not time_ints)):
        # Get the frame and errors as np.array objects so we can operate on them.
        integration = segments.data[k].values
        errs = segments.err[k].values

        # Track outliers flagged in this frame and iterations performed.
        bad_pix_removed = 0
        iteration_N = 1

        # Then start iterating over this frame and keep going until the iteration stop condition is met.
        stop_iterating = False
        while not stop_iterating:
            # Estimate readnoise value.
            var2 = errs**2 - integration
            var2[var2 < 0] = 0 # enforce positivity.
            rn = np.sqrt(var2) # estimate readnoise array

            # Build the noise model.
            noise_model = build_noise_model(integration, rn)
            if inpt_dict["fine_structure"]:
                F = build_fine_structure_model(integration)

            # Subsample the array.
            subsample, original_shape = subsample_frame(integration,
                                                        factor=inpt_dict["led_factor"])
            
            # Convolve subsample with laplacian.
            lap_img = np.convolve(l.flatten(),subsample.flatten(),mode='same').reshape(subsample.shape)
            lap_img[lap_img < 0] = 0 # force positivity
            
            # Resample laplacian-convolved subsampled image to original size.
            resample = resample_frame(lap_img, original_shape)

            # Divide by the noise model scaled by the resampling factor.
            S = resample/(inpt_dict["led_factor"]*noise_model)
            
            # Remove sampling flux to protect data from being targeted by LED.
            S = S - median_filter(S, size=5)

            # Spot outliers.
            S[np.abs(S) < inpt_dict["led_sigma"]] = 0 # any not zero after this are rays.
            S[S!=0] = 1 # for visualization and comparison to fine structure model.

            # If we have a fine structure model, we also need to check the contrast.
            if inpt_dict["fine_structure"]:
                contrast_image = resample/F
                contrast_image[contrast_image < inpt_dict["contrast_factor"]] = 0 # any not zero after this are rays.
                contrast_image[contrast_image!=0] = 1 # for visualization and comparison to sampling flux model.
                
                # Then we need to merge the results of S = Laplacian_image/factor*noise_model - sampling_flux
                # and contrast_image = Laplacian_image/Fine_structure_model so that we only take where both are 1.
                S = np.where(S == contrast_image, 1, 0)

            # Locate new bad pixels.
            bad_pix_last_frame = -100
            if iteration_N != 1:
                bad_pix_last_frame = bad_pix_this_frame # if this is not the first time we've done this, we need to store the last frame's bad pix count before updating it.

            S = np.where(bad_pix_map[k,:,:] == 1, 0, S) # if the pixel was already reported as bad in a previous iteration, don't double count it.
            bad_pix_map[k,:,:] += S
            bad_pix_this_frame = np.count_nonzero(S)
            bad_pix_removed += bad_pix_this_frame

            # Report progress.
            if inpt_dict["verbose"] == 2:
                print("Bad pixels caught on iteration %.0f: %.0f" % (iteration_N, bad_pix_this_frame))

            # Correct frames, if asked.
            if inpt_dict["led_replace"]:
                med_filter_image = median_filter(integration,size=5)
                integration = np.where(S != 0, med_filter_image, integration)

            # Make some plots if asked.
            if (plot_step or save_step) and k == 0:
                # Plot the noise model and fine structure model of the first integration.
                fig, ax, im = img(noise_model, aspect=5, title="LED Noise Model",
                                  norm='linear', verbose=inpt_dict["verbose"])
                if save_step:
                    plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_spatial-LED_noise-model_N{}_int{}.png".format(iteration_N, k)),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show()
                plt.close()

                if inpt_dict["fine_structure"]:
                    fig, ax, im = img(contrast_image, aspect=5, title="LED Fine Structure Model",
                                      norm='linear', verbose=inpt_dict["verbose"])
                    if save_step:
                        plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_spatial-LED_fine-structure-model_N{}_int{}.png".format(iteration_N, k)),
                                    dpi=300, bbox_inches='tight')
                    if plot_step:
                        plt.show()
                    plt.close()

            # Increment iteration number and check if condition to stop iterating is hit.
            iteration_N += 1
            if (inpt_dict["n"] != None and iteration_N > inpt_dict["n"]): # if it has hit the iteration cap
                stop_iterating = True
            if (inpt_dict["n"] == None and bad_pix_this_frame == bad_pix_last_frame): # if it has stalled out on finding new outliers
                stop_iterating = True
        
        # Report that the iterations for this frame have stopped.
        if inpt_dict["verbose"] == 2:
            print("Finished cleaning frame %.0f in %.0f iterations." % (k, iteration_N-1))
            print("Total pixels corrected: %.0f out of %.0f" % (bad_pix_removed, S.shape[0]*S.shape[1]))
        # And replace the xarray datasets if asked.
        if inpt_dict["space_replace"]:
            segments.data.values[k] = np.where(segments.data[k].values != integration,integration,segments.data.values[k])

    # Update data flags.
    segments.dq.values = np.where(bad_pix_map != 0, 1, segments.dq.values)

    if (plot_step or save_step):
        # Create plots of the entire bad_pix_map collapsed in on itself in time.
        bad_pix_alltime = np.sum(bad_pix_map,axis=0)
        bad_pix_alltime[bad_pix_alltime>0] = 1
        fig, ax, im = img(bad_pix_alltime, aspect=5, title='Laplacian Edge Detection DQ flags',
                          vmin=0, vmax=1, norm='linear', verbose=inpt_dict["verbose"])
        if save_step:
            plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_spatial-LED_flags.png"),
                        dpi=300, bbox_inches='tight')
        if plot_step:
            plt.show()
        plt.close()
    
    if (plot_ints or save_ints):
        # Create plots of each frame of the bad_pix_map in time.
        for i in range(bad_pix_map.shape[0]):
            fig, ax, im = img(bad_pix_map[i,:,:], aspect=5, title='Laplacian Edge Detection DQ flags',
                              vmin=0, vmax=1, norm='linear', verbose=inpt_dict["verbose"])
            if save_step:
                plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],"S3_spatial-LED_flags_int{}.png".format(i)),
                            dpi=300, bbox_inches='tight')
            if plot_step:
                plt.show()
            plt.close()

    # Log.
    if inpt_dict["verbose"] >= 1:
        print("All integrations cleaned of spatial outliers by LED.")

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)
    
    return segments

def build_noise_model(data_frame, readnoise):
    """Builds a noise model for the given data frame, following van Dokkum 2001 methods.

    Args:
        data_frame (np.array): Integration from the segments.data DataSet, used to build the noise model.
        readnoise (float): Readnoise estimated to be in the data frame.

    Returns:
        np.array: 2D array same size as the data frame, a noise model describing noise in the frame.
    """
    noise_model = np.sqrt(median_filter(np.abs(data_frame),size=5)+readnoise**2)
    noise_model[noise_model <= 0] = np.mean(noise_model) # really want to avoid nans
    return noise_model

def subsample_frame(data_frame, factor=2):
    """Subsamples the input frame by the given subsampling factor.

    Args:
        data_frame (np.array): Integration from the segments.data DataSet.
        factor (int, optional): int >= 2. Factor by which to subsample the array. Defaults to 2.

    Returns:
        np.array: sub-sampled data frame.
    """
    factor = int(factor) # Force integer
    if factor < 2:
        print("Subsampling factor must be at least 2, forcing factor to 2...")
        factor = 2 # Force factor 2 or more
    
    original_shape = np.shape(data_frame)
    ss_shape = (original_shape[0]*factor,original_shape[1]*factor)
    subsample = np.empty(ss_shape)
    
    # Subsample the array.
    for i in range(ss_shape[0]):
        for j in range(ss_shape[1]):
            try:
                subsample[i,j] = data_frame[int((i+1)/2),int((j+1)/2)]
            except IndexError:
                subsample[i,j] = 0
    return subsample, original_shape

def resample_frame(data_frame, original_shape):
    """Resamples a subsampled array back to the original shape.

    Args:
        data_frame (np.array): Subsampled integration from the segments.data DataSet.
        original_shape (tuple of int): Original shape of the subsampled array.

    Returns:
        np.array: 2D array with original shape resampled from the data frame.
    """
    resample = np.empty(original_shape)
    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            resample[i,j] = 0.25*(data_frame[2*i-1,2*j-1] +
                                  data_frame[2*i-1,2*j] +
                                  data_frame[2*i,2*j-1] +
                                  data_frame[2*i,2*j])
    return resample

def build_fine_structure_model(data_frame):
    """Builds a fine structure model for the data frame.

    Args:
        data_frame (np.array): Native resolution data.

    Returns:
        np.array: fine structure model built from the data.
    """
    F = median_filter(data_frame, size=3) - median_filter(median_filter(data_frame, size=3), size=7)
    F[F <= 0] = np.mean(F) # really want to avoid nans
    return F