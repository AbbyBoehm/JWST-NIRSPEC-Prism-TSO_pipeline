import time
from tqdm import tqdm
import numpy as np

from astropy import modeling

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer

def track_pos(segments, inpt_dict):
    """Tracks position of trace in each integration.

    Args:
        segments (xarray): its segments DataSet is the integrations which will be tracked.
        inpt_dict (dict): instructions for running this step.

    Returns:
        xarray, list, list, list: the segments array with updated data quality flags, and the disp. positions, cross-disp. positions, and identified indices of bad frames.
    """

    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Tracking motion of the trace...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Start tracking.
    bad_k = []
    bad_frame_map = np.zeros_like(segments.data)
    if inpt_dict["track_disp"]:
        dispersion_position = []
        for k in tqdm(range(segments.data.shape[0]),
                    desc='Fitting trace dispersion position in each integration...',
                    disable=(not time_ints)):
            profile = np.nansum(segments.data.values[k,:,:], axis=0)
            dispersion_position.append(fit_profile(profile,guess_pos=profile.shape[0]*0.20))
        
        if inpt_dict["reject_disp"]:
            # Flag any integration with sudden movement.
            med_disp, std_disp = np.median(dispersion_position), np.std(dispersion_position)

            for k in tqdm(range(segments.data.shape[0]),
                          desc='Identifying trace dispersion position outliers...',
                          disable=(not time_ints)):
                if np.abs(med_disp - dispersion_position[k]) > 3*std_disp:
                    # The frame moved by 3 sigma, kick it.
                    bad_k.append(k)
                    bad_frame_map[k,:,:] = np.ones_like(bad_frame_map[k,:,:]) # the whole frame is flagged for data quality

    if inpt_dict["track_spatial"]:
        crossdispersion_position = []
        for k in tqdm(range(segments.data.shape[0]),
                    desc='Fitting trace cross-dispersion position in each integration...',
                    disable=(not time_ints)):
            profile = np.nansum(segments.data.values[k,:,:], axis=1)
            crossdispersion_position.append(fit_profile(profile,guess_pos=profile.shape[0]*0.50))

        if inpt_dict["reject_spatial"]:
            # Flag any integration with sudden movement.
            med_cross, std_cross = np.median(crossdispersion_position), np.std(crossdispersion_position)

            for k in tqdm(range(segments.data.shape[0]),
                          desc='Identifying trace cross-dispersion position outliers...',
                          disable=(not time_ints)):
                if np.abs(med_cross - crossdispersion_position[k]) > 3*std_cross:
                    # The frame moved by 3 sigma, kick it if it isn't already kicked.
                    if k not in bad_k:
                        bad_k.append(k)
                        bad_frame_map[k,:,:] = np.ones_like(bad_frame_map[k,:,:]) # the whole frame is flagged for data quality

    # Update data flags.
    segments.dq.values = np.where(bad_frame_map != 0, 1, segments.dq.values)

    # Report outliers found.
    if inpt_dict["verbose"] >= 1:
        print("Frame tracking complete.")
        if bad_k:
            print("Total frames rejected for sudden motion: {}".format(len(bad_k)))

    # Report time, if asked.
    if time_step:
        timer(time.time()-t0,None,None,None)

    return segments, dispersion_position, crossdispersion_position, bad_k

def fit_profile(profile,guess_pos):
    """Simple utility to fit a Gaussian profile to the trace position.

    Args:
        profile (np.array): a dispersion or cross-dispersion profile whose position is to be tracked.
        guess_pos (float): initial guess for the position of the source.

    Returns:
        float: the position of the profile.
    """
    profile = profile/np.max(profile) # normalize amplitude to 1 for ease of fit
    fitter = modeling.fitting.LevMarLSQFitter()
    model = modeling.models.Gaussian1D(amplitude=1, mean=guess_pos, stddev=1)
    fitted_model = fitter(model, [i for i in range(profile.shape[0])], profile)
    return fitted_model.mean[0]