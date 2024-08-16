import time
from tqdm import tqdm

import numpy as np
from scipy.optimize import minimize

from juniper.stage5 import batman_handler, fit_handler, exotic_handler
from juniper.util.diagnostics import tqdm_translate, plot_translate, timer
from juniper.util.cleaning import median_timeseries_filter

def lsqfit_one(lc_time, light_curve, errors, waves, planets, flares, systematics, LD, inpt_dict, is_spec=False):
    """Performs linear least squares fitting on the given array(s) using scipy.
    Fits a single light curve. Useful for fitting spectroscopic curves.

    Args:
        lc_time (np.array): mid-exposure times for each point in the light curve.
        light_curve (np.array): median-normalized flux with time.
        errors (np.array): uncertainties associated with each data point, used
        in weighting the residuals.
        waves (np.array): used to supply the wavelength range to
        ExoTiC-LD if needed.
        planets (dict): uninitialized planet dictionaries which need to be
        initialized with the batman_handler.
        flares (dict): a series of dictionary entries describing each flaring
        event suspected to have occurred during the observation.
        systematics (dict): a series of dictionary entries describing each
        systematic model to detrend for.
        LD (dict): a dictionary describing the limb darkening model, including
        the star's physical characteristics.
        inpt_dict (dict): instructions for running this step.
        is_spec (bool, optional): whether this is a fit to a spectroscopic
        curve, in which case certain system parameters are to be locked.
        Defaults to False.
    
    Returns:
        dict, dict, dict, dict: planets, flares, systematics, and LD updated
        with fitted values.
    """
    # Copy planets, flares, systematics, and stellar limb darkening in their unmodified state.
    old_planets = planets.copy()
    old_flares = flares.copy()
    old_systematics = systematics.copy()
    old_LD = LD.copy()

    # Check if position detrending is available.
    xpos, ypos, widths = [], [], []
    if systematics["pos_detrend"]:
        xpos = systematics["xpos"]
        ypos = systematics["ypos"]

        # Smooth the positions in case the locators had trouble.
        xpos = median_timeseries_filter(xpos,sigma=3.0,kernel=31)
        ypos = median_timeseries_filter(ypos,sigma=3.0,kernel=31)
        
    if systematics["width_detrend"]:
        widths = systematics["width"]
        # Smooth the widths in case the fitter had trouble.
        widths = median_timeseries_filter(widths,sigma=3.0,kernel=31)

    # If you are doing a poly fit, set the first polynomial coefficient better.
    if systematics["poly"]:
        systematics["poly_coeffs"][0] = np.median(light_curve)

    # Check if ExoTiC-LD is being used.
    if LD["use_exotic"]:
        # We need to update our parameters then.
        LD["wavelength_range"] = np.array([np.min(waves), np.max(waves)])
        LD["LD_initialguess"] = exotic_handler.get_exotic_coefficients(LD)
    
    # (Re-)Initialize the planets, giving them the LD info they need to talk to batman properly.
    planets = batman_handler.batman_init_all_planets(lc_time, planets, LD,
                                                     event=inpt_dict["event_type"])
    
    # Build a priors dictionary.
    params_priors = fit_handler.build_priors_dict(planets,flares,systematics,LD,
                                                  is_spec=is_spec)

    # Then build the lsq bounds object.
    bounds = fit_handler.build_bounds(params_priors, priors_type=inpt_dict["priors_type"])

    # Conveniently, the priors also tells us which keys are getting fit.
    fit_param_keys = list(params_priors.keys())

    # Translate planets, flares, systematics, and LDs into a single fitting dictionary.
    params_to_fit = fit_handler.bundle_planets_flares_systematics_and_LD(planets,
                                                                         flares,
                                                                         systematics,
                                                                         LD)
    
    # Turn that into an array so scipy will accept it.
    params_array = fit_handler.dict_to_array(params_to_fit, fit_param_keys)

    # Now do lsq.
    results = minimize(fit_handler._residuals,
                       x0=params_array,
                       args=(lc_time, light_curve, errors, params_to_fit, fit_param_keys, xpos, ypos, widths),
                       method=inpt_dict["LSQ_type"],
                       tol=inpt_dict["LSQ_tolerance"],
                       bounds=bounds,
                       options={"maxiter":inpt_dict["LSQ_iter"]})
    
    if inpt_dict["verbose"] == 2:
        print(results.message)
    
    # The array is here.
    fitted_array = results.x

    # Turn it back into a dict.
    fitted_dict = fit_handler.array_to_dict(fitted_array, params_to_fit, fit_param_keys)

    # And then turn those back into planets, flares, and systematics.
    repack_xpos, repack_ypos, repack_widths = [],[],[]
    if "xpos" in systematics.keys():
        repack_xpos = systematics["xpos"]
    if "ypos" in systematics.keys():
        repack_ypos = systematics["ypos"]
    if "width" in systematics.keys():
        repack_widths = systematics["width"]
    planets, flares, systematics, LD = fit_handler.unpack_params_back_to_dicts(fitted_dict,
                                                                               repack_xpos,
                                                                               repack_ypos,
                                                                               repack_widths)
    
    # Fill in anything that went missing.
    planets, flares, systematics, LD = fit_handler.refill(planets,flares,systematics,LD,
                                                          old_planets,old_flares,old_systematics,old_LD)
    
    # Re-initialize the planets.
    planets = batman_handler.batman_init_all_planets(lc_time, planets, LD,
                                                     event=inpt_dict["event_type"])
    
    # And return the fitted parameters.
    return planets, flares, systematics, LD

def lsqfit_joint(lc_time, light_curve, errors, waves, planets, flares, systematics, LD, inpt_dict):
    """Performs linear least squares fitting on the given array(s) using scipy.
    Fits multiple light curves simultaneously, forcing them to share system
    parameters (LD, systematics, and depth can be free in each detector).

    Args:
        lc_time (np.array): mid-exposure times for each point in each light curve.
        light_curve (np.array): median-normalized flux with time.
        errors (np.array): uncertainties associated with each data point, used
        in weighting the residuals.
        waves (np.array): used to supply the wavelength range to
        ExoTiC-LD if needed.
        planets (dict): uninitialized planet dictionaries which need to be
        initialized with the batman_handler.
        flares (dict): a series of dictionary entries describing each flaring
        event suspected to have occurred during the observation.
        systematics (dict): a series of dictionary entries describing each
        systematic model to detrend for.
        LD (dict): a dictionary describing the limb darkening model, including
        the star's physical characteristics.
        inpt_dict (dict): instructions for running this step.
    
    Returns:
        dict, dict, dict, dict: planets, flares, systematics, and LD updated
        with fitted values.
    """
    # Copy planets, flares, systematics, and stellar limb darkening in their unmodified state.
    old_planets = planets.copy()
    old_flares = flares.copy()
    old_systematics = systematics.copy()
    old_LD = LD.copy()

    # Check if position detrending is available.
    xpos, ypos, widths = [], [], []
    if systematics["pos_detrend"]:
        xpos = systematics["xpos"]
        ypos = systematics["ypos"]
    if systematics["width_detrend"]:
        widths = systematics["width"]

    # Check if ExoTiC-LD is being used.
    if LD["use_exotic"]:
        # We need to update our parameters then.
        LD["wavelength_range"] = np.array([np.min(waves), np.max(waves)])
        LD["LD_initialguess"] = exotic_handler.get_exotic_coefficients(LD)

    # Make dictionaries for each detector, forcing all but rp, LD, and
    # systematics to be shared.
    detectors = {}
    for l in range(light_curve.shape[0]):
        key = "detector" + str(l+1)
        detector = {}
        detector["planets"] = planets
        detector["flares"] = flares
        detector["systematics"] = systematics
        detector["LD"] = LD
        detectors[key] = detector
    
    # (Re-)Initialize the planets in each detector, giving them the LD info they
    # need to talk to batman properly.
    for detector in detectors.keys():
        D = detectors[detector]
        D["planets"] = batman_handler.batman_init_all_planets(lc_time, D["planets"], D["LD"],
                                                              event=inpt_dict["event_type"])
    
        # Build a priors dictionary.
        D["priors"] = fit_handler.build_priors_dict(D["planets"],D["flares"],D["systematics"],D["LD"])

        # Then build the lsq bounds object.
        D["bounds"] = fit_handler.build_bounds(D["priors"], priors_type=inpt_dict["priors_type"])

        # Conveniently, the priors also tells us which keys are getting fit.
        D["fit_param_keys"] = list(D["priors"].keys())

        # Translate planets, flares, systematics, and LDs into a single fitting dictionary.
        D["params_to_fit"] = fit_handler.bundle_planets_flares_systematics_and_LD(D["planets"],
                                                                                  D["flares"],
                                                                                  D["systematics"],
                                                                                  D["LD"])
    
        # Turn that into an array so scipy will accept it.
        D["params_array"] = fit_handler.dict_to_array(D["params_to_fit"], D["fit_param_keys"])

    # Now that we have N_detectors worth of params_arrays, we must consolidate them.
    params_array = fit_handler.consolidate_multiple_detectors(detectors)

    # Now do lsq.
    '''
    results = minimize(fit_handler._residuals,
                       x0=params_array,
                       args=(lc_time, light_curve, errors, params_to_fit, fit_param_keys, xpos, ypos, widths),
                       method=inpt_dict["LSQ_type"],
                       bounds=bounds)
    
    # The array is here.
    fitted_array = results.x

    # Turn it back into a dict.
    fitted_dict = fit_handler.array_to_dict(fitted_array, params_to_fit, fit_param_keys)

    # And then turn those back into planets, flares, and systematics.
    planets, flares, systematics, LD = fit_handler.unpack_params_back_to_dicts(fitted_dict,
                                                                               xpos,
                                                                               ypos,
                                                                               widths)
    
    # Fill in anything that went missing.
    planets, flares, systematics, LD = fit_handler.refill(planets,flares,systematics,LD,
                                                          old_planets,old_flares,old_systematics,old_LD)
    
    # Re-initialize the planets.
    planets = batman_handler.batman_init_all_planets(lc_time, planets, LD,
                                                     event=inpt_dict["event_type"])
    '''
    # WIP!
    # And return the fitted parameters.
    return planets, flares, systematics, LD