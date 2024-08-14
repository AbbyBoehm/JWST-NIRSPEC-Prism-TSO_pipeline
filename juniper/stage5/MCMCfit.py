import time
from tqdm import tqdm

import numpy as np
import emcee

from juniper.stage5 import batman_handler, fit_handler, exotic_handler
from juniper.util.diagnostics import tqdm_translate, plot_translate, timer

def mcmcfit_one(time, light_curve, errors, waves, planets, flares, systematics, LD, inpt_dict):
    """Performs Markov Chain Monte Carlo fitting on the given array(s) using emcee.
    Fits a single light curve. Useful for fitting spectroscopic curves.

    Args:
        time (np.array): mid-exposure times for each point in the light curve.
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
        widths = systematics["widths"]

    # Check if ExoTiC-LD is being used.
    if LD["use_exotic"]:
        # We need to update our parameters then.
        LD["wavelength_range"] = np.array([np.min(waves), np.max(waves)])
        LD["LD_initialguess"] = exotic_handler.get_exotic_coefficients(LD)
    
    # (Re-)Initialize the planets, giving them the LD info they need to talk to batman properly.
    planets = batman_handler.batman_init_all_planets(time, planets, LD,
                                                     event=inpt_dict["event_type"])
    
    # Build a priors dictionary.
    params_priors = fit_handler.build_priors_dict(planets,flares,systematics,LD)

    # Conveniently, the priors also tells us which keys are getting fit.
    fit_param_keys = list(params_priors.keys())

    # Translate planets, flares, systematics, and LDs into a single fitting dictionary.
    params_to_fit = fit_handler.bundle_planets_flares_systematics_and_LD(planets,
                                                                         flares,
                                                                         systematics,
                                                                         LD)
    
    # Turn that into an array so emcee will accept it.
    params_array = fit_handler.dict_to_array(params_to_fit, fit_param_keys)
    
    # Set up the chains with a little bit of scatter around the initial guess.
    pos = params_array + 1e-4 * np.random.randn(inpt_dict["MCMC_chains"],
                                                params_array.shape[0])
    nwalkers, ndim = pos.shape

    # Define how many steps we want to keep.
    if inpt_dict["MCMC_burnin"] >= 1:
        # If it is a whole integer, the user has asked to burn this many steps.
        discard = int(inpt_dict["MCMC_burnin"])
    else:
        # If it is a fraction, the user has asked to burn a fraction of the steps.
        discard = int(inpt_dict["MCMC_burnin"]*inpt_dict["MCMC_steps"])
    
    # Define the emcee sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, fit_handler.log_probability,
                                    args=(params_to_fit, fit_param_keys, time, light_curve, errors,
                                          params_priors, inpt_dict["priors_type"], xpos, ypos, widths),)
    
    # And run it!
    sampler.run_mcmc(pos, inpt_dict["MCMC_steps"], progress=True)#;
    
    # Pull the sampled posteriors and discard the burn-in and flatten it.
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=discard, flat=True)

    # Get how many parameters were fit and what each one is called.
    n = np.shape(samples[:,:,0])[0]*np.shape(samples[:,:,0])[1]
    labels = [key for key in fit_param_keys]

    # Turn the flattened chains into arrays.
    fitted_array, fitted_errs_array = fit_handler.get_result_from_post(ndim, flat_samples)
    
    # Turn the arrays back into dicts.
    fitted_dict = fit_handler.array_to_dict(fitted_array, params_to_fit, fit_param_keys)
    fitted_errs_dict = fit_handler.array_to_dict(fitted_errs_array, params_to_fit, fit_param_keys)

    # And then turn those back into planets, flares, and systematics.
    planets, flares, systematics, LD = fit_handler.unpack_params_back_to_dicts(fitted_dict,
                                                                               xpos,
                                                                               ypos,
                                                                               widths)
    
    planets_e, flares_e, systematics_e, LD_e = fit_handler.unpack_params_back_to_dicts(fitted_errs_dict,
                                                                                       xpos,
                                                                                       ypos,
                                                                                       widths)
    
    # Fill in anything that went missing.
    planets, flares, systematics, LD = fit_handler.refill(planets,flares,systematics,LD,
                                                          old_planets,old_flares,old_systematics,old_LD)
    
    planets_e, flares_e, systematics_e, LD_e = fit_handler.refill(planets_e,flares_e,systematics_e,LD_e,
                                                                  old_planets,old_flares,old_systematics,old_LD)
    
    # Re-initialize the planets.
    planets = batman_handler.batman_init_all_planets(time, planets, LD,
                                                     event=inpt_dict["event_type"])
    
    # And return the fitted parameters.
    return planets, flares, systematics, LD, planets_e, flares_e, systematics_e, LD_e