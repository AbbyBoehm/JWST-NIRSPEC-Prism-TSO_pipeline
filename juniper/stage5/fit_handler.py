import numpy as np

from juniper.stage5 import models

def bundle_planets_flares_systematics_and_LD(planets,flares,systematics,LD):
    """Simple function which unpacks every provided planet, flare, and
    systematics model and puts them into a single dictionary to be passed to
    least-squares and emcee fitters.

    Args:
        planets (dict): series of entries describing each planet in the model,
        tagged by "planet1", "planet2", etc.
        flares (dict): series of entries describing each flare in the model,
        tagged by "flare1", "flare2", etc.
        systematics (dict): systematics info containing "poly", "poly_coeffs",
        "mirrortilt", "mirrortilt_coeffs", etc.
        LD (dict): information on stellar limb darkening model. If you are fitting
        for this, you need to add its info in.

    Returns:
        dict: contents of all three dictionaries spilled into an array.
    """
    params_to_fit = {}
    for planet_name in planets.keys():
        planet = planets[planet_name]
        for key in planet.keys():
            if "prior" in key:
                continue
            params_to_fit[key] = planet[key]
    for flare_ID in flares.keys():
        flare = flares[flare_ID]
        for key in flare.keys():
            if "prior" in key:
                continue
            params_to_fit[key] = flare[key]
    for key in systematics.keys():
        if "coeffs" not in key:
            continue
        params_to_fit[key] = systematics[key]
    if any(LD["fit_LDs"]):
        # We are indeed fitting LDs so we need to tell params_to_fit what it needs to know.
        params_to_fit["LD_initialguess"] = LD["LD_initialguess"]
        # It needs to know which ones are fixed and which are fitted.
        params_to_fit["fit_LDs"] = LD["fit_LDs"]
    return params_to_fit

def unpack_params_back_to_dicts(params_to_fit, xpos, ypos, widths):
    """Slightly less simple function which takes the unified parameters
    dictionary and separates it back into planets, flares, and systematics.

    Args:
        params_to_fit (dict): contains keys like "rp1", "A1", and "poly_coeffs".
        xpos (np.array): if detrending w.r.t. position, the dispersion position array.
        ypos (np.array): if detrending w.r.t. position, the cross-dispersion position array.
        widths (np.array): if detrending w.r.t. position, the cross-dispersion widths array.

    Returns:
        dict, dict, dict, dict: the planets, flares, systematics and LD dictionaries rebuilt.
    """
    # First, find every planet.
    special_keys = ["rp","fp","t_prim","t_seco","period","aor","incl","ecc",
                    "longitude","batman_model","batman_params"]
    # Now we are going to parse params_to_fit for individual planets.
    planets = {}
    planet_N = 1 # we're going to step up planet_N until we run out of planets
    had_KeyError = False # haven't hit a key error
    while not had_KeyError:
        # Create a key for the Nth planet.
        planet_name = "planet{}".format(planet_N)
        # And a dictionary for that planet.
        planet = {}
        try:
            for key in special_keys:
                planet[key+str(planet_N)] = params_to_fit[key+str(planet_N)] # as long as a planet of this number exists, params_to_fit will have this key
            
            # Now add the planet to the planets.
            planets[planet_name] = planet

            # And step up the planet number.
            planet_N += 1

        except KeyError:
            # We have run out of planet.
            had_KeyError = True
    
    # We found all the planets. Now to find all the flares.
    special_keys = ["A","B","C","Dr","Ds","Fr","E"]

    # Now we are going to parse params_to_fit for individual flares.
    flares = {}
    flare_ID = 1 # we're going to step up flare_ID until we run out of flares
    had_KeyError = False # haven't hit a key error
    while not had_KeyError:
        # Create a key for the next flare.
        flare_name = "flare{}".format(flare_ID)
        # And a dictionary for that flare.
        flare = {}
        try:
            for key in special_keys:
                flare[key+str(flare_ID)] = params_to_fit[key+str(flare_ID)] # as long as a flare of this number exists, params_to_fit will have this key

            # Now add the flare to the flares.
            flares[flare_name] = flare

            # And step up the flare ID number.
            flare_ID += 1

        except KeyError:
            # We have run out of flare.
            had_KeyError = True
    
    # We found the planets and the flares. Now to parse the systematics.
    special_keys = ["poly","mirrortilt","pos_detrend","singleramp","doubleramp"]
    pos_detrend_keys = ["xpos","ypos","widths"]
    systematics = {}
    for key in special_keys:
        try:
            systematics[key+"_coeffs"] = params_to_fit[key+"_coeffs"]
            systematics[key] = True # if we haven't failed yet, then this must be True.
            if key == "pos_detrend":
                for key, pos in zip(pos_detrend_keys, (xpos, ypos, widths)):
                    systematics[key] = pos
        except:
            # If this failed, then we were not fitting that kind of systematic.
            systematics[key] = False

    # Finally, we need to read the LD info back out.
    # This information will be there only if we fit the LDs though.
    special_keys = ["LD_initialguess"]
    LD = {}
    for key in special_keys:
        try:
            LD[key] = params_to_fit[key]
        except:
            # We expect an exception if LD info was not fit to begin with.
            pass

    # And we have now unpacked the params_to_fit dict.
    return planets, flares, systematics, LD

def refill(new_planets, new_flares, new_systematics, new_LD, old_planets, old_flares, old_systematics, old_LD):
    """Checks if the newly-fitted dictionaries are missing anything and replaces the missing entries.

    Args:
        new_planets (dict): series of entries describing the newly-fitted planets.
        new_flares (dict): series of entries describing the newly-fitted flares.
        new_systematics (dict): series of entries describing the newly-fitted systematics.
        new_LD (dict): series of entries describing the newly-fitted limb darkening.
        old_planets (dict): series of entries describing the original planets.
        old_flares (dict): series of entries describing the original flares.
        old_systematics (dict): series of entries describing the original systematics.
        old_LD (dict): series of entries describing the original limb darkening.

    Returns:
        dict, dict, dict: the new planets, flares, and systematics with any holes filled.
    """
    # Check if anything is missing from the new planets.
    for planet_name in new_planets.keys():
        new_planet = new_planets[planet_name]
        old_planet = old_planets[planet_name]
        unfilled_keys = [key for key in old_planet.keys() if key not in new_planet.keys()]
        for key in unfilled_keys:
            new_planet[key] = old_planet[key]
    
    # And the new flares.
    for flare_ID in new_flares.keys():
        new_flare = new_flares[flare_ID]
        old_flare = old_flares[flare_ID]
        unfilled_keys = [key for key in old_flare.keys() if key not in new_flare.keys()]
        for key in unfilled_keys:
            new_flare[key] = old_flare[key]

    # And the systematics.
    unfilled_keys = [key for key in old_systematics.keys() if key not in new_systematics.keys()]
    for key in unfilled_keys:
        new_systematics[key] = old_systematics[key]

    # And the LDs.
    unfilled_keys = [key for key in old_LD.keys() if key not in new_LD.keys()]
    for key in unfilled_keys:
        new_LD[key] = old_LD[key]

    return new_planets, new_flares, new_systematics, new_LD

def dict_to_array(params_to_fit, fit_param_keys):
    """Simple function to take the dictionary of parameters that needs to be
    fitteed and spit out its contents as an array.

    Args:
        params_to_fit (dict): all of the parameters needed to fit, including
        every rp1, rp2, rpN, plus every flare,e systematic model, and LDs.
        fit_param_keys (list of str): the keys that are actually getting
        modified during fitting.

    Returns:
        np.array, dict: an array of the contents of params_to_fit, and a guide
        to which parameters need to be updated.
    """
    # While params_to_fit does contain everything needed to evaluate a fit, not
    # all of its contents are tunables. Plus, some of its contents are lists that
    # must be pulled apart. So let's crack into it.
    fit_param_keys = [str.replace(key,"_prior","") for key in fit_param_keys] # get rid of the prior tag

    params_to_arrayify = []
    # First, copy wholesale what can easily be copied.
    for key in fit_param_keys:
        try:
            if isinstance(params_to_fit[key],float) or isinstance(params_to_fit[key],int):
                # It's a simple float or integer, so we can just tack it on there.
                params_to_arrayify.append(params_to_fit[key])
        except KeyError:
            # This key does not exist in params_to_fit, so it must be a special key (LD, poly, etc.).
            pass

    # Now parse the systematics.
    special_keys = ["poly","mirrortilt","pos_detrend","singleramp","doubleramp"]
    for key in special_keys:
        # There will always be at least coeff1 in any model. So this is a simple
        # way to check that this model is being fitted.
        if key+str(1) in fit_param_keys:
            coeffs = params_to_fit[key+"_coeffs"] # all of the coefficients are bundled here.
            for coeff in coeffs:
                params_to_arrayify.append(coeff)
    
    # Now check out LDs.
    try:
        for i, bool in enumerate(params_to_fit["fit_LDs"]):
            # For every bool here, check if it's fitted.
            if bool:
                params_to_arrayify.append(params_to_fit["LD_initialguess"][i])
    except KeyError:
        # We are not fitting LDs at all, so pass.
        pass

    # Make it an array! Now we can give it to scipy.
    arr = np.array(params_to_arrayify)
    return arr

def array_to_dict(params_array, params_to_fit, fit_param_keys):
    """Slightly less simple function which uses the original params_to_fit
    input dict and the array-ified version of params_to_fit to re-dictionary-ify
    the array.

    Args:
        params_array (np.array): the array-ified version of the input.
        params_to_fit (dict): the original dictionary that was given to the
        fitter when the models.full_model() was initialized.
        fit_param_keys (list of str): the keys that are actually getting
        modified during fitting.

    Returns:
        dict: the parameters back in dictionary form.
    """
    # Remove _prior from keys.
    fit_param_keys = [str.replace(key,"_prior","") for key in fit_param_keys]
    # Open a new dictionary.
    redicted_params = {}
    systematics = {}
    LDs = {}

    # Stash initial LD guess.
    redicted_params["LD_initialguess"] = params_to_fit["LD_initialguess"]
    redicted_params["fit_LDs"] = params_to_fit["fit_LDs"]

    # Keep "organized keys" for later.
    organized_keys = list(params_to_fit.keys())

    # Define systematics keys.
    special_keys = ["poly","mirrortilt","pos_detrend","singleramp","doubleramp"]
    for i, key in enumerate(fit_param_keys):
        # Some of these keys will be able to be transferred wholesale.
        # The exceptions are systematics models (must be bundled as "model_coeffs")
        # and limb darkening coefficients (must be bundled as "LD_initialguess")
        if any([special_key in key for special_key in special_keys]):
            # It's a systematic moodel coefficient! Store it to process later.
            systematics[key+str(i)] = params_array[i]
        elif "LD" in key:
            # It's an LD! Store it to process later.
            LDs[key] = params_array[i]
        else:
            # It's nothing special, just take it as is.
            redicted_params[key] = params_array[i]

    # Now we need to put the systematics back in.
    for special_key in special_keys:
        # The systematics are checked out one by one. poly, then mirrortilt, then etc.
        system_keys = [key for key in systematics.keys() if special_key in key]
        system_model = []
        for key in system_keys:
            system_model.append(systematics[key])
        # And now we bundle all the systematic coefficients together again.
        if system_model:
            # That is, only if that system model is there at all. No need to make empty tags.
            redicted_params[special_key+"_coeffs"] = system_model

    # And let's put the LDs back in.
    for key in LDs.keys():
        # The key itself will tell us redicted_params["LD_initialguess"] items to update.
        index_to_update = int(str.replace(key,"LD",""))-1
        redicted_params["LD_initialguess"][index_to_update] = LDs[key]

    # Finally, fill in anything that is missing.
    for key in params_to_fit.keys():
        if key not in redicted_params.keys():
            # If it was not fitted for, resupply it here.
            redicted_params[key] = params_to_fit[key]

    # It is important that the order of items in the dictionary is correct.
    reorganized_params = {}
    for key in organized_keys:
        reorganized_params[key] = redicted_params[key]
    
    return reorganized_params

def build_priors_dict(planets, flares, systematics, LD):
    """Simple function to get the priors on every fitting parameter.

    Args:
        planets (dict): series of entries describing each planet in the model.
        flares (dict): series of entries describing each flare in the model.
        systematics (dict): series of entries describing systematic trends
        in the model.
        LD (dict): series of entries describing the limb darkening model.
    
    Returns:
        dict: each entry is a list of two numbers and this dict will be fed
        into the build_bounds function.
    """
    # Initialize the priors dict.
    param_priors = {}

    # First, gut every planet.
    special_keys = ["rp","fp","t_prim","t_seco","period","aor","incl","ecc","longitude"]
    special_keys = [i+"_prior" for i in special_keys]

    for i, planet_name in enumerate(planets.keys()):
        planet = planets[planet_name]
        for key in special_keys:
            if planet[key+str(i+1)]: # if this is not None, it's being fitted.
                param_priors[key+str(i+1)] = planet[key+str(i+1)]
    
    # We gutted all the planets. Now to gut all the flares.
    special_keys = ["A","B","C","Dr","Ds","Fr","E"]
    special_keys = [i+"_prior" for i in special_keys]

    for i, flare_ID in enumerate(flares.keys()):
        flare = flares[flare_ID]
        for key in special_keys:
            if flare[key+str(i+1)]: # if this is not None, it's being fitted.
                param_priors[key+str(i+1)] = flare[key+str(i+1)]
    
    # We need to unpack systematic info.
    special_keys = ["poly","mirrortilt","pos_detrend","singleramp","doubleramp"]
    for key in special_keys:
        if systematics[key]:
            # If this systematic is included, we need to put a broad bound on every parameter.
            for i,coeff in enumerate(systematics[key+"_coeffs"]):
                param_priors[key+str(i+1)] = [-1e10,1e10]

    # And LD info, if applicable.
    for i, bool in enumerate(LD["fit_LDs"]):
        # If any of the LDs are getting fit, we need a bound on it.
        if bool:
            param_priors["LD"+str(i+1)] = [-10,10]

    return param_priors

def build_bounds(params_priors, priors_type):
    """Builds bounds for linear least squares fitting.

    Args:
        params_priors (dict): each entry is a list of two numbers which are
        either lower/upper limit (for uniform priors) or mean/sigma (gaussian).
        For Gaussian, bounds will be set as the 5-sigma limits.
        priors_type (str): options of 'uniform' or 'gaussian'.

    Returns:
        tuple: the lower and upper bounds on each parameter to fit.
    """
    # Initialize lists for each boundary.
    #lower_bounds, upper_bounds = [], []
    bounds = []

    # And unpack.
    for key in list(params_priors.keys()):
        if priors_type == 'uniform':
            lower, upper = params_priors[key]
            bounds.append((lower, upper))
            #lower_bounds.append(lower)
            #upper_bounds.append(upper)
        elif priors_type == 'gaussian':
            mean, sigma = params_priors[key]
            bounds.append((mean-(5*sigma),mean+(5*sigma)))
            #lower_bounds.append(mean-(5*sigma))
            #upper_bounds.append(mean+(5*sigma))
    #return (lower_bounds, upper_bounds)
    return bounds

def _residuals(params_array, time, light_curve,  params_to_fit, fit_param_keys, xpos, ypos, widths):
    """Computes the residuals between the full model and the given light curve.

    Args:
        params_array (np.array): the array-ified version of params_to_fit.
        time (np.array): mid-exposure time of each point in the light curve.
        light_curve (np.array): flux at each point in the light curve.
        params_to_fit (dict): the parameters being fitted, in dict form.
        fit_param_keys (list of str): the keys that are actually getting
        modified during fitting.
        xpos (np.array): if detrending w.r.t. position, the dispersion position array.
        ypos (np.array): if detrending w.r.t. position, the cross-dispersion position array.
        widths (np.array): if detrending w.r.t. position, the cross-dispersion widths array.
    """
    # Turn the array back into a dictionary.
    redicted_params = array_to_dict(params_array, params_to_fit, fit_param_keys)
    
    # Then, separate those back into planets, flares, and systematics.
    planets_fit, flares_fit, systematics_fit, LD_fit = unpack_params_back_to_dicts(redicted_params,
                                                                                   xpos, ypos, widths)
    
    # Now redo the flux model calculation, this time supplying redicted_params as an argument.
    model, components = models.full_model(time, planets_fit, flares_fit, systematics_fit,
                                          params_to_fit=redicted_params, fit_param_keys=fit_param_keys)

    # And compare to the data.
    residuals = np.sum((model-light_curve)**2)

    return residuals