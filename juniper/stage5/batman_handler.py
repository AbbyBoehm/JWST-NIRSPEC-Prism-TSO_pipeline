import batman

def batman_transit_params(exoplanet_params, planet_ID, LD_initialguess, model_type):
    """Simple function to set up the batman.TransitParams() object.

    Args:
        exoplanet_params (dict): exoplanet parameters.
        planet_ID (str): the ID of this planet.
        LD_initialguess (list): the limb darkening coefficients.
        model_type (str): the type of limb darkening model in use.

    Returns:
        batman.TransitParams(): parameters for batman model.
    """
    params = batman.TransitParams()
    params.per = exoplanet_params["period"+planet_ID]                                 #orbital period in days
    params.rp = exoplanet_params["rp"+planet_ID]                                      #planet radius (in units of stellar radii)
    params.t0 = exoplanet_params["t_prim"+planet_ID]                                  #time of inferior conjunction in days
    params.a = exoplanet_params["aor"+planet_ID]                                      #semi-major axis (in units of stellar radii)
    params.inc = exoplanet_params["incl"+planet_ID]                                   #orbital inclination (in degrees)
    params.ecc = exoplanet_params["ecc"+planet_ID]                                    #eccentricity
    params.w = exoplanet_params["longitude"+planet_ID]                                #longitude of periastron (in degrees)
    params.u = LD_initialguess                                                        #limb darkening coefficients [u1, u2] or [u1, u2, u3, u4] etc.
    params.limb_dark = model_type                                                     #limb darkening model

    return params

def batman_eclipse_params(exoplanet_params, planet_ID):
    """Simple function to set up the batman.TransitParams() object.

    Args:
        exoplanet_params (dict): exoplanet parameters.
        planet_ID (str): the ID of this planet.

    Returns:
        batman.TransitParams(): parameters for batman model.
    """
    params = batman.TransitParams()
    params.per = exoplanet_params["period"+planet_ID]                                 #orbital period in days
    params.rp = exoplanet_params["rp"+planet_ID]                                      #planet radius (in units of stellar radii)
    params.fp = exoplanet_params["fp"+planet_ID]                                      #planet flux (in units of stellar flux)
    params.t_secondary = exoplanet_params["t_seco"+planet_ID]                         #time of superior conjunction in days
    params.a = exoplanet_params["aor"+planet_ID]                                      #semi-major axis (in units of stellar radii)
    params.inc = exoplanet_params["incl"+planet_ID]                                   #orbital inclination (in degrees)
    params.ecc = exoplanet_params["ecc"+planet_ID]                                    #eccentricity
    params.w = exoplanet_params["longitude"+planet_ID]                                #longitude of periastron (in degrees)

    return params

def batman_init_one_model(t, exoplanet_params, event, planet_ID, LD_initialguess, model_type):
    """Simple function to initialize a batman transit or eclipse model.

    Args:
        t (np.array): time.
        exoplanet_params (dict): exoplanet parameters which tells
        batman how to build the model.
        event (str): options are 'primary' or 'secondary'.
        planet_ID (str): the ID of this planet.
        LD_initialguess (list): the limb darkening coefficients.
        model_type (str): the type of limb darkening model in use.

    Returns:
        batman.TransitModel(): batman transit model for the transit or eclipse.
    """
    # Translate exoplanet parameters to batman.TransitParams() object.
    if event == 'primary':
        batman_params = batman_transit_params(exoplanet_params, planet_ID, LD_initialguess, model_type)
    if event == 'secondary':
        batman_params = batman_eclipse_params(exoplanet_params, planet_ID)

    # Initialize a batman_model.
    batman_model = batman.TransitModel(batman_params, t, transittype=event)
    return batman_model, batman_params

def batman_init_all_planets(t, planets, LD, event):
    """Wrapper to init models for all planets.

    Args:
        t (np.array): time.
        planets (dict): a series of dictionary entries describing each planet
        in the transit or eclipse curve.
        LD (dict): instructions on handling stellar limb darkening, necessary for
        talking to batman.
        event (str): options are 'primary' or 'secondary'.

    Returns:
        dict: planets updated with keywords "batman_model" and "batman_params".
    """
    # Grab the LD info we need to properly initialize batman.
    LD_initialguess = LD["LD_initialguess"]
    model_type = LD["LD_model"]

    # Update planets to have batman models and parameters.
    for planet_name in list(planets.keys()):
        # Grab the planet-specific dict.
        planet = planets[planet_name]

        # Supply its ID number so that we can read out the right tags.
        planet_ID = str.replace(planet_name,"planet","")
        planet["batman_model"+planet_ID], planet["batman_params"+planet_ID] = batman_init_one_model(t, planet, event, planet_ID, LD_initialguess, model_type)
    return planets

# You ever stare at a screen so long you stop noticing the word 'batman'?

def batman_flux_update(params_to_fit, fit_param_keys, batman_params, batman_model):
    """Simple function to get the new batman flux model.

    Args:
        params_to_fit (list): the parameters we are fitting for. Can be None
        if you just want to get the batman flux as-is.
        fit_param_keys (list of str): the parameters that are actually being updated
        during fitting. Guides the whole process.
        batman_params (list): list of batman.TransitParams() objects to update
        and supply to the batman_model objects.
        batman_model (list): batman.TranstiModel() objects which return
        transit/eclipse flux when supplied with parameters.

    Returns:
        np.array: total flux for the transit/eclipse events.
    """
    # Update the batman_params if asked.
    if params_to_fit:
        # Need to update the params for each model.
        for i, (batman_params_i, batman_model_i) in enumerate(zip(batman_params,batman_model)):
            batman_params_i = update_batman_params(params_to_fit, fit_param_keys, batman_params_i, str(i+1))
    
    # And calculate and sum bat_flux.
    for i, (batman_params_i, batman_model_i) in enumerate(zip(batman_params,batman_model)):
        if i == 0:
            bat_flux = batman_model_i.light_curve(batman_params_i)
        else:
            bat_flux += batman_model_i.light_curve(batman_params_i)
    return bat_flux

def update_batman_params(params_to_fit, fit_param_keys, batman_params, planet_ID):
    """Simple function to help the params_to_fit dictionary talk
    to the batman.TransitParams() object.

    Args:
        params_to_fit (dict): the fit parameters in dict form.
        fit_param_keys (list of str): the parameters that are actually being updated
        during fitting. Guides the whole process.
        batman_params (batman.TransitParams()): batman.TransitParams() object
        which needs to be updated
        planet_ID (str): number of the planet being worked on. Helps grab
        the correct tags.
        
    Returns:
        batman.TransitParams(): updated parameters for batman.
    """
    # First, remove the prior tag so it's just the proper keys.
    fit_params = [str.replace(key, "_prior", "") for key in fit_param_keys]
    if "rp"+planet_ID in fit_params:
        batman_params.rp = params_to_fit["rp"+planet_ID]
    if "fp"+planet_ID in fit_params:
        batman_params.fp = params_to_fit["fp"+planet_ID]
    if "t_prim"+planet_ID in fit_params:
        batman_params.t0 = params_to_fit["t_prim"+planet_ID]
    if "t_seco"+planet_ID in fit_params:
        batman_params.t_secondary = params_to_fit["t_seco"+planet_ID]
    if "period"+planet_ID in fit_params:
        batman_params.per = params_to_fit["period"+planet_ID]
    if "aor"+planet_ID in fit_params:
        batman_params.a = params_to_fit["aor"+planet_ID]
    if "incl"+planet_ID in fit_params:
        batman_params.inc = params_to_fit["incl"+planet_ID]
    if "ecc"+planet_ID in fit_params:
        batman_params.ecc = params_to_fit["ecc"+planet_ID]
    if "longitude"+planet_ID in fit_params:
        batman_params.w = params_to_fit["longitude"+planet_ID]

    # LD has a slight bit of nuance to it.
    if any(params_to_fit["fit_LDs"]):
        # This will have been updated before.
        batman_params.u = params_to_fit["LD_initialguess"]
    
    return batman_params