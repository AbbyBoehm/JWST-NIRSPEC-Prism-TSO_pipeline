import batman

def batman_transit_params(exoplanet_params):
    """Simple function to set up the batman.TransitParams() object.

    Args:
        exoplanet_params (dict): exoplanet parameters.

    Returns:
        batman.TransitParams(): parameters for batman model.
    """
    params = batman.TransitParams()
    params.per = exoplanet_params["period"]                                 #orbital period in days
    params.rp = exoplanet_params["rp"]                                      #planet radius (in units of stellar radii)
    params.t0 = exoplanet_params["t_prim"]                                  #time of inferior conjunction in days
    params.a = exoplanet_params["aor"]                                      #semi-major axis (in units of stellar radii)
    params.inc = exoplanet_params["incl"]                                   #orbital inclination (in degrees)
    params.ecc = exoplanet_params["ecc"]                                    #eccentricity
    params.w = exoplanet_params["longitude"]                                #longitude of periastron (in degrees)
    params.u = exoplanet_params["LD_coeffs"]                                #limb darkening coefficients [u1, u2] or [u1, u2, u3, u4] etc.
    params.limb_dark = exoplanet_params["model_type"]                       #limb darkening model

    return params

def batman_eclipse_params(exoplanet_params):
    """Simple function to set up the batman.TransitParams() object.

    Args:
        exoplanet_params (dict): exoplanet parameters.

    Returns:
        batman.TransitParams(): parameters for batman model.
    """
    params = batman.TransitParams()
    params.per = exoplanet_params["period"]                                 #orbital period in days
    params.rp = exoplanet_params["rp"]                                      #planet radius (in units of stellar radii)
    params.fp = exoplanet_params["fp"]                                      #planet flux (in units of stellar flux)
    params.t_secondary = exoplanet_params["t_seco"]                         #time of superior conjunction in days
    params.a = exoplanet_params["aor"]                                      #semi-major axis (in units of stellar radii)
    params.inc = exoplanet_params["incl"]                                   #orbital inclination (in degrees)
    params.ecc = exoplanet_params["ecc"]                                    #eccentricity
    params.w = exoplanet_params["longitude"]                                #longitude of periastron (in degrees)

    return params

def batman_init_one_model(t, exoplanet_params, event):
    """Simple function to initialize a batman transit or eclipse model.

    Args:
        t (np.array): time.
        exoplanet_params (dict): exoplanet parameters which tells
        batman how to build the model.
        event (str): options are 'primary' or 'secondary'.

    Returns:
        batman.TransitModel(): batman transit model for the transit or eclipse.
    """
    # Translate exoplanet parameters to batman.TransitParams() object.
    if event == 'primary':
        batman_params = batman_transit_params(exoplanet_params)
    if event == 'secondary':
        batman_params = batman_eclipse_params(exoplanet_params)

    # Initialize a batman_model.
    batman_model = batman.TransitModel(batman_params, t, transittype=event)
    return batman_model, batman_params

def batman_init_all_planets(t, planets, event):
    """Wrapper to init models for all planets.

    Args:
        t (np.array): time.
        planets (dict): a series of dictionary entries describing each planet
        in the transit or eclipse curve.
        event (str): options are 'primary' or 'secondary'.

    Returns:
        dict: planets updated with keywords "batman_model" and "batman_params".
    """
    # Update planets to have batman models and parameters.
    for planet_name in list(planets.keys()):
        planet = planets[planet_name] # grab the planet dictionary and update it below
        planet["batman_model"], planet["batman_params"] = batman_init_one_model(t, planet, event)
    return planets

def batman_flux_update(params_to_fit, batman_params, batman_model):
    """Simple function to get the new batman flux model.

    Args:
        params_to_fit (dict): the parameters we are fitting for. Can be None
        if you just want to get the batman flux as-is.
        batman_params (batman.TransitParams()): batman.TransitParams() object
        to update and supply to the batman_model.
        batman_model (batman.TransitModel()): batman.TranstiModel() object
        which returns transit/eclipse flux when supplied with parameters.

    Returns:
        np.array: flux for the transit/eclipse event.
    """
    # Update the batman_params if asked.
    if params_to_fit:
        batman_params = update_batman_params(params_to_fit, batman_params)
    return batman_model.light_curve(batman_params)

def update_batman_params(params_to_fit, batman_params):
    """Simple function to help the params_to_fit dictionary talk
    to the batman.TransitParams() object.

    Args:
        params_to_fit (dict): the parameters we are fitting for.
        batman_params (batman.TransitParams()): batman.TransitParams() object
        which needs to be updated.

    Returns:
        batman.TransitParams(): updated parameteres for batman.
    """
    fit_params = list(params_to_fit.keys())
    if "rp" in fit_params:
        batman_params.rp = params_to_fit["rp"]
    if "fp" in fit_params:
        batman_params.fp = params_to_fit["fp"]
    if "LD_coeffs" in fit_params:
        batman_params.u = params_to_fit["LD_coeffs"]
    if "t_prim" in fit_params:
        batman_params.t0 = params_to_fit["t_prim"]
    if "t_seco" in fit_params:
        batman_params.t_secondary = params_to_fit["t_seco"]
    if "period" in fit_params:
        batman_params.per = params_to_fit["period"]
    if "aor" in fit_params:
        batman_params.a = params_to_fit["aor"]
    if "incl" in fit_params:
        batman_params.inc = params_to_fit["incl"]
    if "ecc" in fit_params:
        batman_params.ecc = params_to_fit["ecc"]
    if "longitude" in fit_params:
        batman_params.w = params_to_fit["longitude"]
    
    return batman_params