import numpy as np

from juniper.stage5 import batman_handler

def full_model(t, planets, flares, systematics):
    """Builds a model of a transit/eclipse light curve using the provided
    information on observed planets, suspected flaring events, and systematic
    models to detrend for.

    Args:
        t (np.array): time.
        planets (dict): a series of dictionary entries describing each planet
        in the transit curve. They have been batman-initialized so they already
        contain parameters "batman_model" and "batman_params".
        flares (dict): a series of dictionary entries describing each flaring
        event suspected to have occurred during the observation.
        systematics (dict): a series of dictionary entries describing each
        systematic model to detrend for.
    
    Returns:
        np.array, dict: Sys(t;A)*(Sum[planets(t;B)+flares(t;C)]), or the sum of
        planet and flare events multiplied by systematic detrending. Every component
        of the model is also output individually as a dictionary.
    """
    # Initialize flux against time as 1s.
    flx = np.ones_like(t)
    # Track individual models as well.
    models = {}

    # Build planetary flux.
    for planet_name in list(planets.keys()):
        planet = planets[planet_name] # dict, contains rp, rp_prior, fp, fp_prior, etc. as well as batman_model
        batman_flux = batman_handler.batman_flux_update(params_to_fit=None,
                                                        batman_params=planet["batman_params"],
                                                        batman_model=planet["batman_model"])
        # Add planet's flux contribution into the full model.
        flx += batman_flux
        models[planet_name] = batman_flux
    
    # Build flare flux.
    for flare_ID in list(flares.keys()):
        flare = flares[flare_ID] # dict, contains flare parameters
        flare_flux = np.zeros_like(t)
        # WIP!
        # Add flare's flux contribution into the full model.
        flx += flare_flux
        models[flare_ID] = flare_flux

    # Build systematics modifier.
    system = np.ones_like(t)

    # Polynomial trend.
    if systematics["poly"]:
        poly = systematic_polynomial(t, systematics["poly_coeffs"])
        system *= poly
        models["poly"] = poly

    # Mirror tilt event.
    if systematics["mirrortilt"]:
        mirrortilt = systematic_mirrortilt(t, systematics["mirrortilt_coeffs"])
        system *= mirrortilt
        models["mirrortilt"] = mirrortilt

    # Single ramp fit.
    if systematics["singleramp"]:
        expramp = systematic_expramp(t, systematics["singleramp_coeffs"])
        system *= expramp
        models["singleramp"] = expramp

    # Double ramp fit.
    if systematics["doubleramp"]:
        expramp = systematic_doubleramp(t, systematics["doubleramp_coeffs"])
        system *= expramp
        models["doubleramp"] = expramp

    # Position detrend.
    if systematics["pos_detrend"]:
        jitter = systematic_jitter(systematics["xpos"], systematics["ypos"],
                                   systematics["width"], systematics["pos_detrend_coeffs"])
        system *= jitter
        models["pos_detrend"] = jitter

    # And fold all together.
    return system*flx, models

def systematic_polynomial(t, coeffs):
    """Returns a polynomial of specified order in time.

    Args:
        t (np.array): time.
        coeffs (list): polynomial coefficients.

    Returns:
        np.array: polynomial model to be added to Sys(t;A).
    """
    # Set up empty polynomial.
    poly = np.array([0 for i in t])

    # And populate.
    for n, o in enumerate(coeffs):
        poly += o*(t**n)

    return poly

def systematic_expramp(t, coeffs):
    """Returns a single exponential ramp trend in time.

    Args:
        t (np.array): time.
        coeffs (list): exponential ramp coefficients.

    Returns:
        np.array: single ramp model to be added to Sys(t;A).
    """
    single_ramp = 1 + coeffs[0]*np.exp(coeffs[1]*t + coeffs[2])
    return single_ramp

def systematic_doubleramp(t, coeffs):
    """Returns a double exponential ramp trend in time.

    Args:
        t (np.array): time.
        coeffs (list): exponential ramp coefficients.

    Returns:
        np.array: double ramp model to be added to Sys(t;A).
    """
    double_ramp = (1 + coeffs[0]*np.exp(coeffs[1]*t + coeffs[2])
                     + coeffs[3]*np.exp(coeffs[4]*t + coeffs[5]))
    return double_ramp

def systematic_mirrortilt(t, coeffs):
    """Returns a step function modelling an arbitrary number
     of mirror tilt events.

    Args:
        t (np.array): time.
        coeffs (list): mirror tilt coefficients.
    
    Returns:
        np.array: mirror tilts model to be added to Sys(t;A).
    """
    flx = coeffs[0]*np.ones_like(t) # there is a pre-tilt baseline flux [0]
    for n in range(1,len(coeffs)):
        flx[coeffs[n][0]:] += coeffs[n][1] # and then after time index [n][0], there is a step of [n][1] which can be up or down
    return flx

def systematic_jitter(xpos, ypos, widths, coeffs):
    """Returns a polynomial correlated to trace position and width.

    Args:
        xpos (np.array): dispersion position with time.
        ypos (np.array): cross-dispersion position with time.
        widths (np.array): cross-dispersion width with time.
        coeffs (list): jitter polynomial fits.
    
    Returns:
        np.array: mirror tilt step model to be added to Sys(t;A).
    """
    jitter = 1 + coeffs[0]*xpos + coeffs[1]*ypos + coeffs[2]*widths
    return jitter