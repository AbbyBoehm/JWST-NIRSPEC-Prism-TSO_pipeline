import numpy as np

def dict_to_array(params_to_fit):
    """Simple function to take the dictionary of parameters that needs to be
    fitteed and spit out its contents as an array.

    Args:
        params_to_fit (dict): all of the parameters to fit, including every
        rp1, rp2, rpN, plus every systematic model.

    Returns:
        np.array: an array of the contents of params_to_fit.
    """
    arr = np.array([params_to_fit[key] for key in list(params_to_fit.keys())])
    return arr

def array_to_dict(params_array, params_to_fit):
    """Slightly less simple function which uses the original params_to_fit
    input dict and the array-ified version of params_to_fit to re-dictionary-ify
    the array.

    Args:
        params_array (np.array): the array-ified version of the input.
        params_to_fit (dict): the original dictionary that was given to the
        fitter when the models.full_model() was initialized.

    Returns:
        dict: the parameters back in dictionary form.
    """
    redicted_params = {}
    for i, key in enumerate(list(params_to_fit.keys())):
        redicted_params[key] = params_array[i]
    return redicted_params

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
    lower_bounds, upper_bounds = [], []

    # And unpack.
    for key in list(params_priors.keys()):
        if priors_type == 'uniform':
            lower, upper = params_priors[key]
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        elif priors_type == 'gaussian':
            mean, sigma = params_priors[key]
            lower_bounds.append(mean-(5*sigma))
            upper_bounds.append(mean+(5*sigma))
    return (lower_bounds, upper_bounds)