import time
from tqdm import tqdm

import numpy as np
from scipy.optimize import minimize

from juniper.stage5 import batman_handler
from juniper.util.diagnostics import tqdm_translate, plot_translate, timer

def lsq_one(time, light_curve, pos,
            planets, flares, systematics,
            inpt_dict):
    """Performs linear least squares fitting on the given array(s) using scipy.
    More than one light curve may be fit at a time (i.e. fitting two simultaneous
    transits observed on different detectors).

    Args:
        time (np.array): mid-exposure times for each point in the light curve.
        light_curve (np.array): median-normalized flux with time.
        pos (dict): xpos, ypos, and widths of the trace with time.
        planets (dict): uninitialized planet dictionaries which need to be
        initialized with the batman_handler.
        flares (dict): a series of dictionary entries describing each flaring
        event suspected to have occurred during the observation.
        systematics (dict): a series of dictionary entries describing each
        systematic model to detrend for.
        inpt_dict (dict): instructions for running this step.

    Return:
        ???: full model, individual components of the model, and all
        parameters both fixed and fitted.
    """
    