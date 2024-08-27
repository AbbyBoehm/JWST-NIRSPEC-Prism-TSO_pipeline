import os
from tqdm import tqdm

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colormaps as C

from juniper.stage5 import batman_handler, models
from juniper.stage5.bin_light_curves import time_bin
from juniper.stage6.plot_fit_and_res import get_fit_and_res
from juniper.util.plotting import plot_fit, plot_res
from juniper.util.diagnostics import tqdm_translate, plot_translate, timer

def plot_model_panel(t, lc, lc_err, planets, flares, systematics, LD, inpt_dict):
    """Plots each part of the fitted model.

    Args:
        t (np.array): mid-exposure timestamps of the light curve.
        lc (np.array): flux of the light curve in arbitrary units.
        lc_err (np.array): uncertainties of the light curve in arbitrary
        units.
        planets (dict): a dictionary of every fitted planet.
        flares (dict): a dictionary of every fitted flare.
        systematics (dict): a dictionary of every fitted systematic.
        LD (dict): a dictionary of the limb darkening model.
        inpt_dict (dict): instructions for running this step.

    Returns:
        fig, axes: matplotlib plot with the model pieces on it.
    """
    # Get the model components and how many there are.
    t_interp, lc_interp, components, residuals = get_fit_and_res(t, lc, lc_err, planets, flares, systematics, LD, inpt_dict)
    N_comps = len(list(components.keys()))

    # Try to find the next largest square to N_comps.
    sqrt = math.sqrt(N_comps)
    n_cols = math.ceil(sqrt)

    # We need the fist n_rows such that n_rows x n_cols >= N_comps.
    n_rows = 0
    while n_rows * n_cols < N_comps:
        n_rows += 1
    
    # Build a grid.
    template = [str(i) for i in range(n_cols)]
    total = ['a' for i in template]
    mosaic = [total,total,] # first two rows are for the total model.
    keys = []
    for n in n_rows:
        mosaic.append([str(n)+i for i in template])
        for key in mosaic[-1]:
            keys.append(key)

    # We'll plot a total model on top and panels for each component on the sides.
    fig, axes = plt.subplot_mosaic(mosaic=mosaic, figsize=(10,10))

    axes['a'].errorbar(t,lc,yerr=lc_err,fmt='ko',ls='none',capsize=3,label='light curve')
    axes['a'].plot(t_interp, lc_interp, color='red',label='full model')
    axes['a'].legend(loc='upper right')
    
    if inpt_dict["plot_bin"]:
        bin_t = time_bin(t,bin_size=inpt_dict["plot_bin"],mode='median')
        bin_lc = time_bin(lc,bin_size=inpt_dict["plot_bin"],mode='median')
        axes['a'].scatter(t,lc,color='blue',alpha=0.5)
    
    for component, key in zip(list(components.keys()),keys):
        axes[key].plot(t_interp,components[component],color='darkblue')
        axes[key].title(component)

    return fig, axes
    