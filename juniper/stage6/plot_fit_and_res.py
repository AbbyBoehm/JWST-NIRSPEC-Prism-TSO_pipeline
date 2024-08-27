import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as C

from juniper.stage5 import batman_handler, models
from juniper.stage5.bin_light_curves import time_bin
from juniper.util.plotting import plot_fit, plot_res
from juniper.util.diagnostics import tqdm_translate, plot_translate, timer

def get_fit_and_res(t, lc, lc_err, planets, flares, systematics, LD, inpt_dict):
    """Gets the fitted model and residuals for the given light curve
    and model dictionaries.

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
    """
    # Initialize the planets, giving them the LD info they need to talk to batman properly.
    planets = batman_handler.batman_init_all_planets(t, planets, LD,
                                                     event=inpt_dict["event_type"])
    
    # Create the full model and components.
    full_flux, components = models.full_model(t, planets, flares, systematics,
                                              None, None)
    
    # Compute the residuals.
    residuals = ((full_flux-lc))**2

    # Create interpolated model.
    t_interp = np.linspace(np.min(t),np.max(t),1000)
    planets = batman_handler.batman_init_all_planets(t_interp, planets, LD,
                                                     event=inpt_dict["event_type"])
    
    # Create the full model and components.
    lc_interp, components = models.full_model(t_interp, planets, flares, systematics,
                                              None, None)
    
    return t_interp, lc_interp, components, residuals

def plot_fit_and_res(t, lc, lc_err, planets, flares, systematics, LD, inpt_dict):
    """Gets the fitted model and residuals for the given light curve
    and model dictionaries.

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
        fig, axes: matplotlib plot with the fit and res on it.
    """
    # Get the needed info from get_fit_and_res.
    t_interp, lc_interp, components, residuals = get_fit_and_res(t, lc, lc_err, planets, flares, systematics, LD, inpt_dict)

    # And plot.
    fig, axes = plt.subplots(2,1,figsize=(7,5))
    axes[0] = plot_fit(axes[0],t,lc,lc_err,t_interp=t_interp,lc_interp=lc_interp)
    axes[1] = plot_res(axes[1],t,residuals,lc_err)

    if inpt_dict["plot_bin"]:
        bin_t = time_bin(t,inpt_dict["plot_bin"],'median')
        bin_lc = time_bin(lc,inpt_dict["plot_bin"],'median')
        bin_res = time_bin(residuals,inpt_dict["plot_bin"],'median')
        
        axes[0].scatter(bin_t,bin_lc,color='blue',alpha=0.5)
        axes[1].scatter(bin_t,bin_res,color='blue',alpha=0.5)

    return fig, axes

def plot_waterfall(wavelengths, ts, lcs, lc_errs, t_interps, lc_interps, residualses, inpt_dict):
    """Plots a waterfall of the given light curves and fits.

    Args:
        wavelengths (np.array): central wavelength for each light curve.
        ts (list): timestamps for each curve.
        lcs (list): light curves for each curve.
        lc_errs (list): light curve uncertainties for each curve.
        t_interps (list): model timestamps for each curve.
        lc_interps (list): model fluxes for each curve.
        residualses (list): residuals for each model.
        inpt_dict (dict): instructions for running this step.

    Returns:
        fig, axes: the waterfall plot.
    """
    # Initialize the waterfall plot.
    fig, axes = plt.subplots(1,2,figsize=(7,20))

    # For each light curve, offset it by an increasing amount.
    spacing = inpt_dict["waterfall_space"]/1e6
    offsets = [-i*spacing for i in range(len(ts))]

    # Normalize and apply the offsets.
    for i, (lc, lc_interp, res, offset) in enumerate(zip(lcs, lc_interp, residualses, offsets)):
        # First, normalize the light curve, model, and residuals.
        lcs[i] /= np.median(lc)
        lc_interps[i] /= np.median(lc_interp)
        residualses[i] /= np.median(lc)

        # Then, offset and turn residuals into ppm.
        lcs[i] = [k+offset for k in lcs[i]]
        lc_interps[i] = [k+offset for k in lc_interps[i]]
        residualses[i] = [(k+offset)*1e6 for k in res]

    # Define a color map for each light curve.
    wav_bounds = [np.min(wavelengths),np.max(wavelengths)]
    if inpt_dict["waterfall_wavs"]:
        # Use these as our bounds instead.
        wav_bounds = inpt_dict["waterfall_wavs"]
    
    # Normalize.
    wavelengths -= wav_bounds[0] # subtract the lowest value.
    wavelengths[wavelengths <= 0] = 0 # anything lower than that minimum is 0.
    wavelengths /= wav_bounds[1] # and normalize by the highest value.
    wavelengths[wavelengths >= 1] = 1 # anything above that maximum is 1.

    # Define the colormap object.
    colormap = C[inpt_dict['waterfall_cmap']]

    # Then plot.
    for wave, t, lc, lc_err, t_interp, lc_interp in zip(wavelengths,ts,lcs,lc_errs,
                                                        t_interps,lc_interps):
        axes[0].scatter(t,lc,c=colormap(wave))
        axes[0].errorbar(t,lc,yerr=lc_err,ls='none',c=colormap(wave),capsize=3)
        axes[0].plot(t_interp,lc_interp,color=colormap(wave))

    for wave, t, residual, lc_err, offset in zip(wavelengths,ts,residualses,lc_errs,offsets):
        axes[1].plot(t,[offset for i in t],color='k',ls='--')
        axes[1].scatter(t,residual,c=colormap(wave))
        axes[1].errorbar(t,residual,yerr=lc_err,ls='none',c=colormap(wave),capsize=3)
    
    axes[0].set_xlabel('time [mjd_utc]')
    axes[1].set_xlabel('time [mjd_utc]')
    axes[0].set_ylabel('flux+offsets [normalized]')
    axes[1].set_ylabel('residuals+offsets [ppm]')
    return fig, axes