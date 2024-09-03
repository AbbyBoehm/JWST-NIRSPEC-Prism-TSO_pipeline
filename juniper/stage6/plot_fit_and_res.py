import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    # Array-ify as needed.
    t = np.asarray(t)
    lc = np.asarray(lc)
    lc_err = np.asarray(lc_err)

    # Initialize the planets, giving them the LD info they need to talk to batman properly.
    planets = batman_handler.batman_init_all_planets(t, planets, LD,
                                                     event=inpt_dict["event_type"])
    
    # Create the full model and components.
    full_flux, components = models.full_model(t, planets, flares, systematics,
                                              None, None)
    
    # Compute the residuals.
    residuals = full_flux-lc

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
    # Array-ify as needed.
    t = np.asarray(t)
    lc = np.asarray(lc)
    lc_err = np.asarray(lc_err)

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
        
        axes[0].scatter(bin_t,bin_lc,color='blue',alpha=0.5, zorder=1)
        axes[1].scatter(bin_t,bin_res,color='blue',alpha=0.5, zorder=1)

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
    # Array-ify.
    wavelengths = np.asarray(wavelengths)
    # Initialize the waterfall plot.
    fig, axes = plt.subplots(1,2,figsize=(4,10))

    # For each light curve, offset it by an increasing amount.
    spacing = inpt_dict["waterfall_space"]/1e6
    offsets = [-i*spacing for i in range(len(ts))]

    # Normalize and apply the offsets.
    for i in tqdm(range(len(offsets)),
                  desc='Normalizing and offsetting models...'):
        # First, normalize the light curve, model, and residuals.
        lcs[i] /= np.median(lcs[i])
        lc_interps[i] /= np.median(lc_interps[i])
        residualses[i] /= np.median(lcs[i])

        # Then, offset and swap to ppm.
        lcs[i] = [(1e6*k)+offsets[i] for k in lcs[i]]
        lc_interps[i] = [(1e6*k)+offsets[i] for k in lc_interps[i]]
        residualses[i] = [(1e6*k)+offsets[i] for k in residualses[i]]

    # Define a color map for each light curve.
    print("Setting up colormap bounds...")
    wav_bounds = [np.min(wavelengths),np.max(wavelengths)]
    if len(inpt_dict["waterfall_wavs"]) != 0:
        # Use these as our bounds instead.
        wav_bounds = inpt_dict["waterfall_wavs"]
    
    # Normalize.
    print("Normalizing wavelengths for colormap...")
    wavelengths -= wav_bounds[0] # subtract the lowest value.
    wavelengths[wavelengths <= 0] = 0 # anything lower than that minimum is 0.
    wavelengths /= wav_bounds[1] # and normalize by the highest value.
    wavelengths[wavelengths >= 1] = 1 # anything above that maximum is 1.

    # Define the colormap object.
    print("Defining colormap...")
    colormap = mpl.colormaps[inpt_dict['waterfall_cmap']]

    # Then plot.
    ppm_min = 0
    ppm_max = 0
    for i in tqdm(range(len(ts)),
                  desc='Plotting waterfall models...'):
        wave = wavelengths[i]
        t = ts[i]
        lc = lcs[i]
        lc_err = lc_errs[i]
        t_interp = t_interps[i]
        lc_interp = lc_interps[i]
        if inpt_dict['waterfall_bin']:
            t = time_bin(np.asarray(t),inpt_dict["waterfall_bin"],'median')
            lc = time_bin(np.asarray(lc),inpt_dict["waterfall_bin"],'median')
            lc_err = time_bin(np.asarray(lc_err),inpt_dict["waterfall_bin"],'median')
        axes[0].scatter(t,lc,color=colormap(wave))
        axes[0].errorbar(t,lc,yerr=lc_err,ls='none',color=colormap(wave),capsize=3)
        axes[0].plot(t_interp,lc_interp,color=colormap(wave))

    for i in tqdm(range(len(ts)),
                  desc='Plotting waterfall residuals...'):
        wave = wavelengths[i]
        t = ts[i]
        residual = residualses[i]
        lc_err = lc_errs[i]
        offset = offsets[i]
        if inpt_dict['waterfall_bin']:
            t = time_bin(np.asarray(t),inpt_dict["waterfall_bin"],'median')
            residual = time_bin(np.asarray(residual),inpt_dict["waterfall_bin"],'median')
            lc_err = time_bin(np.asarray(lc_err),inpt_dict["waterfall_bin"],'median')
        axes[1].plot(t,[offset for i in t],color='k',ls='--')
        axes[1].scatter(t,residual,color=colormap(wave))
        axes[1].errorbar(t,residual,yerr=lc_err,ls='none',color=colormap(wave),capsize=3)
    
    axes[0].set_xlabel('time [mjd_utc]')
    axes[1].set_xlabel('time [mjd_utc]')
    axes[0].set_ylabel('flux+offsets [normalized]')
    axes[1].set_ylabel('residuals+offsets [ppm]')
    return fig, axes