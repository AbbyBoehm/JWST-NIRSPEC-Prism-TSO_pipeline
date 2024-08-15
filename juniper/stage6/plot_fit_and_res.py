import os

import numpy as np
import matplotlib.pyplot as plt

from juniper.stage5 import batman_handler, models
from juniper.util.plotting import plot_fit, plot_res

def plot_fit_and_res(t, lc, lc_err, planets, flares, systematics, LD, inpt_dict):
    """Plots the fitted model and residuals for the given light curve
    and model dictionaries.

    Args:
        t (np.array): _description_
        lc (np.array): _description_
        lc_err (np.array): _description_
        planets (dict): _description_
        flares (dict): _description_
        systematics (dict): _description_
        LD (dict): _description_
        inpt_dict (dict): _description_
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

    # And plot.
    fig, axes = plt.subplots(2,1,figsize=(7,5))
    axes[0] = plot_fit(axes[0],t,lc,lc_err,t_interp=t_interp,lc_interp=lc_interp)
    axes[1] = plot_res(axes[1],t,residuals,lc_err)

    # Save plot.
    plt.savefig(os.path.join(inpt_dict["diagnostic_plots"],inpt_dict["name"]),
                dpi=300,bbox_inches='tight')
    plt.close()