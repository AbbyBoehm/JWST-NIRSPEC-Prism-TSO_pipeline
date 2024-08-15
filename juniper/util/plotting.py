import matplotlib.pyplot as plt
from matplotlib.scale import get_scale_names

def img(array, aspect=1, title=None, vmin=None, vmax=None, norm=None, verbose=2):
    """Image plotting utility to plot the given 2D array.

    Args:
        array (np.array): Image you want to plot.
        aspect (int, optional): Aspect ratio. Useful for visualizing narrow arrays. Defaults to 1.
        title (str, optional): Title to give the plot. Defaults to None.
        vmin (float, optional): Minimum value for color mapping. Defaults to None.
        vmax (float, optional): Maximum value for color mapping. Defaults to None.
        norm (str, optional): Type of normalisation scale to use for this image. Defaults to None.
        verbose (int, optional): From 0 to 2. Specifies how many print statements to issue. Defaults to 2.

    Returns:
        figure, axis, image: matplotlib figure environment, axis object, and image mappable.
    """
    fig, ax = plt.subplots(figsize=(20, 25))
    if (norm == None or norm not in get_scale_names()):
        # Either they didn't specify, or they specified incorrectly. In either case, default to linear.
        if verbose == 2:
            print("Plot normalization unspecified or unrecognized, defaulting to 'linear'...")
        norm = 'linear'
    im = ax.imshow(array, aspect=aspect, norm=norm, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(mappable=im,fraction=min(0.5/aspect,0.15))
    ax.set_title(title)
    return fig, ax, im

def plot_fit(ax, t, lc, lc_err, t_interp, lc_interp):
    """Plots the fitted model over the data.

    Args:
        ax (_type_): _description_
        t (_type_): _description_
        lc (_type_): _description_
        lc_err (_type_): _description_
        t_interp (_type_): _description_
        lc_interp (_type_): _description_
    """
    if lc_err:
        ax.errorbar(t, lc, yerr=lc_err, fmt='ko')
    else:
        ax.scatter(t, lc, color='k')
    
    ax.plot(t_interp, lc_interp, color='red')
    return ax

def plot_res(ax, t, res, lc_err):
    """Plots the residuals on the given axis.

    Args:
        ax (_type_): _description_
        t (_type_): _description_
        res (_type_): _description_
        lc_err (_type_): _description_
    """
    if lc_err:
        ax.errorbar(t, res, yerr=lc_err, fmt='ko')
    else:
        ax.scatter(t, res, color='k')

    ax.axhline(y=0,color='red',ls='--')
    return ax