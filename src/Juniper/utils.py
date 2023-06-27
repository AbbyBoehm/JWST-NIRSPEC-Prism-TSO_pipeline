import matplotlib.pyplot as plt


def img(array, aspect=1, title=None, vmin=None, vmax=None, norm=None):
    '''
    Image plotting utility to plot the input 2D array.
    
    :param array: 2D array. Image you want to plot.
    :param aspect: float. Aspect ratio. Useful for visualizing narrow arrays.
    :param title: str. Title to give the plot.
    :param vmin: float. Minimum value for color mapping.
    :param vmax: float. Maximum value for color mapping.
    :param norm: str. Type of normalisation scale to use for this image.
    '''
    fig, ax = plt.subplots(figsize=(20, 25))
    if norm == None:
        im = ax.imshow(array, aspect=aspect, origin="lower", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(array, aspect=aspect, norm=norm, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    return fig, ax, im