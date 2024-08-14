import numpy as np
from scipy import signal
from astropy.stats import sigma_clip

def median_spatial_filter(data, sigma, kernel):
    """Cleans one 2D array with median spatial filtering. 
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).

    Args:
        data (np.array): 2D array of data.
        sigma (float): sigma threshold at which to reject outliers.
        kernel (tuple): tuple of two ints which must be odd. Kernal used for median filtering.

    Returns:
        _type_: _description_
    """
    '''
    Cleans one 2D array with median spatial filtering.
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).
    
    :param data: 2D array. Array that will be median-filtered.
    :param sigma: float. Sigma at which to reject outliers.
    :param kernel: tuple of odd int. Kernel to use for median filtering.
    :return: cleaned 2D array.
    '''
    medfilt = signal.medfilt2d(data, kernel)
    diff = data - medfilt
    temp = sigma_clip(diff, sigma=sigma, axis=0)
    mask = temp.mask
    int_mask = mask.astype(float) * medfilt
    test = (~mask).astype(float)
    return (data*test) + int_mask

def colbycol_bckg(data, bckg_rows=[], trace_mask=None):
    """Performs column-by-column background subtraction on a given array.

    Args:
        data (np.array): 2D array of data, row x col.
        bckg_rows (list, optional): list of integers which defines the background rows. Defaults to [].
        trace_mask (np.ma.masked_array, optional): mask to hide trace pixels with. Defaults to None.

    Returns:
        np.array: data array with column-by-column noise removed, and background of that noise.
    """
    # Define the background region using background rows and/or masks, if applicable.
    background_region = np.ma.masked_array(data,mask=trace_mask)
    if bckg_rows:
         background_region = background_region[bckg_rows, :]
            
    # Define the median background in each column and extend to a full-size array.
    background = np.ma.median(background_region, axis=0)
    background = np.array([background,]*data.shape[0])

    # And remove background from data.
    data -= background
    return data, background

def get_trace_mask(data, threshold=10000):
    """Build a mask using the given 2D data frame.

    Args:
        data (np.array): 2D array of data.
        threshold (float, optional): counts level at which to declare something
        as definitely part of the trace. Defaults to 10000.
    
    Returns:
        np.ma.mask: np mask hiding trace pixels.
    """
    # Determine statistics of region assumed to be background by count level.
    mu = np.median(data[data<threshold])
    sig = np.std(data[data<threshold])

    # If the data /positively/ exceeds the mean background level by a 95%
    # significant amount, it is definitely trace and must be masked.
    masked_fg = np.ma.masked_where(data - mu > 2*sig, data)
    return np.ma.getmask(masked_fg)