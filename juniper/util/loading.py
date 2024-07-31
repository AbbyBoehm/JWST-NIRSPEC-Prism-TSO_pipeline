from tqdm import tqdm

import numpy as np
import xarray as xr
from astropy.io import fits

from jwst import datamodels as dm

def stitch_files(files, time_step, verbose):
    """Reads all supplied files and stitches them together into a single array.

    Args:
        files (lst of str): filepaths to files that are to be loaded.
        time_ints (bool): whether to report timing with tqdm.
        verbose (int): from 0 to 2. How much logging to do.

    Returns:
        xarray: loaded data.
    """
    # Log.
    if verbose >= 1:
        print("Stitching data files together for post-processing...")
    
    if verbose == 2:
        print("Will stitch together the following files:")
        for i, f in enumerate(files):
            print(i, f)

    # Initialize some empty lists.
    data, err, wav, dq = [], [], [], [] # the data_vars of the xarray
    time = [] # the coords of the xarray
    int_count = [] # the attributes of the array

    # Read in each file.
    for file in tqdm(files,
                     desc = 'Stitching files...',
                     disable=(not time_step)):
        data_i, err_i, int_count_i, wav_i, dq_i, time_i = read_one_datamodel(file)
        # 1D and 2D objects append right away
        int_count.append(int_count_i)
        time.append(time_i)

        # 3D objects may need a little more nuance
        for i in range(data.shape[0]):
            data.append(data_i[i])
            err.append(err_i[i])
            wav.append(wav_i[i])
            dq.append(dq_i[i])

    # Now convert to xarray.
    segment = xr.Dataset(data_vars=dict(
                                    data=(["time", "x", "y"], data),
                                    err=(["time", "x", "y"], err),
                                    dq = (["time", "x", "y"], dq),
                                    wav = (["time", "x", "y"], wav),
                                    ),
                        coords=dict(
                                time = (["time"], time),
                                ),
                        attrs = dict(
                                integrations = int_count,
                                )
    )

    # Log.
    if verbose >= 1:
        print("Files stitched together into xarray.")
    
    return segment
    
def read_one_datamodel(file):
    """Read one .fits file as a datamodel and return its attributes.

    Args:
        file (str): path to the .fits file you want to read out.

    Returns:
        np.array, np.array, int, np.array, np.array, np.array: the data, errors, integration count, wavelength solution, data quality array, and exposure mid-times.
    """
    with dm.open(file) as f:
         data = f.data
         err = f.err
         int_count = data.shape[0]
         wav = f.wavelength
         dq = f.dq
         t = f.int_times["int_mid_MJD_UTC"]
    return data, err, int_count, wav, dq, t

def read_one_postproc(file):
    """Read one post-processing .nc file and return its attributes.

    Args:
        file (str): path to the .nc file you want to read out.

    Returns:
        xarray: xarray representing one post-processed fits file.
    """
    segment = ':D'
    return segment