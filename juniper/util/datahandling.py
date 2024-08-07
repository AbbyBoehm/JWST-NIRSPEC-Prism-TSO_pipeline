import os
from tqdm import tqdm

import numpy as np
import xarray as xr
from astropy.io import fits

from jwst import datamodels as dm

def stitch_files(files, time_step, verbose):
    """Reads all supplied files and stitches them together into a single array.

    Args:
        files (lst of str): filepaths to files that are to be loaded.
        time_step (bool): whether to report timing with tqdm.
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
    data, err, dq, cdisp, disp, wav = [], [], [], [], [], [] # the data_vars of the xarray
    time = [] # the coords of the xarray
    int_count, flagged, details = [], [], [] # the attributes of the array

    # Read in each file.
    for file in tqdm(files,
                     desc = 'Stitching files...',
                     disable=(not time_step)):
        if ".fits" in file:
            # It's Stage 2 output.
            data_i, err_i, int_count_i, wav_i, dq_i, time_i, details_i = read_one_datamodel(file)
            wav_i = [wav_i for i in range(time_i.shape[0])]
            # Placeholder empty arrays.
            disp_i, cdisp_i, flagged_i = np.zeros_like(time_i), np.zeros_like(time_i), np.zeros_like(time_i)
        elif ".nc" in file:
            # It's Stage 3 output.
            data_i, err_i, int_count_i, wav_i, dq_i, time_i, disp_i, cdisp_i, flagged_i = read_one_postproc(file)
        # Attributes are appended once.
        int_count.append(int_count_i)
        flagged.append(flagged_i)
        details.append(details_i)

        # Datavars and coords are unpacked.
        for i in range(data_i.shape[0]):
            data.append(data_i[i])
            err.append(err_i[i])
            dq.append(dq_i[i])
            time.append(time_i[i])
            disp.append(disp_i[i])
            cdisp.append(cdisp_i[i])
            wav.append(wav_i[i])

    # Now convert to xarray.
    segments = xr.Dataset(data_vars=dict(
                                    data=(["time", "x", "y"], data),
                                    err=(["time", "x", "y"], err),
                                    dq = (["time", "x", "y"], dq),
                                    disp = (["time"], disp),
                                    cdisp = (["time"], cdisp),
                                    wavelengths = (["time", "x", "y"], wav),
                                    ),
                        coords=dict(
                               time = (["time"], time),
                               ),
                        attrs=dict(
                              integrations = int_count,
                              flagged = flagged,
                              details = details,
                              )
    )

    # Log.
    if verbose >= 1:
        print("Files stitched together into xarray.")
    
    return segments
    
def read_one_datamodel(file):
    """Read one .fits file as a datamodel and return its attributes.

    Args:
        file (str): path to the .fits file you want to read out.

    Returns:
        np.array, np.array, int, np.array, np.array, np.array: the data, errors, integration count, wavelength solution, data quality array, and exposure mid-times.
    """
    with dm.open(file) as f:
         # Get data.
         data = f.data
         err = f.err
         int_count = data.shape[0]
         wav = f.wavelength
         dq = f.dq
         t = f.int_times["int_mid_MJD_UTC"]

         # And get observation details.
         obs_instrument = f.header["INSTRUME"]
         obs_detector = f.header["DETECTOR"]
         obs_filter = f.header["FILTER"]
         obs_grating = f.header["GRATING"]
         obs_details = [obs_instrument,
                        obs_detector,
                        obs_filter,
                        obs_grating]
    return data, err, int_count, wav, dq, t, obs_details

def read_one_postproc(file):
    """Read one post-processing .nc file and return its attributes.

    Args:
        file (str): path to the .nc file you want to read out.

    Returns:
        np.array, np.array, int, np.array, np.array, np.array, np.array, np.array, np.array: the data, errors, integration count, wavelength solution, data quality array, exposure mid-times, cross/dispersion positions, and frame numbers flagged for motion.
    """
    segment = xr.open_dataset(file)
    data = segment.data.values
    err = segment.err.values
    int_count = segment.integrations
    wav = segment.wavelengths.values
    dq = segment.dq.values
    time = segment.time.values
    disp = segment.disp.values
    cdisp = segment.cdisp.values
    flagged = segment.flagged
    details = segment.details
    return data, err, int_count, wav, dq, time, disp, cdisp, flagged, details

def save_s3_output(segments, disp_pos, cdisp_pos, moved_ints, outfiles, outdir):
    """Saves an xarray for every cleaned segment in the stitched-together files.

    Args:
        segments (xarray): the xarray generated by stitch_files.
        disp_pos (list): dispersion positions. Could be an empty list.
        cdisp_pos (list): cross-dispersion positions. Could be an empty list.
        moved_ints (list): integrations flagged for movement. Could be an empty list.
        outfiles (list of str): names for each output file.
        outdir (str): directory to save the output files to.
    """
    # For every segment in the array, we need to break it up.
    int_left = 0
    int_right = 0
    for i, (ints, outfile) in enumerate(zip(segments.integrations, outfiles)):
        # Get the next limit.
        int_right += ints

        # Snip just the data that we need.
        data = segments.data.values[int_left:int_right,:,:]
        err = segments.err.values[int_left:int_right,:,:]
        dq = segments.dq.values[int_left:int_right,:,:]
        time = segments.time.values[int_left:int_right]
        wavelengths = segments.wavelengths.values[int_left:int_right,:,:]
        details = segments.details[i]

        # Plus the new tracking data, if there is any.
        disp = []
        if disp_pos:
            disp = disp_pos[int_left:int_right]
        cdisp = []
        if cdisp_pos:
            cdisp = cdisp_pos[int_left:int_right]

        # If a moved integration was in this data, report it.
        moved_int = []
        if moved_ints:
            moved_int = sorted([j for j in moved_ints if (j >= int_left and j <= int_right)])

        # Now convert to xarray.
        segment = xr.Dataset(data_vars=dict(
                                    data=(["time", "x", "y"], data),
                                    err=(["time", "x", "y"], err),
                                    dq = (["time", "x", "y"], dq),
                                    disp = (["time"], disp),
                                    cdisp = (["time"], cdisp),
                                    wavelengths = (["time", "x", "y"], wavelengths),
                                    ),
                        coords=dict(
                               time = (["time"], time),
                               ),
                        attrs=dict(
                              integrations = ints,
                              flagged = moved_int,
                              details = details,
                              )
        )

        # And save that segment as a file.
        segment.to_netcdf(os.path.join(outdir, '{}.nc'.format(outfile)))

        # Advance int_left.
        int_left = int_right

def save_s4_output(oneD_spec, oneD_err, time, wav_sols, shifts, details, outfile, outdir):
    """Saves an xarray for the extracted 1D spectra.

    Args:
        oneD_spec (np.array): extracted 1D spectra.
        oneD_err (np.array): extracted uncertainties on 1D spectra.
        time (np.array): mid-exposure times for each 1D spectrum.
        wav_sols (np.array): wavelength solutions for the 1D spectra.
        shifts (np.array): cross-correlation shfits for 1D spectra.
        details (list of list): observing details, including instrument, detector, filter, and grating.
        outfile (str): name of the output file.
        outdir (str): directory to which the .nc file will be saved to.
    """
    # First, patch for missing shifts.
    if len(shifts) == 0:
        shifts = [0 for i in time]

    # Convert to xarray.
    spectra = xr.Dataset(data_vars=dict(
                                    spectrum=(["time", "wavelength"], oneD_spec),
                                    err=(["time", "wavelength"], oneD_err),
                                    waves=(["time", "wavelength"],wav_sols),
                                    shifts=(["time"],shifts),
                                    ),
                        coords=dict(
                               time = (["time"], time),
                               ),
                        attrs=dict(
                              details = details
                              )
    )

    # And save that segment as a file.
    spectra.to_netcdf(os.path.join(outdir, '{}.nc'.format(outfile)))

def read_one_spec(file):
    """Read one 1D spectra .nc file and return its attributes.

    Args:
        file (str): path to the .nc file you want to read out.

    Returns:
        np.array, np.array, np.array, np.array, np.array, list: the spectrum,
        uncertainties, wavelength solutions, alignment shifts, times of
        mid-exposure for each spectrum, and the observing details which are
        instrument, detector, filter, and grating.
    """
    spectra = xr.open_dataset(file)
    spectrum = spectra.spectrum.values
    err = spectra.err.values
    waves = spectra.waves.values
    shifts = spectrum.shifts.values
    time = spectrum.time.values
    details = spectrum.details
    return spectrum, err, waves, shifts, time, details

def stitch_spectra(files, detector_method, time_step, verbose):
    """Reads in *1Dspec.nc files and concatenates them if needed.

    Args:
        files (list of str): paths to all *1Dspec.nc files you intend to process.
        detector_method (str): if not None, how to handle when 1D spectra from multiple detectors are found.
        time_step (bool): whether to report timing with tqdm.
        verbose (int): from 0 to 2. How much logging to do.

    Returns:
        xarray: 1D spectra xarray containing spectra.
    """
    # Log.
    if verbose >= 1:
        print("Stitching data files together for post-processing...")
    
    if verbose == 2:
        print("Will parse spectra from the following files:")
        for i, f in enumerate(files):
            print(i, f)

    # If there is just one file, we can take it as an xarray right now.
    if len(files) == 1:
        spectra = xr.open_dataset(files[0])

    else:
        # Initialize some empty lists.
        spectra, errors, waves, shifts = [], [], [], [] # the data_vars of the xarray
        time = [] # the coords of the xarray
        details = [] # the attributes of the xarray

        # Read in each file.
        for file in tqdm(files,
                        desc = 'Parsing spectral files...',
                        disable=(not time_step)):
            spectrum_i, err_i, waves_i, shifts_i, time_i, details_i = read_one_spec(file)
            spectra.append(spectrum_i)
            errors.append(err_i)
            waves.append(waves_i)
            shifts.append(shifts_i)
            time.append(time_i)
            details.append(details_i)
        
        # Now check instructions.
        if detector_method == "parallel":
            # We do not join anything. Instead, xarray needs to contain each spectrum separately.
            spectra = xr.Dataset(data_vars=dict(
                                    spectrum=(["time", "wavelength", "detector"], spectra),
                                    err=(["time", "wavelength", "detector"], errors),
                                    waves=(["time", "wavelength", "detector"],waves),
                                    shifts=(["time", "detector"],shifts),
                                    ),
                        coords=dict(
                               time = (["time", "detector"], time),
                               ),
                        attrs=dict(
                              details = details
                              )
                              )
        
        elif detector_method == "join":
            # For every timestamp, we have to concatenate each 1D spectrum together as well as the wavelength solution.
            con_spec, con_err, con_waves, con_shifts = [], [], [], []
            time = np.array(time)
            time = np.median(time,axis=0) # should collapse time to roughly the mid-exposure times for all 1D spectra being joined
            for i in range(time[0].shape):
                # At every time stamp, grab each spectrum's 1D spec.
                spec_i = spectra[0][i,:]
                err_i = errors[0][i,:]
                waves_i = waves[0][i,:]
                shifts_i = shifts[0][i,:]
                for j in range(1,len(spectra)):
                    spec_i = np.concatenate(spec_i,spectra[j][i,:])
                    err_i = np.concatenate(err_i,errors[j][i,:])
                    waves_i = np.concatenate(waves_i,waves[j][i,:])
                    shifts_i = np.concatenate(shifts_i,shifts[j][i,:])
                con_spec.append(spec_i)
                con_err.append(err_i)
                con_waves.append(waves_i)
                con_shifts.append(shifts_i)
            
            spectra = xr.Dataset(data_vars=dict(
                                    spectrum=(["time", "wavelength"], con_spec),
                                    err=(["time", "wavelength"], con_err),
                                    waves=(["time", "wavelength"],con_waves),
                                    shifts=(["time"],shifts),
                                    ),
                        coords=dict(
                               time = (["time"], time),
                               ),
                        attrs=dict(
                              details = details
                              )
                              )

    return spectra