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
    data, err, dq, cdisp, disp, cwidth, wav = [], [], [], [], [], [], [] # the data_vars of the xarray
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
            disp_i, cdisp_i, cwidth_i, flagged_i = np.zeros_like(time_i), np.zeros_like(time_i), np.zeros_like(time_i), np.zeros_like(time_i)
        elif ".nc" in file:
            # It's Stage 3 output.
            data_i, err_i, int_count_i, wav_i, dq_i, time_i, disp_i, cdisp_i, cwidth_i, flagged_i, details_i = read_one_postproc(file)
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
            cwidth.append(cwidth_i[i])
            wav.append(wav_i[i])

    # Now convert to xarray.
    segments = xr.Dataset(data_vars=dict(
                                    data = (["time", "x", "y"], data),
                                    err = (["time", "x", "y"], err),
                                    dq = (["time", "x", "y"], dq),
                                    disp = (["time"], disp),
                                    cdisp = (["time"], cdisp),
                                    cwidth = (["time"], cwidth),
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
    with fits.open(file) as f:
         obs_instrument = f[0].header["INSTRUME"]
         obs_detector = f[0].header["DETECTOR"]
         obs_filter = f[0].header["FILTER"]
         obs_grating = f[0].header["GRATING"]
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
        np.array, np.array, int, np.array, np.array, np.array, np.array, np.array,
        np.array, np.array: the data, errors, integration count, wavelength solution,
        data quality array, exposure mid-times, cross/dispersion positions and widths,
        and frame numbers flagged for motion.
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
    cwidth = segment.cwidth.values
    flagged = segment.flagged
    details = segment.details.values #[segment.instrument,segment.detector,segment.filter,segment.grating]
    return data, err, int_count, wav, dq, time, disp, cdisp, cwidth, flagged, details

def save_s3_output(segments, disp_pos, cdisp_pos, cdisp_widths, moved_ints, outfiles, outdir):
    """Saves an xarray for every cleaned segment in the stitched-together files.

    Args:
        segments (xarray): the xarray generated by stitch_files.
        disp_pos (list): dispersion positions. Could be an empty list.
        cdisp_pos (list): cross-dispersion positions. Could be an empty list.
        cdisp_widths (list): cross-dispersion widths. Could be an empty list.
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
        #instrument, detector, filter, grating = segments.details[i]
        seg_details = segments.details[i]

        # Plus the new tracking data, if there is any.
        disp = []
        if disp_pos:
            disp = disp_pos[int_left:int_right]
        cdisp = []
        if cdisp_pos:
            cdisp = cdisp_pos[int_left:int_right]
        cwidth = []
        if cdisp_widths:
            cwidth = cdisp_widths[int_left:int_right]

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
                                    cwidth = (["time"], cwidth),
                                    wavelengths = (["time", "x", "y"], wavelengths),
                                    ),
                        coords=dict(
                               time = (["time"], time),
                               details = (["observation_mode"], seg_details),
                               ),
                        attrs=dict(
                              integrations = ints,
                              flagged = moved_int,
                              )
        )

        # And save that segment as a file.
        segment.to_netcdf(os.path.join(outdir, '{}.nc'.format(outfile)))

        # Advance int_left.
        int_left = int_right

def save_s4_output(oneD_spec, oneD_err, time, wav_sols, shifts,
                   xpos, ypos, widths, details, outfile, outdir):
    """Saves an xarray for the extracted 1D spectra.

    Args:
        oneD_spec (np.array): extracted 1D spectra.
        oneD_err (np.array): extracted uncertainties on 1D spectra.
        time (np.array): mid-exposure times for each 1D spectrum.
        wav_sols (np.array): wavelength solutions for the 1D spectra.
        shifts (np.array): cross-correlation shfits for 1D spectra.
        xpos (np.array): dispersion positions for trace.
        ypos (np.array): cross-dispersion positions for trace.
        widths (np.array): cross-dispersion widths for trace.
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
                                    xpos=(["time"],xpos),
                                    ypos=(["time"],ypos),
                                    widths=(["time"],widths),
                                    ),
                        coords=dict(
                               time = (["time"], time),
                               details = (["observation_mode"], details),
                               ),
                        attrs=dict(
                              )
    )

    # And save that segment as a file.
    spectra.to_netcdf(os.path.join(outdir, '{}.nc'.format(outfile)))

def read_one_spec(file):
    """Read one 1D spectra .nc file and return its attributes.

    Args:
        file (str): path to the .nc file you want to read out.

    Returns:
        np.array, np.array, np.array, np.array, np.array, np.array, np.array,
        np.array, list: the spectrum, uncertainties, wavelength solutions,
        alignment shifts, dispersion/cross-dispersion positions and widths,
        times of mid-exposure for each spectrum, and the observing details
        which are instrument, detector, filter, and grating.
    """
    spectra = xr.open_dataset(file)
    spectrum = spectra.spectrum.values
    err = spectra.err.values
    waves = spectra.waves.values
    shifts = spectra.shifts.values
    xpos = spectra.xpos.values
    ypos = spectra.ypos.values
    widths = spectra.widths.values
    time = spectra.time.values
    details = spectra.details.values #[spectra.instrument,spectra.detector,spectra.filter,spectra.grating]
    return spectrum, err, waves, shifts, xpos, ypos, widths, time, details

def stitch_spectra(files, detector_method, time_step, verbose):
    """Reads in *1Dspec.nc files and concatenates them if needed.

    Args:
        files (list of str): paths to all *1Dspec.nc files you intend to process.
        detector_method (str): if not None, how to handle when 1D spectra from
        multiple detectors are found.
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

    # If there is just one file, we can take it as an xarray and adjust it to have the detector dim.
    if len(files) == 1:
        print("Reading out one 1D spectrum...")
        spectrum, err, waves, shifts, xpos, ypos, widths, time, details = read_one_spec(files[0])
        spectra = xr.Dataset(data_vars=dict(
                                    spectrum=(["detector", "time", "wavelength"], [spectrum,]),
                                    err=(["detector", "time", "wavelength"], [err,]),
                                    waves=(["detector", "time", "wavelength"],[waves,]),
                                    shifts=(["detector", "time"],[shifts,]),
                                    xpos=(["detector", "time"],[xpos,]),
                                    ypos=(["detector", "time"],[ypos,]),
                                    widths=(["detector", "time"],[widths,]),
                                    ),
                        coords=dict(
                               time = (["detector","time"], [time,]),
                               detector = (["detector"], [0,]),
                               details = (["detector","observation_mode"], [details,]), # this has the form [[INSTRUMENT, DETECTOR, FILTER, GRATING]]
                               ),
                        attrs=dict(
                              )
                              )

    else:
        # Initialize some empty lists.
        spectra, errors, waves, shifts, xpos, ypos, widths = [], [], [], [], [], [], [] # the data_vars of the xarray
        time = [] # the coords of the xarray
        details = [] # the attributes of the xarray

        # Read in each file.
        for file in tqdm(files,
                        desc = 'Parsing spectral files...',
                        disable=(not time_step)):
            spectrum_i, err_i, waves_i, shifts_i, xpos_i, ypos_i, widths_i, time_i, details_i = read_one_spec(file)
            spectra.append(spectrum_i)
            errors.append(err_i)
            waves.append(waves_i)
            shifts.append(shifts_i)
            xpos.append(xpos_i)
            ypos.append(ypos_i)
            widths.append(widths_i)
            time.append(time_i)
            details.append(details_i)
        
        # Now check instructions.
        if detector_method == "parallel":
            # We do not join anything. Instead, xarray needs to contain each spectrum separately.
            print("Parallelising multiple spectra...")
            spectra = xr.Dataset(data_vars=dict(
                                    spectrum=(["detector", "time", "wavelength"], spectra),
                                    err=(["detector", "time", "wavelength"], errors),
                                    waves=(["detector", "time", "wavelength"],waves),
                                    shifts=(["detector", "time"],shifts),
                                    xpos=(["detector", "time"],xpos),
                                    ypos=(["detector", "time"],ypos),
                                    widths=(["detector", "time"],widths),
                                    ),
                        coords=dict(
                               time = (["detector","time"], time),
                               detectors = (["detector"], [i for i in range(len(files))]),
                               details = (["detector","observation_mode"], details), # this has the form Ndetectors x [[INSTRUMENT, DETECTOR, FILTER, GRATING]]
                               ),
                        attrs=dict(
                              )
                              )
        
        elif detector_method == "join":
            # Check which detectors you are trying to join and warn the user about heinous combos.
            print("Joining multiple spectra...")
            gratings = [x[-1] for x in details]
            if any([gratings[0] != grating for grating in gratings]): # if any grating shows up that does not match the first one
                if verbose >= 1:
                    print("Warning: I noticed you are trying to stitch together files that use different gratings.")
                    print("While I commend your bravery, please note that the ''join'' method of combining files")
                    print("was intended only for single gratings which span multiple detectors (e.g. G395H)")
                    print("and the correct method for treating multiple gratings is ''parallel''.")
                    print("If you are not using limb darkening models like ExoTiC-LD, this should not crash the code.")
                    print("(It will still cause some creative and surprising behavior though.)")
                    print("If you are using limb darkening models, please relaunch Stage 5 with the ''detectors'' keyword set to ''parallel''.")
            
            # For every timestamp, we have to concatenate each 1D spectrum together as well as the wavelength solution.
            con_spec, con_err, con_waves = [], [], []
            time = np.array(time)
            time = np.median(time,axis=0) # should collapse time to roughly the mid-exposure times for all 1D spectra being joined
            shifts = np.array(shifts)
            shifts = np.median(shifts,axis=0) # should be approximately the same since the detectors are parallel 

            # FIX: these should not be the same but i'll figure it out later.
            xpos = np.array(xpos)
            xpos = np.median(xpos,axis=0) # should be approximately the same since the detectors are parallel 
            ypos = np.array(ypos)
            ypos = np.median(ypos,axis=0) # should be approximately the same since the detectors are parallel 
            widths = np.array(widths)
            widths = np.median(widths,axis=0) # should be approximately the same since the detectors are parallel 

            for i in range(time.shape[0]):
                # At every time stamp, grab each spectrum's 1D spec.
                spec_i = spectra[0][i,:]
                err_i = errors[0][i,:]
                waves_i = waves[0][i,:]
                for j in range(1,len(spectra)):
                    spec_i = np.concatenate((spec_i,spectra[j][i,:]))
                    err_i = np.concatenate((err_i,errors[j][i,:]))
                    waves_i = np.concatenate((waves_i,waves[j][i,:]))
                con_spec.append(spec_i)
                con_err.append(err_i)
                con_waves.append(waves_i)
            
            spectra = xr.Dataset(data_vars=dict(
                                    spectrum=(["detector", "time", "wavelength"], [con_spec,]),
                                    err=(["detector", "time", "wavelength"], [con_err,]),
                                    waves=(["detector", "time", "wavelength"],[con_waves,]),
                                    shifts=(["detector", "time"],[shifts,]),
                                    xpos=(["detector", "time"],[xpos,]),
                                    ypos=(["detector", "time"],[ypos,]),
                                    widths=(["detector", "time"],[widths,]),
                                    ),
                        coords=dict(
                               time = (["detector", "time"], [time,]),
                               detectors = (["detector",], [0,]), # there is now just one "detector" which is the joined dataset
                               details = (["detector","observation_mode"], [details[0],]), # this has the form [[INSTRUMENT, DETECTOR, FILTER, GRATING]]
                               ),
                        attrs=dict(
                              )
                              )

    return spectra

def read_one_lc(file):
    """Read one light curve .nc file and return its attributes.

    Args:
        file (str): path to the .nc file you want to read out.

    Returns:
        np.array, np.array, np.array, np.array, np.array, list: the spectrum,
        uncertainties, wavelength solutions, alignment shifts, times of
        mid-exposure for each spectrum, and the observing details which are
        instrument, detector, filter, and grating.
    """
    curves = xr.open_dataset(file)
    
    return curves

def save_s5_output(planets, planets_err, flares, flares_err,
                   systematics, systematics_err, LD, LD_err,
                   time, light_curve, outfile, outdir):
    """Writes out the results of a fit to a pickle file.

    Args:
        planets (dict): every fitted planet.
        planets_err (dict): every fitted planet's uncertainties.
        flares (dict): every fitted flare.
        flares_err (dict): every fitted flare's uncertainties.
        systematics (dict): fitted systematics models.
        systematics_err (dict): uncertainties on systematics models.
        LD (dict): fitted limb darkening model.
        LD_err (dict): uncertainties on limb darkening model.
        time (np.array): timestamps for each flux point.
        light_curve (np.array): flux at each point in time.
        outfile (str): name to give the saved file.
        outdir (str): where to save the file to.
    """
    # Dict everything together.
    output = {"planets":planets,
              "planet_errs":planets_err,
              "flares":flares,
              "flare_errs":flares_err,
              "systematics":systematics,
              "systematic_errs":systematics_err,
              "LD":LD,
              "LD_err":LD_err,
              "time":time,
              "light_curve":light_curve}
    
    filename = (os.path.join(outdir, '{}.npy'.format(outfile)))
    np.save(filename,output)