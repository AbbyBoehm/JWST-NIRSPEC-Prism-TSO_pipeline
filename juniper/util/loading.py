import numpy as np
from astropy.io import fits

from jwst import datamodels as dm

def stitch_files(files):
    """Reads all supplied files and stitches them together into a single array.

    Args:
        files (lst of str): filepaths to files that are to be loaded.

    Returns:
        np.array: loaded data.
    """
    if "postprocessed" in files[0]:
        print("Reading post-processed files, adjusting outputs accordingly...")
        # Reading a postprocessed file, adjust strategy!
        for i, file in enumerate(files):
            print("Attempting to locate file: " + file)
            fitsfile = dm.open(file)
            print("Loaded the file " + file + " successfully.")

            # Need to retrieve the wavelength object, science object, and DQ object from this.
            if i == 0:
                segments = fitsfile.data
                errors = fitsfile.err
                segstarts = [np.shape(fitsfile.data)[0]]
                wavelengths = [fitsfile.wavelength]
                dqflags = fitsfile.dq
                times = fitsfile.int_times["int_mid_MJD_UTC"]
            else:
                segments = np.concatenate([segments, fitsfile.data], 0)
                errors = np.concatenate([errors, fitsfile.err], 0)
                segstarts.append(np.shape(fitsfile.data)[0] + sum(segstarts))
                wavelengths.append(fitsfile.wavelength)
                dqflags = np.concatenate([dqflags, fitsfile.dq], 0)
                times = np.concatenate([times, fitsfile.int_times["int_mid_MJD_UTC"]], 0)
            
            fitsfile.close()
            
            with fits.open(file) as f:
                frames_to_reject = f["REJECT"].data
            
            print("Retrieved segments, wavelengths, DQ flags, and times from file: " + file)
            print("Closing file and moving on to next one...")
        return segments, errors, segstarts, wavelengths, dqflags, times, frames_to_reject
    else:
        for i, file in enumerate(files):
            print("Attempting to locate file: " + file)
            fitsfile = dm.open(file)
            print("Loaded the file " + file + " successfully.")

            # Need to retrieve the wavelength object, science object, and DQ object from this.
            if i == 0:
                segments = fitsfile.data
                errors = fitsfile.err
                segstarts = [np.shape(fitsfile.data)[0]]
                wavelengths = [fitsfile.wavelength]
                dqflags = fitsfile.dq
                times = fitsfile.int_times["int_mid_MJD_UTC"]
            else:
                segments = np.concatenate([segments, fitsfile.data], 0)
                errors = np.concatenate([errors, fitsfile.err], 0)
                segstarts.append(np.shape(fitsfile.data)[0] + sum(segstarts))
                wavelengths.append(fitsfile.wavelength)
                dqflags = np.concatenate([dqflags, fitsfile.dq], 0)
                times = np.concatenate([times, fitsfile.int_times["int_mid_MJD_UTC"]], 0)

            print("Retrieved segments, wavelengths, DQ flags, and times from file: " + file)
            print("Closing file and moving on to next one...")
            fitsfile.close()
        return segments, errors, segstarts, wavelengths, dqflags, times