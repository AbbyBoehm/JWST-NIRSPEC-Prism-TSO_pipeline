import os
import glob

import numpy as np
import time

import matplotlib.pyplot as plt
from scipy import signal
from astropy.io import fits

from jwst.pipeline import Spec2Pipeline

def doStage2(filesdir, outfiles, outdir,
             assign_wcs={"skip":False},
             extract_2d={"skip":False},
             srctype={"skip":False},
             wavecorr={"skip":False},
             flat_field={"skip":False},
             pathloss={"skip":True},
             photom={"skip":True},
             resample_spec={"skip":True},
             extract_1d={"skip":True}
             ):
    '''
    Performs Stage 2 calibration on a list of files.
    
    :param filesdir: str. Directory where the *_calints.fits files you want to calibrate are stored.
    :param outfiles: str. Name to give to the calibrated file.
    :param outdir: str. Location of where to save the calibrated file to.
    :param assign_wcs, background, extract2d, etc.: dict. These are the dictionaries shown in the Spec2Pipeline() documentation, which control which steps are run and what parameters they are run with. Please consult jwst-pipeline.readthedocs.io for more information on these dictionaries.
    :return: a Stage 2 calibrated file *_calints.fits saved to the outdir.
    '''
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Locate rateints files to calibrate.
    files = sorted(glob.glob(os.path.join(filesdir,'*_rateints.fits')))

    print("Performing Stage 2 JWST calibrations on the following files:")
    for file in files:
        with fits.open(file) as file:
            print(file[0].header["FILENAME"])

    print("Running JWST Stage 2 pipeline for spectroscopic data. This stage is primarily a wrapper for JWST Stage 2, with an additional step for correcting curved traces as needed. Beginning process...")
    master_t0 = time.time()
    for filepath, outfile in zip(files, outfiles):
        t0 = time.time()
        result = Spec2Pipeline.call(filepath, output_file=outfile, output_dir=outdir,
                                    steps={"assign_wcs":assign_wcs,
                                        "extract_2d":extract_2d,
                                        "srctype":srctype,
                                        "wavecorr":wavecorr,
                                        "flat_field":flat_field,
                                        "pathloss":pathloss,
                                        "photom":photom,
                                        "resample_spec":resample_spec,
                                        "extract_1d":extract_1d})
        # Check whether the file is a G395H file and thus needs aligned.
        output_file = os.path.join(outdir, outfile+"_calints.fits")
        with fits.open(output_file) as file:
            grating = file[0].header['GRATING']
            if grating in ("G395M","G395H"):
                print("{} grating detected, correcting for trace curvature...".format(grating))
                scidata = file['SCI'].data
                wavelengths = file['WAVELENGTH'].data
                shifted_scidata, shifted_wavelengths = fix_curvature(scidata, wavelengths)
                write_curve_fixed_file(output_file, shifted_scidata, shifted_wavelengths)
        print("File " + filepath + " calibrated and saved in %.3f minutes."% ((time.time() - t0)/60))
    print("Stage 2 calibrations completed in %.3f minutes." % ((time.time() - master_t0)/60))

def fix_curvature(scidata, wavelengths):
    # Takes one scidata array and, for each frame, rolls columns so that the trace lines up nicely.
    medframe = np.median(scidata,axis=0)
    medframe[np.isnan(medframe)] = 0
    rolls = get_rolls(medframe)
    plt.plot(rolls)
    plt.xlabel("column position [pixels]")
    plt.ylabel("roll [pixels]")
    plt.title("Rolls needed to correct frame 0")
    plt.ylim(-13, 13)
    plt.show()
    plt.close()

    # There is only one wavelength frame, so roll it by the median rolls in time.
    shifted_wavelengths = roll_one_frame(wavelengths, rolls)

    # Then for each frame in scidata, need to roll it.
    shifted_scidata = np.empty_like(scidata)
    for i in range(scidata.shape[0]):
        shifted_scidata[i,:,:] = roll_one_frame(scidata[i,:,:], rolls)
        if i == 0:
            plt.imshow(shifted_scidata[i,:,:])
            plt.title("Rolled frame 0")
            plt.show()
            plt.close()
    
    return shifted_scidata, shifted_wavelengths

def roll_one_frame(frame, rolls):
    retain_last_roll = 0
    for j, roll in enumerate(rolls):
        if abs(roll) > 20:
            #print("Something off with the roll of column {}, using retained roll...".format(j))
            roll = retain_last_roll
        frame[:,j] = np.roll(frame[:,j], int(roll))
        retain_last_roll = roll
    return frame

def get_rolls(frame):
    # From Eureka! S3 straighten.py code.
    # Determine the rolls needed to straighten the trace using the median frame.
    pix_centers = np.arange(frame.shape[0]) + 0.5
    COMs = signal.medfilt((np.sum(pix_centers[:,np.newaxis]*np.abs(frame),axis=0)/np.sum(np.abs(frame),axis=0)),7)
    integer_COMs = np.around(COMs - 0.5).astype(int)
    new_center = int(frame.shape[0]/2) - 1
    rolls = new_center - integer_COMs
    rolls[COMs<0] = 0
    rolls[COMs>frame.shape[0]] = 0
    rolls = signal.medfilt(rolls,41)
    return rolls

def write_curve_fixed_file(output_file, shifted_scidata, shifted_wavelengths):
    print("Writing curvature-corrected fits file...")
    with fits.open(output_file, mode="update") as fits_file:
        # Need to update data and wavelength attributes to be rotated arrays.
        fits_file['SCI'].data = shifted_scidata
        fits_file['WAVELENGTH'].data = shifted_wavelengths

        # All modified headers get written out.
        fits_file.writeto(output_file, overwrite=True)