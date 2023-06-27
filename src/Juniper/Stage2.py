import os

import time

from jwst.pipeline import Spec2Pipeline

def doStage2(filepath, outfile, outdir,
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
    Performs Stage 2 calibration on one file.
    
    :param filepath: str. Location of the file you want to correct. The file must be of type *_rateints.fits.
    :param outfile: str. Name to give to the calibrated file.
    :param outdir: str. Location of where to save the calibrated file to.
    :param assign_wcs, background, extract2d, etc.: dict. These are the dictionaries shown in the Spec2Pipeline() documentation, which control which steps are run and what parameters they are run with. Please consult jwst-pipeline.readthedocs.io for more information on these dictionaries.
    :return: a Stage 2 calibrated file *_calints.fits saved to the outdir.
    '''
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    print("Performing Stage 2 calibration on file: " + filepath)
    print("Running JWST Stage 2 pipeline for spectroscopic data. This stage is a pure wrapper for JWST Stage 2 with no mods. Anyways...")
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
    print("File calibrated and saved.")
    print("Stage 2 calibrations completed in %.3f minutes." % ((time.time() - t0)/60))