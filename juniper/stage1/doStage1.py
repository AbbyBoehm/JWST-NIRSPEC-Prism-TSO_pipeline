import os

import time

from revised_utils import timer

from juniper.stage1 import group_level_bckg_sub, wrap_stage1jwst

def doStage1(filepaths, outfiles, outdir, steps):
    '''
    Performs Stage 1 calibration on a list of files.
    
    :param filepaths: lst of str. Location of the files you want to correct. The files must be of type *_uncal.fits.
    :param outfiles: lst of str. Names to give to the calibrated files.
    :param outdir: str. Location of where to save the calibrated files to.
    :param steps: dict. Instructions on how to run this stage of the pipeline.
    :return: Stage 1 calibrated files *_rateints.fits saved to the outdir.
    '''
    # Time this stage, if asked.
    if steps["timer"]:
        t0 = time.time()
    
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Report initialization of Stage 1.
    print("Performing Stage 1 JWST/Juniper calibrations on files: ", filepaths)

    # Start iterating.
    for filepath, outfile in zip(filepaths,outfiles):
        datamodel = wrap_stage1jwst.wrap_front_end(filepath, steps)
        if not steps["glbs"]["skip"]:
            datamodel = group_level_bckg_sub.do(datamodel, steps["glbs"]["dict"])
        wrap_stage1jwst.wrap_back_end(datamodel, steps, outfile, outdir)
    
    # Report overall runtime, if asked.
    if steps["timer"]:
        timer(time.time()-t0,None,None,None)
    print("Stage 1 JWST/Juniper calibrations of all supplied files complete.")
    return None