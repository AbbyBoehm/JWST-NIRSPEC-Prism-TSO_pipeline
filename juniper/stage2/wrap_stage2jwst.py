import time

from jwst.pipeline import Spec2Pipeline
from util.diagnostics import timer

def wrap(filepath,outfile,outdir,steps):
    """Wrapper for jwst Spec2Pipeline.

    Args:
        filepath (str): a path to the *rateints.fits file you want to operate on.
        outfile (str): what to rename the output files to. Can be None to keep the default name.
        outdir (str): where to save the output *calints.fits files.
        steps (dict): A dictionary containing instructions for all stages of the Spec2Pipeline.
    """
    # Time this step if asked.
    if steps["timer"]:
        t0 = time.time()

    # Process Spec2Pipeline.
    result = Spec2Pipeline.call(filepath, output_file=outfile, output_dir=outdir,
                                steps=steps)
    
    if steps["timer"]:
        timer(time.time()-t0,None,None,None)