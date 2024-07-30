import time

from jwst.pipeline import Spec2Pipeline

from revised_utils import timer

def wrap(filepath,outfile,outdir,steps):
    '''
    Wrapper for jwst Spec2Pipeline. Does nothing else.

    :param filepath: str. A path to the *rateints.fits file you want to operate on.
    :param steps: dict. A dictionary containing instructions for all stages of the Spec2Pipeline.
    :return: outfile saved to outdir. Routine returns no callables.
    '''
    # Time this step if asked.
    if steps["timer"]:
        t0 = time.time()

    # Process Spec2Pipeline.
    result = Spec2Pipeline.call(filepath, output_file=outfile, output_dir=outdir,
                                steps=steps)
    
    if steps["timer"]:
        timer(time.time()-t0,None,None,None)
    return None