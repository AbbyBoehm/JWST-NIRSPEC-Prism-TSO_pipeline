import os
from tqdm import tqdm

from util.diagnostics import tqdm_translate, plot_translate
from stage1 import group_level_bckg_sub, wrap_stage1jwst, NSClean

def do_stage1(filepaths, outfiles, outdir, steps):
    """Performs Stage 1 calibration on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to correct. The files must be of type *_uncal.fits.
        outfiles (list): lst of str. Names to give to the calibrated files.
        outdir (str): location of where to save the calibrated files to.
        steps (dict): instructions on how to run this stage of the pipeline. Contains keywords "highlevel", "pipeline", "glbs", "NSClean".
    """
    # Log.
    if steps["highlevel"]["verbose"] >= 1:
        print("Juniper Stage 1 has initialized.")

    if steps["highlevel"]["verbose"] == 2:
        print("Stage 1 will operate on the following files:")
        for i, f in enumerate(filepaths):
            print(i, f)
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(steps["highlevel"]["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(steps["highlevel"]["show_plots"])
    save_step, save_plots = plot_translate(steps["highlevel"]["save_plots"])
    
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Start iterating.
    for filepath, outfile in tqdm(zip(filepaths, outfiles),
                                  desc='Processing Stage 1...',
                                  disable=(not time_step)):
        # Wrap the first steps of Detector1Pipeline.
        datamodel = wrap_stage1jwst.wrap_front_end(filepath, steps["pipeline"])

        # Perform group-level background subtraction.
        if not steps["glbs"]["skip"]:
            datamodel = group_level_bckg_sub.do(datamodel, steps["glbs"])

        # Wrap the last steps of Detector1Pipeline.
        wrap_stage1jwst.wrap_back_end(datamodel, steps["pipeline"], outfile, outdir)

        # Perform NSClean background subtraction.
        if not steps["NSClean"]["skip"]:
            NSClean(steps["NSClean"])
        
        if steps["highlevel"]["verbose"] == 2:
            print("One iteration complete. Output saved in", outdir, "as file name {}_rateints.fits".format(outfile))
    
    # Log.
    if steps["highlevel"]["verbose"] >= 1:
        print("Juniper Stage 1 is complete.")