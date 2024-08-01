import os
from tqdm import tqdm

from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.stage2 import wrap_stage2jwst, correct_curvature

def do_stage2(filepaths, outfiles, outdir, steps):
    """Performs Stage 2 calibration on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to correct. The files must be of type *_rateints.fits.
        outfiles (list): lst of str. Names to give to the calibrated files.
        outdir (str): location of where to save the calibrated files to.
        steps (dict): instructions on how to run this stage of the pipeline. Loaded from the Stage 2 .berry files.
    """
    # Log.
    if steps["highlevel"]["verbose"] >= 1:
        print("Juniper Stage 2 has initialized.")

    if steps["highlevel"]["verbose"] == 2:
        print("Stage 2 will operate on the following files:")
        for i in filepaths:
            print(i)
    
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
                                  desc='Processing Stage 2...',
                                  disable=(not time_step)):
        # Process Spec2Pipeline.
        wrap_stage2jwst.wrap(filepath, steps["pipeline"])

        # Then curve-correct, if necessary.
        if not steps["curve"]["skip"]:
            correct_curvature.correct_curvature(outfile, outdir, steps["curve"])
        
        if steps["highlevel"]["verbose"] == 2:
            print("One iteration complete. Output saved in", outdir, "as file name {}_calints.fits".format(outfile))
    
    # Log.
    if steps["highlevel"]["verbose"] >= 1:
        print("Juniper Stage 2 is complete.")