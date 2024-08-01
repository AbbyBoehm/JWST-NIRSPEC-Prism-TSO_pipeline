import os
from tqdm import tqdm

from util.diagnostics import tqdm_translate, plot_translate
from util.loading import stitch_files
from stage3 import reject_time, reject_space, reject_flagged, reject_moved

def do_stage3(filepaths, outfiles, outdir, steps):
    """Performs Stage 3 reduction on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to correct. The files must be of type *_calints.fits.
        outfiles (list): lst of str. Names to give to the reduced files.
        outdir (str): location of where to save the reduced files to.
        steps (dict): instructions on how to run this stage of the pipeline. Contains keywords "highlevel", "reject_time", "reject_space", "reject_flagged", "reject_moved".
    """
    # Log.
    if steps["highlevel"]["verbose"] >= 1:
        print("Juniper Stage 3 has initialized.")

    if steps["highlevel"]["verbose"] == 2:
        print("Stage 3 will operate on the following files:")
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

    # Open all files and stitch them together.
    segments = stitch_files(filepaths,
                            time_step=time_step,
                            verbose=steps["highlevel"]["verbose"])
    
    # Mask data flags.
    if not steps["reject_flagged"]["skip"]:
        segments = reject_flagged.mask_flags(segments,
                                             inpt_dict=steps["reject_flagged"])
        
    # Reject outliers in time.
    if not steps["reject_time"]["skip"]:
        if steps["reject_time"]["method"] == "fixed":
            segments = reject_time.iterate_fixed(segments,
                                                 inpt_dict=steps["reject_time"])