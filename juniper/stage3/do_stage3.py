import os
from tqdm import tqdm

from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.loading import stitch_files
from juniper.stage3 import reject_time, reject_space, reject_flagged, reject_moved, subtract_background

def do_stage3(filepaths, outfiles, outdir, steps):
    """Performs Stage 3 reduction on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to correct. The files must be of type *_calints.fits.
        outfiles (list): lst of str. Names to give to the reduced files.
        outdir (str): location of where to save the reduced files to.
        steps (dict): instructions on how to run this stage of the pipeline.
    """
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 3 has initialized.")

    if steps["verbose"] == 2:
        print("Stage 3 will operate and output to the following files:")
        for i, f in enumerate(filepaths):
            print(i, f, "->", outfiles[i])
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(steps["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(steps["show_plots"])
    save_step, save_plots = plot_translate(steps["save_plots"])
    
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Open all files and stitch them together.
    segments = stitch_files(filepaths,
                            time_step=time_step,
                            verbose=steps["verbose"])
    
    # Mask data flags.
    if steps["reject_flagged"]:
        # Build input dict.
        inpt_dict = {}
        for key in ("verbose","show_plots","save_plots","flag_replace",
                    "flag_sigma","flag_kernel"):
            inpt_dict[key] = steps[key]
        segments = reject_flagged.mask_flags(segments, inpt_dict)
        
    # Reject outliers in time.
    if steps["reject_time"]:
        if steps["time_method"] == "fixed":
            segments = reject_time.iterate_fixed(segments, inpt_dict)
        