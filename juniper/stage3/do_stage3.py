import os
from tqdm import tqdm
import numpy as np

from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.datahandling import stitch_files, save_s3_output
from juniper.stage3 import reject_time, reject_space, reject_flagged, track_motion, subtract_background

def do_stage3(filepaths, outfiles, outdir, steps, plot_dir):
    """Performs Stage 3 reduction on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to correct.
        The files must be of type *_calints.fits.
        outfiles (list): lst of str. Names to give to the reduced files.
        outdir (str): location of where to save the reduced files to.
        steps (dict): instructions on how to run this stage of the pipeline.
        plot_dir (str): location to save diagnostic plots to.
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

    # Add the plot_dir to the stes.
    steps["diagnostic_plots"] = plot_dir

    # Open all files and stitch them together.
    segments = stitch_files(filepaths,
                            time_step=time_step,
                            verbose=steps["verbose"])
    
    # Mask data flags.
    if steps["reject_flagged"]:
        if steps["verbose"] >= 1:
            print("JWST flags will be used for this run. Output data quality array will include JWST flag information.")
        segments = reject_flagged.mask_flags(segments, steps)

    # Alternatively, if JWST flags are not of interest, replace them.
    if not steps["reject_flagged"]:
        if steps["verbose"] >= 1:
            print("JWST flags will be ignored for this run. Output data quality array will exclude JWST flag information.")
        segments.dq.values = np.zeros_like(segments.dq.values)
        
    # Reject outliers in time.
    if steps["reject_time"]:
        if steps["time_method"] == "fixed":
            segments = reject_time.iterate_fixed(segments, steps)

        if steps["time_method"] == "free":
            segments = reject_time.iterate_free(segments, steps)
        
    # Reject outliers in space.
    if steps["reject_space"]:
        if steps["space_method"] == "led":
            segments = reject_space.led(segments, steps)
        
        if steps["space_method"] == "smooth":
            segments = reject_space.smooth(segments, steps)

    # Remove background signal.
    if steps["subtract_bckg"]:
        segments = subtract_background.subtract_background(segments, steps)

    # Track motion of the trace.
    disp_pos, cdisp_pos, moved_ints = [],[],[]
    if any((steps["track_disp"],steps["track_spatial"])):
        segments, disp_pos, cdisp_pos, moved_ints = track_motion.track_pos(segments, steps)

    # Save everything out.
    save_s3_output(segments, disp_pos, cdisp_pos, moved_ints, outfiles, outdir)

    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 3 is complete.")