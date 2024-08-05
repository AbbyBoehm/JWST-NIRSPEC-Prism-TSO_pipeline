import os
from tqdm import tqdm
import numpy as np

from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.datahandling import stitch_files, save_s4_output
from juniper.stage4 import extract_1D, align_spec, clean_spec

def do_stage4(filepaths, outfile, outdir, steps, plot_dir):
    """Performs Stage 4 extraction on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to extract from. The files must be of type *_reduced.nc
        outfile (str): name to give to the extracted spectra file.
        outdir (str): location of where to save the spectra file to.
        steps (dict): instructions on how to run this stage of the pipeline.
        plot_dir (str): location to save diagnostic plots to.
    """
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 4 has initialized.")

    if steps["verbose"] == 2:
        print("Stage 4 will operate on the following files:")
        for i, f in enumerate(filepaths):
            print(i, f)
        print("Output will be saved to {}.".format(outfile))
    
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
    
    # Extract 1D spectra.
    if steps["extract_method"] == 'box':
        oneD_spec, oneD_err, wav_sols = extract_1D.box(segments, steps)
    
    elif steps["extract_method"] == 'optimum':
        oneD_spec, oneD_err, wav_sols = extract_1D.optimum(segments, steps)

    # Align spectra.
    if steps["align"]:
        oneD_spec, oneD_err, shifts = align_spec.align(oneD_spec, oneD_err,
                                                       wav_sols, steps)
        
    # Clean spectra.
    if steps["sigmas"]:
        oneD_spec = clean_spec.clean(oneD_spec, oneD_err,
                                     wav_sols, steps)

    # Save everything out.
    save_s4_output(oneD_spec, oneD_err, segments.time.values, wav_sols, shifts, outfile, outdir)

    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 4 is complete.")