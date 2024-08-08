import os
import glob
from tqdm import tqdm

import numpy as np
import xarray as xr

from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.datahandling import stitch_spectra, save_s5_output
from juniper.stage5 import bin_light_curves

def do_stage5(filepaths, outfile, outdir, steps, plot_dir):
    """Performs Stage 5 fitting on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to extract from. The files must be of type *_1Dspec.nc
        outfile (str): name to give to the fitted models file.
        outdir (str): location of where to save the fits to.
        steps (dict): instructions on how to run this stage of the pipeline.
        plot_dir (str): location to save diagnostic plots to.
    """
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 5 has initialized.")

    if steps["verbose"] == 2:
        print("Stage 5 will operate on the following files:")
        for i, f in enumerate(filepaths):
            print(i, f)
        print("Output will be saved to {}.".format(outfile))
    
    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(steps["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(steps["show_plots"])
    save_step, save_ints = plot_translate(steps["save_plots"])
    
    # Create the output directory if it does not yet exist.
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Put the plot directory into the inpt_dict and create it.
    steps["plot_dir"] = plot_dir
    if (not os.path.exists(plot_dir) and any((save_step, save_ints))):
        os.makedirs(plot_dir)

    # Open all files and decide how to handle them.
    if steps["read_1D"]:
        spectra = stitch_spectra(filepaths, steps["detectors"], time_step, steps["verbose"])

        # Bin light curves.
        light_curves = bin_light_curves.bin_light_curves(spectra, steps)

        # Save light curves out.
        light_curves.to_netcdf(os.path.join(outdir, 's5_lightcurves.nc'))
    
    # Read the binned light curve x array.
    xrfile = sorted(glob.glob(os.path.join(outdir,"*lightcurves.nc")))[0]
    light_curves = xr.open_dataset(xrfile)

    # Begin fitting of the light curves.
    # Save everything out.
    save_s5_output()
    
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 5 is complete.")