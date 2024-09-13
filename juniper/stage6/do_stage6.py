import os
import glob
from tqdm import tqdm

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from juniper.config.translate_config import make_planets, make_flares, make_systematics, make_LD
from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.datahandling import stitch_spectra, save_s5_output
from juniper.stage6 import plot_fit_and_res, plot_model_panel, plot_spectrum, compute_depths

def do_stage6(filepaths, outfile, outdir, steps, plot_dir):
    """Performs Stage 6 results processing on the given files.

    Args:
        filepaths (list): list of str. Location of the files you want to extract from. The files must be of type *.npy
        outfile (str): name to give to the spectra files.
        outdir (str): location of where to save the files to.
        steps (dict): instructions on how to run this stage of the pipeline.
        plot_dir (str): location to save diagnostic plots to.
    """
    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 6 has initialized.")

    if steps["verbose"] == 2:
        print("Stage 6 will operate on the following files:")
        for i, f in enumerate(filepaths):
            print(i, f)
        print("Output will be saved to {}*.txt, *.dat, and *.png formats.".format(outfile))
    
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

    # Decide how to handle LSQ and MCMC results.
    results = {}
    if steps["read_LSQ"]:
        # Load each lsq result.
        for f in [filepath for filepath in filepaths if "LSQ" in filepath]:
            result = np.load(f,allow_pickle=True).item()
            key = str.split(f,sep='/')[-1]
            key = str.replace(key,'.npy','')
            results[key] = result

    if steps["read_MCMC"]:
        # Load each mcmc result.
        for f in [filepath for filepath in filepaths if "MCMC" in filepath]:
            result = np.load(f,allow_pickle=True).item()
            key = str.split(f,sep='/')[-1]
            key = str.replace(key,'.npy','')
            results[key] = result
    
    # With the fits loaded, we can start making plots. We start with fits and residuals.
    if steps["plot_individual"]:
        # We want to plot each light curve in its own plot.
        for key in tqdm(list(results.keys()),
                        desc='Processing fit-res for each fit...',
                        disable=(not time_ints)):
            tag = 'MCMC'
            if 'LSQ' in key:
                tag = 'LSQ'
            # Every result has keys planets, planet_errs,
            # flares, flare_errs, systematics, systematic_errs,
            # LD, LD_err, time, light_curve, errors, wavelength.
            result = results[key]

            try:
                if 'broadband' in key:
                    time = result['time'][0]
                    light_curve = result['light_curve'][0]
                    errors = result['errors'][0]
                else:
                    time = result['time']
                    light_curve = result['light_curve']
                    errors = result['errors']
                
                fig, ax = plot_fit_and_res.plot_fit_and_res(time, light_curve, errors,
                                                            result['planets'],result['flares'],
                                                            result['systematics'],result['LD'],steps)
                
                if (result['wavelength'] == 'broadband' and save_step):
                    plt.savefig(os.path.join(plot_dir,'s5_{}_broadband_fit-res{}.png'.format(outfile,tag)),
                                dpi=300,bbox_inches='tight')
                elif save_ints:
                    plt.savefig(os.path.join(plot_dir,'s5_{}_{}_fit-res{}.png'.format(outfile,result['wavelength'],tag)),
                                dpi=300,bbox_inches='tight')
                if (result['wavelength'] == 'broadband' and plot_step):
                    plt.show(block=True)
                elif plot_ints:
                    plt.show(block=True)
                plt.close()
            except KeyError:
                print("Fault with result {}, passing...".format(key))

    # We can plot panels of the models, too.
    if steps["plot_components"]:
        # We want to plot each light curve's components in panels.
        for key in tqdm(list(results.keys()),
                        desc='Processing model components for each fit...',
                        disable=(not time_ints)):
            tag = 'MCMC'
            if 'LSQ' in key:
                tag = 'LSQ'
            # Every result has keys planets, planet_errs,
            # flares, flare_errs, systematics, systematic_errs,
            # LD, LD_err, time, light_curve, errors, wavelength.
            result = results[key]

            if 'broadband' in key:
                time = result['time'][0]
                light_curve = result['light_curve'][0]
                errors = result['errors'][0]
            else:
                time = result['time']
                light_curve = result['light_curve']
                errors = result['errors']

            fig, axes = plot_model_panel.plot_model_panel(time, light_curve, errors,
                                                          result['planets'],result['flares'],
                                                          result['systematics'],result['LD'],steps)
            
            if (result['wavelength'] == 'broadband' and save_step):
                plt.savefig(os.path.join(plot_dir,'s5_{}_broadband_fit-comps{}.png'.format(outfile,tag)),
                            dpi=300,bbox_inches='tight')
            elif save_ints:
                plt.savefig(os.path.join(plot_dir,'s5_{}_{}_fit-comps{}.png'.format(outfile,result['wavelength'],tag)),
                            dpi=300,bbox_inches='tight')
            if (result['wavelength'] == 'broadband' and plot_step):
                plt.show(block=True)
            elif plot_ints:
                plt.show(block=True)
            plt.close()

    # We can plot a waterfall of the fits and residuals.
    if steps["plot_waterfall"]:
        for tag in ('LSQ','MCMC'):
            result_keys = [key for key in list(results.keys()) if tag in key]

            wavelengths, ts, lcs, lc_errs = [], [], [], []
            t_interps, lc_interps, residualses = [], [], []
            for key in tqdm(list(results.keys()),
                            desc='Processing fit-res for waterfall plot...',
                            disable=(not time_ints)):
                result = results[key]
                if result['wavelength'] != 'broadband':
                    wavelengths.append(result['wavelength'])
                    ts.append(result['time'])
                    lcs.append(result['light_curve'])
                    lc_errs.append(result['errors'])
                    t_interp, lc_interp, comps, residuals = plot_fit_and_res.get_fit_and_res(result['time'],result['light_curve'],result['errors'],
                                                                                             result['planets'], result['flares'], result['systematics'], result['LD'],
                                                                                             steps)
                    t_interps.append(t_interp)
                    lc_interps.append(lc_interp)
                    residualses.append(residuals)
            fig, axes = plot_fit_and_res.plot_waterfall(wavelengths, ts, lcs, lc_errs, t_interps, lc_interps, residualses, steps)
            if save_step:
                plt.savefig(os.path.join(plot_dir,'s5_{}_waterfall{}.png'.format(outfile,tag)),
                            dpi=300,bbox_inches='tight')
            if plot_step:
                plt.show(block=True)
            plt.close()

    # Now we should compute the spectrum and save it.
    if steps["get_spectrum"]:
        for tag in ('LSQ','MCMC'):
            # We don't know in advance how many planets were fit,
            # so we're going to have to keep trying until we run
            # out of planets.
            planet_ID = 1
            keyError_happened = False
            while not keyError_happened:
                try:
                    # Get just the keys for the type of fit we are probing.
                    result_keys = [key for key in list(results.keys()) if tag in key]

                    # Open some lists for this spectrum.
                    waves = []
                    depths = []
                    errors = []
                    for key in [key for key in result_keys if results[key]['wavelength'] != 'broadband']:
                        # Get the result's planets.
                        planets = results[key]['planets']
                        planet_errs = results[key]['planet_errs']
                        waves.append(float(results[key]['wavelength']))

                        # Get the spectrum for the current planet of interest.
                        if steps["spectrum_type"] == 'rprs':
                            depth, err = compute_depths.compute_depth_rprs(planets['planet{}'.format(planet_ID)],
                                                                           planet_errs['planet{}'.format(planet_ID)],
                                                                           str(planet_ID))
                        if steps["spectrum_type"] == 'rprs2':
                            depth, err = compute_depths.compute_depth_rprs2(planets['planet{}'.format(planet_ID)],
                                                                           planet_errs['planet{}'.format(planet_ID)],
                                                                           str(planet_ID))
                        if steps["spectrum_type"] == 'aover':
                            depth, err = compute_depths.compute_depth_aoverlap(planets['planet{}'.format(planet_ID)],
                                                                           planet_errs['planet{}'.format(planet_ID)],
                                                                           str(planet_ID))
                        if steps["spectrum_type"] == 'fpfs':
                            depth, err = compute_depths.compute_depth_fpfs(planets['planet{}'.format(planet_ID)],
                                                                           planet_errs['planet{}'.format(planet_ID)],
                                                                           str(planet_ID))

                        depths.append(depth)
                        errors.append(err)
                    
                    # Plot, if asked.
                    if (plot_step or save_step):
                        bin_factors = steps["bin_factors"]
                        wave_bounds = steps["wave_bounds"]
                        if not steps["bin_factors"]:
                            bin_factors = [1,]
                        if 1 not in bin_factors:
                            bin_factors.append(1) # ensure there is always native res
                        for bin_f in bin_factors:
                            fig, ax = plot_spectrum.plot_spectrum(waves,depths,errors,
                                                                bin_f,wave_bounds,steps["spectrum_type"])
                            if save_step:
                                plt.savefig(os.path.join(plot_dir,'s5_{}_planet{}_spectrum{}_fit{}_bin{}.png'.format(outfile,
                                                                                                                     planet_ID,
                                                                                                                     steps["spectrum_type"],
                                                                                                                     tag,
                                                                                                                     bin_f)),
                                            dpi=300,bbox_inches='tight')
                            if plot_step:
                                plt.show(block=True)
                            plt.close()
                    
                    # And save.
                    fname = os.path.join(outdir,'s5_{}_planet{}_spectrum{}_fit{}.dat'.format(outfile,
                                                                                             planet_ID,
                                                                                             steps["spectrum_type"],
                                                                                             tag))
                    with open(fname,mode='w') as f:
                        f.write("#wavelength[um] depth[{}] err[{}]\n".format(steps["spectrum_type"],steps["spectrum_type"]))
                        for w,d,e in zip(waves,depths,errors):
                            f.write("{}   {}   {}\n".format(w,d,e))
                    
                    # Advance to next planet!
                    planet_ID += 1
                except KeyError:
                    keyError_happened = True

    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 6 is complete.")