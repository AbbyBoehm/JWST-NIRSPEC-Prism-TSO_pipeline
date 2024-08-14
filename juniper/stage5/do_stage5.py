import os
import glob
from tqdm import tqdm

import numpy as np
import xarray as xr

from juniper.config.translate_config import make_planets, make_flares, make_systematics, make_LD
from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.datahandling import stitch_spectra, save_s5_output
from juniper.stage5 import bin_light_curves, LSQfit, MCMCfit

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

    # Begin fitting of the light curves by loading the needed dictionaries.
    planets = make_planets(steps)
    flares = make_flares(steps)
    systematics = make_systematics(steps,
                                   xpos=light_curves.xpos.values,
                                   ypos=light_curves.ypos.values,
                                   widths=light_curves.widths.values,)
    LD = make_LD(steps)

    # Now fit the broadband curves.
    if steps["fit_broadband"]:
        # Fit whatever broad-band light curves are supplied. If more than one is supplied,
        # (i.e. if len(light_curves.detectors.values) > 1), we will fit in parallel.
        if steps["use_LSQ"] and len(light_curves.detectors.values)==1:
            if steps["verbose"] == 2:
                print("Linear least squares fitting to a single broadband curve...")
            planets, flares, systematics, LD = LSQfit.lsqfit_one(time=light_curves.time.values[0,:],
                                                                 light_curve=light_curves.broadband.values[0,:],
                                                                 errors=light_curves.broaderr.values[0,:],
                                                                 waves=light_curves.broadbins.values[0,:],
                                                                 planets=planets,flares=flares,
                                                                 systematics=systematics,LD=LD,
                                                                 inpt_dict=steps)
            # Save output. Needs to be formatted as if there is more than one dimension.
            planets_err, flares_err, systematics_err, LD_err = planets, flares, systematics, LD
            save_s5_output(planets, planets_err, flares, flares_err,
                            systematics, systematics_err, LD, LD_err,
                            light_curves.time.values[:,:], light_curves.broadband.values[:,:],
                            outfile+"_broadbandLSQ", outdir)

        elif steps["use_LSQ"] and len(light_curves.detectors.values)!=1:
            if steps["verbose"] == 2:
                print("Linear least squares fitting to multiple broadband curves in parallel...")
            planets, flares, systematics, LD = LSQfit.lsqfit_joint(time=light_curves.time.values[:,:],
                                                                   light_curve=light_curves.broadband.values[:,:],
                                                                   errors=light_curves.broaderr.values[:,:],
                                                                   waves=light_curves.broadbins.values[:,:],
                                                                   planets=planets,flares=flares,
                                                                   systematics=systematics,LD=LD,
                                                                   inpt_dict=steps)
            # Save output.
            planets_err, flares_err, systematics_err, LD_err = planets, flares, systematics, LD
            save_s5_output(planets, planets_err, flares, flares_err,
                           systematics, systematics_err, LD, LD_err,
                           light_curves.time.values[:,:], light_curves.broadband.values[:,:],
                           outfile+"_broadbandLSQ", outdir)
        


        # Now refine those linear fits with MCMC, or just go straight to MCMC if desired.
        if steps["use_MCMC"] and len(light_curves.detectors.values)==1:
            if steps["verbose"] == 2:
                print("Markov Chain Monte Carlo fitting to a single broadband curve...")
            planets, flares, systematics, LD, p_err, f_err, s_err, L_err = MCMCfit.mcmcfit_one(time=light_curves.time.values[0,:],
                                                                                               light_curve=light_curves.broadband.values[0,:],
                                                                                               errors=light_curves.broaderr.values[0,:],
                                                                                               waves=light_curves.broadbins.values[0,:],
                                                                                               planets=planets,flares=flares,
                                                                                               systematics=systematics,LD=LD,
                                                                                               inpt_dict=steps)
            
            # Save output.
            save_s5_output(planets, p_err, flares, f_err,
                           systematics, s_err, LD, L_err,
                           light_curves.time.values[0,:], light_curves.broadband.values[0,:],
                           outfile+"_broadbandMCMC", outdir)
        
        else:
            if steps["verbose"] == 2:
                print("Markov Chain Monte Carlo fitting to multiple broadband curves in parallel...")
            '''
            planets, flares, systematics, LD, p_err, f_err, s_err, L_err = MCMCfit.mcmcfit_joint(time=light_curves.time.values[0,:],
                                                                                               light_curve=light_curves.broadband.values[0,:],
                                                                                               errors=light_curves.broaderr.values[0,:],
                                                                                               waves=light_curves.broadbins.values[0,:],
                                                                                               planets=planets,flares=flares,
                                                                                               systematics=systematics,LD=LD,
                                                                                               inpt_dict=steps)
            # Save output.
            save_s5_output(planets, p_err, flares, f_err,
                            systematics, s_err, LD, L_err,
                            light_curves.time.values[0,:], light_curves.broadband.values[0,:],
                            outfile+"_broadbandMCMC", outdir)
            '''
    
    # Then fit the spectroscopic curves.
    if steps["fit_spec"]:
        # Reserve planets, flares, systematics, and LD originals.
        planets0 = planets.copy()
        flares0 = flares.copy()
        systematics0 = systematics.copy()
        LD0 = LD.copy()
        
        # We need to treat one light curve at a time, one detector at a time.
        for detector in tqdm(light_curves.detectors.values,
                             desc="Processing each detector's spectrum...",
                             disable=(not time_step)):
            for wavelength in tqdm(range(light_curves.specwave.values[detector,:].shape[0]),
                                   desc='Processing each wavelength in that detector...',
                                   disable=(not time_ints)):
                # Fetch the relevant curves and values.
                time = light_curves.time.values[detector,:]
                light_curve = light_curves.spec.values[detector,wavelength,:]
                errors = light_curves.specerr.values[detector,wavelength,:],
                waves = light_curves.specbins.values[detector,wavelength,:]
                wavestr = np.round(light_curves.specwave.values[detector,wavelength],3)

                # First, LSQ.
                if steps["use_LSQ"]:
                    if steps["verbose"] == 2:
                        print("Linear least squares fitting to single spectroscopic light curve..")
                    planets, flares, systematics, LD = LSQfit.lsqfit_one(time=time,
                                                                         light_curve=light_curve,
                                                                         errors=errors,
                                                                         waves=waves,
                                                                         planets=planets,flares=flares,
                                                                         systematics=systematics,LD=LD,
                                                                         inpt_dict=steps,
                                                                         is_spec=True)
                    # Save output.
                    planets_err, flares_err, systematics_err, LD_err = planets, flares, systematics, LD
                    save_s5_output(planets, planets_err, flares, flares_err,
                                   systematics, systematics_err, LD, LD_err,
                                   time, light_curve,
                                   outfile+"_spec{}LSQ".format(wavestr), outdir)
                    
                # Then MCMC.
                if steps["use_MCMC"]:
                    if steps["verbose"] == 2:
                        print("Markov Chain Monte Carlo fitting to single spectroscopic light curve..")
                    planets, flares, systematics, LD, p_err, f_err, s_err, L_err = MCMCfit.mcmcfit_one(time=time,
                                                                                                       light_curve=light_curve,
                                                                                                       errors=errors,
                                                                                                       waves=waves,
                                                                                                       planets=planets,flares=flares,
                                                                                                       systematics=systematics,LD=LD,
                                                                                                       inpt_dict=steps,
                                                                                                       is_spec=True)
            
                    # Save output.
                    save_s5_output(planets, p_err, flares, f_err,
                                   systematics, s_err, LD, L_err,
                                   time, light_curve,
                                   outfile+"_spec{}MCMC".format(wavestr), outdir)
                
                # Reset planets, etc. to originals.
                planets, flares, systematics, LD = planets0, flares0, systematics0, LD0


    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 5 is complete.")