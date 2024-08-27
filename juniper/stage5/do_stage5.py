import os
import glob
from tqdm import tqdm

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from juniper.config.translate_config import make_planets, make_flares, make_systematics, make_LD
from juniper.util.diagnostics import tqdm_translate, plot_translate
from juniper.util.datahandling import stitch_spectra, save_s5_output
from juniper.util.plotting import plot_chains, plot_corner, plot_post
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
        print("Output will be saved to {}_####.npy.".format(outfile))
    
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
                                   xpos=light_curves.xpos.values[0],
                                   ypos=light_curves.ypos.values[0],
                                   widths=light_curves.widths.values[0],)
    LD = make_LD(steps)

    # Now fit the broadband curves.
    if steps["fit_broadband"]:
        # Fit whatever broad-band light curves are supplied. If more than one is supplied,
        # (i.e. if len(light_curves.detectors.values) > 1), we will fit in parallel.
        if steps["use_LSQ"] and len(light_curves.detectors.values)==1:
            if steps["verbose"] == 2:
                print("Linear least squares fitting to a single broadband curve...")
            planets, flares, systematics, LD = LSQfit.lsqfit_one(lc_time=light_curves.time.values[0,:],
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
                            light_curves.time.values[:,:], light_curves.broadband.values[:,:],light_curves.broaderr.values[0,:],'broadband',
                            outfile+"_broadbandLSQ", outdir)

        elif steps["use_LSQ"] and len(light_curves.detectors.values)!=1:
            if steps["verbose"] == 2:
                print("Linear least squares fitting to multiple broadband curves in parallel...")
            planets, flares, systematics, LD = LSQfit.lsqfit_joint(lc_time=light_curves.time.values[:,:],
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
                           light_curves.time.values[:,:], light_curves.broadband.values[:,:],light_curves.broaderr.values[:,:],'broadband',
                           outfile+"_broadbandLSQ", outdir)
        


        # Now refine those linear fits with MCMC, or just go straight to MCMC if desired.
        if steps["use_MCMC"] and len(light_curves.detectors.values)==1:
            if steps["verbose"] == 2:
                print("Markov Chain Monte Carlo fitting to a single broadband curve...")
            planets, flares, systematics, LD, p_err, f_err, s_err, L_err, plotting_items = MCMCfit.mcmcfit_one(lc_time=light_curves.time.values[0,:],
                                                                                                               light_curve=light_curves.broadband.values[0,:],
                                                                                                               errors=light_curves.broaderr.values[0,:],
                                                                                                               waves=light_curves.broadbins.values[0,:],
                                                                                                               planets=planets,flares=flares,
                                                                                                               systematics=systematics,LD=LD,
                                                                                                               inpt_dict=steps)
            
            # Save output.
            save_s5_output(planets, p_err, flares, f_err,
                           systematics, s_err, LD, L_err,
                           light_curves.time.values[:,:], light_curves.broadband.values[:,:], light_curves.broaderr.values[0,:], 'broadband',
                           outfile+"_broadbandMCMC", outdir)
            
            # Plot, if asked.
            if (plot_step or save_step):
                # Unpack plotting items.
                ndim, samples, flat_samples, labels, n = plotting_items
                labels = [str.replace(a, "_prior", "") for a in labels] # clip off the prior tag
                # Plot posteriors.
                fig, ax = plot_post(ndim,samples,labels,n)
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_broadbandMCMC-posterior.png"),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show(block=True)
                plt.close()

                # Plot corners.
                fig = plot_corner(flat_samples,labels)
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_broadbandMCMC-corner.png"),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show(block=True)
                plt.close()

                # Plot chains.
                fig, ax = plot_chains(ndim,samples,labels)
                if save_step:
                    plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_broadbandMCMC-chains.png"),
                                dpi=300, bbox_inches='tight')
                if plot_step:
                    plt.show(block=True)
                plt.close()
        
        else:
            if steps["verbose"] == 2:
                print("Markov Chain Monte Carlo fitting to multiple broadband curves in parallel...")
            '''
            planets, flares, systematics, LD, p_err, f_err, s_err, L_err = MCMCfit.mcmcfit_joint(lc_time=light_curves.time.values[0,:],
                                                                                               light_curve=light_curves.broadband.values[0,:],
                                                                                               errors=light_curves.broaderr.values[0,:],
                                                                                               waves=light_curves.broadbins.values[0,:],
                                                                                               planets=planets,flares=flares,
                                                                                               systematics=systematics,LD=LD,
                                                                                               inpt_dict=steps)
            # Save output.
            save_s5_output(planets, p_err, flares, f_err,
                            systematics, s_err, LD, L_err,
                            light_curves.time.values[0,:], light_curves.broadband.values[0,:], light_curves.broaderr.values[0,:], 'broadband',
                            outfile+"_broadbandMCMC", outdir)
            '''
    
    # Then fit the spectroscopic curves.
    if steps["fit_spec"]:
        # Load planets, flares, systematics, and LD from a fitted model, if available.
        if not steps["fit_broadband"]:
            try:
                result = np.load(os.path.join(outdir,outfile+"_broadbandMCMC.npy"), allow_pickle=True).item()
                planets, flares, systematics, LD = (result['planets'],result['flares'],
                                                    result['systematics'],result['LD'])
                print("MCMC broadband fit successfully loaded.")
            except FileNotFoundError:
                print("No MCMC results found, proceeding with initial guesses.")
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

                # Confirm it's not empty.
                if all([i == 0 for i in light_curve]) or all([np.isnan(i) for i in light_curve]):
                    print("Light curve number {} is an empty light curve, passing...".format(wavelength))
                    continue

                # Confirm it spans more than one wavelength.
                if waves[0] == waves[1]:
                    print("Light curve number {} does not span a wavelength range, passing...".format(wavelength))
                    continue

                # Confirm it is not spanning wavelength 0 nm, for which no photons exist.
                if 0 in waves:
                    print("Light curve number {} spans a wavelength range where photons don't exist, passing...".format(wavelength))
                    continue

                # First, LSQ.
                try:
                    if steps["use_LSQ"]:
                        if steps["verbose"] == 2:
                            print("Linear least squares fitting to single spectroscopic light curve..")
                        planets, flares, systematics, LD = LSQfit.lsqfit_one(lc_time=time,
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
                                    time, light_curve, errors, wavestr,
                                    outfile+"_spec{}LSQ".format(wavestr), outdir)
                except:
                    if steps["verbose"] == 2:
                        print("Linear least squares fitting failed, likely due to batman convergence failure.")
                # Then MCMC.
                try:
                    if steps["use_MCMC"]:
                        if steps["verbose"] == 2:
                            print("Markov Chain Monte Carlo fitting to single spectroscopic light curve...")
                        planets, flares, systematics, LD, p_err, f_err, s_err, L_err, plotting_items = MCMCfit.mcmcfit_one(lc_time=time,
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
                                        time, light_curve, errors, wavestr,
                                        outfile+"_spec{}MCMC".format(wavestr), outdir)
                        
                        # Plot, if asked.
                        if (plot_ints or save_ints):
                            # Unpack plotting items.
                            ndim, samples, flat_samples, labels, n = plotting_items
                            labels = [str.replace(a, "_prior", "") for a in labels] # clip off the prior tag
                            # Plot posteriors.
                            fig, ax = plot_post(ndim,samples,labels,n)
                            if save_ints:
                                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}MCMC-posterior.png".format(wavestr)),
                                            dpi=300, bbox_inches='tight')
                            if plot_ints:
                                plt.show(block=True)
                            plt.close()

                            # Plot corners.
                            fig = plot_corner(flat_samples,labels)
                            if save_ints:
                                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}MCMC-corner.png".format(wavestr)),
                                            dpi=300, bbox_inches='tight')
                            if plot_ints:
                                plt.show(block=True)
                            plt.close()

                            # Plot chains.
                            fig, ax = plot_chains(ndim,samples,labels)
                            if save_step:
                                plt.savefig(os.path.join(plot_dir,"s5_"+outfile+"_spec{}MCMC-chains.png".format(wavestr)),
                                            dpi=300, bbox_inches='tight')
                            if plot_step:
                                plt.show(block=True)
                            plt.close()


                except:
                    if steps["verbose"] == 2:
                        print("Markov Chain Monte Carlo fititng failed, likely due to batman convergence failure.")
                # Reset planets, etc. to originals.
                planets, flares, systematics, LD = planets0, flares0, systematics0, LD0


    # Log.
    if steps["verbose"] >= 1:
        print("Juniper Stage 5 is complete.")