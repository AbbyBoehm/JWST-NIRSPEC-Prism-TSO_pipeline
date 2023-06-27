import os
from copy import deepcopy

import numpy as np

import matplotlib.pyplot as plt

def doStage5(curvesdir, outdir, exoplanet_params, systematics, spectral_range,
             do_fit={"WLC_LM":True,
                     "WLC_MCMC":True,
                     "spec_LM":True,
                     "spec_MCMC":True,},
             limb_darkening_model={"model_type":"quadratic",
                                   "stellar_params":None,
                                   "initial_guess":[0.1,0.1],},
             fixed_param={"LD_coeffs":False,
                          "t0":False,
                          "period":False,
                          "aoR":False,
                          "impact":False,
                          "ecc":False,
                          "lop":False,
                          "offset":False},
             priors_dict={"t0":[-0.1, 0.1],
                         "period":[0, 100],
                         "aoR":[0.00001, 10],
                         "impact":[0,100],
                         "ecc":[0, 1],
                         "lop":[0, 90],
                         "offset":[-0.5, 0.5]},
             priors_type="uniform",
             reject_threshold=3,
             raise_alarm=10,
             exoticLD={"available":False,
                       "ld_data_path":None,
                       "ld_grid":'kurucz',
                       "ld_interpolate_type":'trilinear'},
             save_plots={"WLC_LM":True,
                         "WLC_MCMC":True,
                         "spec_LM":True,
                         "spec_MCMC":True}):
    '''
    Performs Stage 5 LM and MCMC fitting of the light curves in the specified directory.
    
    :param curvesdir: str. Where the .txt files of the curves you want to analyze are stored.
    :param exoplanet_params: dict of float. Contains keywords "t0", "period", "rp", "aoR", "impact", "ecc", "lop", "offset".
    :param systematics: tuple of float. Contains parameters for a linear-in-time fit a+bx.
    :param spectral_range: tuple of float. Spectral range being covered.
    :param limb_darkening_model: dict. Contains "model_type" str which defines model choice (e.g. quadratic, 4-param), "stellar_params" tuple of (M_H, Teff, logg) or None if not using, "initial_guess" keyword containing tuple of floats that will be supplied as the initial guess of LD coefficients if EXoTiC-LD is not being used and the LD coefficients are not otherwise being fixed in advance.
    :param fixed_param. dict of bools. Keywords are parameters that can be held fixed or opened for fitting. If True, parameter will be held fixed. If False, parameter is allowed to be fitted.
    :param priors_dict: dict of tuples of floats. Keywords are params that can be held fixed or opened for fitting. The lists of floats define the edges of the uniform priors, or else mu and sigma for Gaussian priors.
    :param priors_type: str. Choices are "uniform" or "gaussian".
    :param reject_threshold: float. Sigma at which to reject outliers from the linear least-squares residuals when performing MCMC fitting.
    :param raise_alarm: int. If this many outliers are rejected from the residuals, raise an "alarm!" print statement. Not a great system, but for now it is suffiicent. When we do logging, we can make this better.
    :param exoticLD: dict. Contains "available" bool for whether EXoTiC-LD is on this system, "ld_data_path" str of where the exotic_ld_data directory is located, "ld_grid" for LD grid to use (e.g. "kurucz", "stagger"), "ld_interpolate_type" for which type of interpolation to use between grid points (e.g. "nearest", "trilinear").
    :param save_ploots: dict. Keywords are "WLC_LM", "WLC_MCMC", "spec_LM", "spec_MCMC". Whether to save plots associated with each of these fitting types.
    :return: parameters from LM and MCMC fits, transit depths, and errors on depths.
    '''
    original_systematics = deepcopy(systematics)
    # Read out wlc first.
    wlc_path = os.path.join(curvesdir, "wlc.txt")
    wlc, times = read_light_curve(wlc_path)
    
    # Perform LM fit of wlc.
    original_exoplanet_params = deepcopy(exoplanet_params)
    if do_fit["WLC_LM"]:
        LM_priors = {}
        if priors_type == "gaussian":
            for key in priors_dict.keys():
                LM_priors[key] = [priors_dict[key][0]-3*priors_dict[key][1],
                                  priors_dict[key][0]+3*priors_dict[key][1]]
        else:
            LM_priors = priors_dict
        fit_params, fit_model, residuals, uncertainty = LMfit(times, wlc, exoplanet_params, systematics, limb_darkening_model,
                                                              fixed_param, LM_priors, exoticLD, spectral_range)
        # Update exoplanet_params based on fit_params.
        systematics = (fit_params["a"], fit_params["b"])
        for key in fit_params.keys():
            if key not in ("a", "b"):
                exoplanet_params[key] = fit_params[key]

        if save_plots["WLC_LM"]:
            WLC_fitsdir = os.path.join(outdir, "WLC_fits")
            if not os.path.exists(WLC_fitsdir):
                os.makedirs(WLC_fitsdir)
            # Plot the fit.
            plt.figure(figsize=(20,5))
            plt.errorbar(times, wlc, yerr=np.full(shape=len(wlc), fill_value=uncertainty), fmt='o', color='black', label='obs', alpha=0.1, zorder=1)
            plt.plot(times, fit_model, 'r-', label="fitted_function", zorder=10)
            plt.xlabel('time since mid-transit [days]')
            plt.ylabel('relative flux [no units]')
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.savefig(os.path.join(WLC_fitsdir, "wlc_fitLM.pdf"), dpi=300)
            plt.show()
            plt.close()

            # Plot the residuals.
            plt.figure(figsize=(20,5))
            plt.scatter(times, residuals)
            plt.xlabel('time since mid-transit [days]')
            plt.ylabel('normalized residuals')
            plt.savefig(os.path.join(WLC_fitsdir, "wlc_fitLM_residuals.pdf"), dpi=300)
            plt.show()
            plt.close()
    else:
        uncertainty = 0.01 # high uncertainty if not pre-fit
    
    # Perform MCMC fit of wlc with initial guesses updated from fit_params.
    original_exoplanet_params = deepcopy(exoplanet_params)
    if do_fit["WLC_MCMC"]:
        # temporary fix on aoR, inc during MCMC
        #fixed_param["aoR"] = True
        #fixed_param["impact"] = True
        #limb_darkening_model["initial_guess"] = [0.1,0.1]
        if do_fit["WLC_LM"]:
            wlc, times, n_reject = reject_outliers(wlc, times, residuals, sigma=reject_threshold, raise_alarm=raise_alarm)
        theta, depth, depth_err, model, residuals, res_err = MCMCfit(times, wlc, [uncertainty for i in wlc],
                                                                     exoplanet_params, systematics,
                                                                     limb_darkening_model,
                                                                     fixed_param, exoticLD, spectral_range,
                                                                     depth_type=MCMC_depth_type,
                                                                     priors_dict=priors_dict,
                                                                     priors_type=priors_type,
                                                                     N_walkers = 80,
                                                                     N_steps = 30000)
        try:
            print("Obtained grazing broadband depth of %.2f +/- %.2f." % (depth[2], depth_err[2]))
        except:
            print("Obtained broadband depth of %.2f +/- %.2f." % (depth, depth_err))

        # Update exoplanet_params based on theta.
        for key in theta.keys():
            exoplanet_params[key] = theta[key]

        if save_plots["WLC_MCMC"]:
            WLC_fitsdir = os.path.join(outdir, "WLC_fits")
            if not os.path.exists(WLC_fitsdir):
                os.makedirs(WLC_fitsdir)
            fig, ax = plt.subplots(figsize=(20, 5))
            ax.plot(times, wlc, lw=3)
            ax.errorbar(times, wlc, yerr=[uncertainty for i in wlc], fmt="none", capsize=3)
            ax.plot(times, model, color="red")
            ax.set_xlabel("time since mid-transit [MJD]")
            ax.set_ylabel("normalized flux [DN/s]")
            ax.set_title("White light curve with MCMC fit")
            plt.savefig(os.path.join(WLC_fitsdir, "wlc_fitMCMC.pdf"), dpi=300)
            plt.show()
            plt.close()

            # Plot the residuals.
            plt.figure(figsize=(20,5))
            plt.scatter(times, residuals)
            plt.xlabel('time since mid-transit [days]')
            plt.ylabel('normalized residuals')
            plt.savefig(os.path.join(WLC_fitsdir, "wlc_fitMCMC_residuals.pdf"), dpi=300)
            plt.show()
            plt.close()
    
    # Reset systematics.
    systematics = original_systematics
    
    # Perform LM fits on spectroscopic light curves.
    spectro_fixed_param = {"LD_coeffs":fixed_param["LD_coeffs"],
                           "t0":True,
                           "period":True,
                           "aoR":True,
                           "impact":True,#"inc":True,
                           "ecc":True,
                           "lop":True,
                           "offset":True}
    slc_paths = sorted(glob.glob(os.path.join(curvesdir, "slc*")))
    spec_uncertainties = []
    spec_updated_guesses = []
    
    original_exoplanet_params = deepcopy(exoplanet_params)
    LM_residuals = []
    if do_fit["spec_LM"]:
        LM_priors = {}
        if priors_type == "gaussian":
            for key in priors_dict.keys():
                LM_priors[key] = [priors_dict[key][0]-3*priors_dict[key][1],
                                  priors_dict[key][0]+3*priors_dict[key][1]]
        else:
            LM_priors = priors_dict
        for slc_path in slc_paths:
            # These files have names slc_#.###mu_#.###mu.txt, can get spectral range out of these.
            slc, times = read_light_curve(slc_path)
            slc_file = str.split(slc_path, sep="/")[-1]
            savetag = slc_file[4:19]
            min_wav = float(slc_file[4:9])
            max_wav = float(slc_file[12:17])
            slc_spectral_range = (min_wav, max_wav)

            exoplanet_params = deepcopy(original_exoplanet_params) # prevents original guess from being modified.
            print(exoplanet_params)
            fit_params, fit_model, residuals, uncertainty = LMfit(times, slc, exoplanet_params, systematics, limb_darkening_model,
                                                                  spectro_fixed_param, LM_priors, exoticLD, slc_spectral_range)
            spec_updated_guesses.append(deepcopy(fit_params))
            spec_uncertainties.append(uncertainty)
            LM_residuals.append(residuals)

            if save_plots["spec_LM"]:
                # Save plots of the LM fits to the spec curves.
                spec_fitsdir = os.path.join(outdir, "spec_fits")
                if not os.path.exists(spec_fitsdir):
                    os.makedirs(spec_fitsdir)
                # Plot the fit.
                plt.figure(figsize=(20,5))
                plt.errorbar(times, slc, yerr=np.full(shape=len(slc), fill_value=uncertainty), fmt='o', color='black', label='obs', alpha=0.1, zorder=1)
                plt.plot(times, fit_model, 'r-', label="fitted_function", zorder=10)
                plt.xlabel('time since mid-transit [days]')
                plt.ylabel('relative flux [no units]')
                plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                plt.savefig(os.path.join(spec_fitsdir, "slc_fitLM_{}.pdf".format(savetag)), dpi=300)
                #plt.show()
                plt.close()

                # Plot the residuals.
                plt.figure(figsize=(20,5))
                plt.scatter(times, residuals)
                plt.xlabel('time since mid-transit [days]')
                plt.ylabel('normalized residuals')
                plt.savefig(os.path.join(spec_fitsdir, "slc_fitLM_residuals_{}.pdf".format(savetag)), dpi=300)
                #plt.show()
                plt.close()
    else:
        spec_uncertainties = [0.01 for i in slc_paths]
        spec_updated_guesses = [original_exoplanet_params for i in slc_paths]
    
    # Now do MCMC fits.
    rp_vals = []
    rp_errs = []
    alt_depths = []
    alt_depth_errs = []
    depths = []
    depth_errs = []
    wavelengths = []
    halfwidths = []
    original_exoplanet_params = deepcopy(exoplanet_params)
    n_rejected = []
    SDNRs = []
    if do_fit["spec_MCMC"]:
        for slc_path, guess, uncertainty, residual in zip(slc_paths,spec_updated_guesses, spec_uncertainties, LM_residuals):
            # These files have names slc_#.###mu_#.###mu.txt, can get spectral range out of these.
            slc, times = read_light_curve(slc_path)
            slc_file = str.split(slc_path, sep="/")[-1]
            savetag = slc_file[4:19]
            min_wav = float(slc_file[4:9])
            max_wav = float(slc_file[12:17])
            slc_spectral_range = (min_wav, max_wav)
            wavelengths.append((min_wav+max_wav)/2)
            halfwidths.append((max_wav-min_wav)/2)
            
            # Use LM residuals to spot outliers from the fit and delete them.
            slc, times, n_reject = reject_outliers(slc, times, residual, sigma=reject_threshold, raise_alarm=raise_alarm)

            # guess has to be used to update exoplanet_params.
            exoplanet_params = deepcopy(original_exoplanet_params) # prevents original guess from being modified.
            for key in exoplanet_params.keys():
                if key not in guess.keys():
                    guess[key] = exoplanet_params[key]

            theta, depth, depth_err, model, residuals, res_err = MCMCfit(times, slc, [uncertainty for i in slc],
                                                                         guess, systematics,
                                                                         limb_darkening_model,
                                                                         spectro_fixed_param, exoticLD, slc_spectral_range,
                                                                         depth_type=MCMC_depth_type,
                                                                         priors_dict=priors_dict,
                                                                         priors_type=priors_type)
            print("For wavelength range {}, {}:".format(min_wav, max_wav))
            try:
                # Outputs as 100*rp/rs, standard, ldcta.
                rp_vals.append(depth[0])
                rp_errs.append(depth_err[0])
                
                depths.append(depth[2])
                depth_errs.append(depth_err[2])
                
                alt_depths.append(depth[1])
                alt_depth_errs.append(depth_err[1])
                
                print("Obtained overlap depth %.2f +/- %.2f, at SDNR = %.2f ppm." % (depth[2], depth_err[2], np.std(residuals)*10**6))
            except:
                print("Obtained depth %.2f +/- %.2f, at SDNR = %.2f ppm." % (depth, depth_err, np.std(residuals)*10**6))
                depths.append(depth)
                depth_errs.append(depth_err)
            
            n_rejected.append(n_reject)
            SDNRs.append(np.std(residuals)*10**6)
            if save_plots["spec_MCMC"]:
                # Save plots of fits.
                spec_fitsdir = os.path.join(outdir, "spec_fits")
                if not os.path.exists(spec_fitsdir):
                    os.makedirs(spec_fitsdir)
                fig, ax = plt.subplots(figsize=(20, 5))
                ax.plot(times, slc, lw=3)
                ax.errorbar(times, slc, yerr=[uncertainty for i in slc], fmt="none", capsize=3)
                ax.plot(times, model, color="red")
                ax.set_xlabel("time since mid-transit [MJD]")
                ax.set_ylabel("normalized flux [DN/s]")
                ax.set_title("Spectroscopic light curve (%.3f micron) with MCMC fit" % wavelengths[-1])
                plt.savefig(os.path.join(spec_fitsdir, "slc_fitMCMC_{}.pdf".format(savetag)), dpi=300)
                #plt.show()
                plt.close()

                # Plot the residuals.
                plt.figure(figsize=(20,5))
                plt.scatter(times, residuals)
                plt.xlabel('time since mid-transit [days]')
                plt.ylabel('normalized residuals')
                plt.savefig(os.path.join(spec_fitsdir, "slc_fitMCMC_residuals_{}.pdf".format(savetag)), dpi=300)
                #plt.show()
                plt.close()

        if save_plots["spec_MCMC"]:
            # Save plots of spec curve MCMC results.
            try:
                fig, ax = plot_transit_spectrum(wavelengths, depths, depth_errs,
                                                ymin=0.95*min(depths), ymax=1.05*max(depths))
                spec_outdir = os.path.join(outdir, "spectrum")
                if not os.path.exists(spec_outdir):
                    os.makedirs(spec_outdir)
                plt.savefig(os.path.join(spec_outdir, "slc_transitspectrum.pdf"), dpi=300)
                plt.show()
                plt.close()
            except:
                print("Error encountered in plotting depths, passing...")
            
            try:
                if rp_vals:
                    fig, ax = plot_transit_spectrum(wavelengths, rp_vals, rp_errs,
                                                ymin=0.95*min(rp_vals), ymax=1.05*max(rp_vals))
                    spec_outdir = os.path.join(outdir, "spectrum")
                    if not os.path.exists(spec_outdir):
                        os.makedirs(spec_outdir)
                    plt.savefig(os.path.join(spec_outdir, "slc_transitspectrum_rprs.pdf"), dpi=300)
                    plt.show()
                    plt.close()
            except:
                print("Error encountered in plotting depths, passing...")
            
            try:
                if alt_depths:
                    fig, ax = plot_transit_spectrum(wavelengths, alt_depths, alt_depth_errs,
                                                ymin=0.95*min(alt_depths), ymax=1.05*max(alt_depths))
                    spec_outdir = os.path.join(outdir, "spectrum")
                    if not os.path.exists(spec_outdir):
                        os.makedirs(spec_outdir)
                    plt.savefig(os.path.join(spec_outdir, "slc_transitspectrum_standard.pdf"), dpi=300)
                    plt.show()
                    plt.close()
            except:
                print("Error encountered in plotting depths, passing...")

        # Write out transit spectrum.
        spec_outdir = os.path.join(outdir, "spectrum")
        try:
            write_transit_spectrum(wavelengths, halfwidths, depths, depth_errs, spec_outdir)
        except:
            print("Error encountered in saving depths, passing...")
        
        try:
            if rp_vals:
                spec_outdir = os.path.join(outdir, "spectrum_rprs")
                write_transit_spectrum(wavelengths, halfwidths, rp_vals, rp_errs, spec_outdir)
        except:
            print("Error encountered in saving depths, passing...")
            
        try:
            if alt_depths:
                spec_outdir = os.path.join(outdir, "spectrum_rprs2")
                write_transit_spectrum(wavelengths, halfwidths, alt_depths, alt_depth_errs, spec_outdir)
        except:
            print("Error encountered in saving depths, passing...")
        with open(os.path.join(spec_outdir,"{}_sigma_SDNR.txt".format(reject_threshold)), mode="w") as f:
            f.write("#wavelength [mu]    n_reject [na]    SDNR [ppm]:\n")
            for wav, n_reject, SDNR in zip(wavelengths, n_rejected, SDNRs):
                f.write("{}    {}    {}\n".format(wav, n_reject, SDNR))

def read_light_curve(filepath):
    '''
    Reads out the light curve .txt located at filepath.
    
    :param filepath: str. Where the light curve .txt object is located.
    :return: lc_n, t object.
    '''
    lc = []
    t = []
    with open(filepath) as f:
            line = f.readline() 
            while line[0] == '#':
                # Read past comments.
                line = f.readline()
            while line != '':
                line = str.split(line)#, sep='   ')
                
                # Extract useful info.
                time = float(line[0]) # time in days relative to mid-transit
                flux = float(str.replace(line[1],'\n','')) # normalized flux
                
                t.append(time)
                lc.append(flux)
                
                line = f.readline()
    return np.array(lc), np.array(t)

### Note to developers: I think a lot of this stuff is going to be replaced.
### I rewrote a lot of the fitting/plotting/saving utils for this stage when
### I was reworking my HST WFC3/G280 TSO pipeline and I like a lot of the
### handling in that version much more than I like the JWST pipeline's version.
### So expect those overhauls shortly!