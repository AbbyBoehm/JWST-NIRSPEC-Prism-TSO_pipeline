import os
import glob
import shutil

import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits

from jwst import datamodels as dm


import corner
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import matplotlib.colors as mplcolors
import matplotlib.cm as cmx
from matplotlib.ticker import FormatStrFormatter as fsf

import batman

try:
    from exotic_ld import StellarLimbDarkening as SLD
    exotic_ld_available = True
except:
    print("EXoTiC-LD not found on this system, fixed limb darkening coefficients will not be available.")
    exotic_ld_available = False

def img(array, aspect=1, title=None, vmin=None, vmax=None, norm=None):
    '''
    Image plotting utility to plot the given 2D array.
    
    :param array: 2D array. Image you want to plot.
    :param aspect: float. Aspect ratio. Useful for visualizing narrow arrays.
    :param title: str. Title to give the plot.
    :param vmin: float. Minimum value for color mapping.
    :param vmax: float. Maximum value for color mapping.
    :param norm: str. Type of normalisation scale to use for this image.
    '''
    fig, ax = plt.subplots(figsize=(20, 25))
    if norm == None:
        im = ax.imshow(array, aspect=aspect, origin="lower", vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(array, aspect=aspect, norm=norm, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    return fig, ax, im

def stitch_files(files):
    '''
    Reads all *_calints.fits files given by the filepaths provided
    and stitches them together.

    :param
    '''
    if "postprocessed" in files[0]:
        print("Reading post-processed files, adjusting outputs accordingly...")
        # Reading a postprocessed file, adjust strategy!
        for i, file in enumerate(files):
            print("Attempting to locate file: " + file)
            fitsfile = dm.open(file)
            print("Loaded the file " + file + " successfully.")

            # Need to retrieve the wavelength object, science object, and DQ object from this.
            if i == 0:
                segments = fitsfile.data
                errors = fitsfile.err
                segstarts = [np.shape(fitsfile.data)[0]]
                wavelengths = [fitsfile.wavelength]
                dqflags = fitsfile.dq
                times = fitsfile.int_times["int_mid_MJD_UTC"]
            else:
                segments = np.concatenate([segments, fitsfile.data], 0)
                errors = np.concatenate([errors, fitsfile.err], 0)
                segstarts.append(np.shape(fitsfile.data)[0] + sum(segstarts))
                wavelengths.append(fitsfile.wavelength)
                dqflags = np.concatenate([dqflags, fitsfile.dq], 0)
                times = np.concatenate([times, fitsfile.int_times["int_mid_MJD_UTC"]], 0)
            
            fitsfile.close()
            
            with fits.open(file) as f:
                frames_to_reject = f["REJECT"].data
            
            print("Retrieved segments, wavelengths, DQ flags, and times from file: " + file)
            print("Closing file and moving on to next one...")
        return segments, errors, segstarts, wavelengths, dqflags, times, frames_to_reject
    else:
        for i, file in enumerate(files):
            print("Attempting to locate file: " + file)
            fitsfile = dm.open(file)
            print("Loaded the file " + file + " successfully.")

            # Need to retrieve the wavelength object, science object, and DQ object from this.
            if i == 0:
                segments = fitsfile.data
                errors = fitsfile.err
                segstarts = [np.shape(fitsfile.data)[0]]
                wavelengths = [fitsfile.wavelength]
                dqflags = fitsfile.dq
                times = fitsfile.int_times["int_mid_MJD_UTC"]
            else:
                segments = np.concatenate([segments, fitsfile.data], 0)
                errors = np.concatenate([errors, fitsfile.err], 0)
                segstarts.append(np.shape(fitsfile.data)[0] + sum(segstarts))
                wavelengths.append(fitsfile.wavelength)
                dqflags = np.concatenate([dqflags, fitsfile.dq], 0)
                times = np.concatenate([times, fitsfile.int_times["int_mid_MJD_UTC"]], 0)

            print("Retrieved segments, wavelengths, DQ flags, and times from file: " + file)
            print("Closing file and moving on to next one...")
            fitsfile.close()
        return segments, errors, segstarts, wavelengths, dqflags, times
    
def initialize_batman_params(exoplanet_params, event_type):
    params = batman.TransitParams()
    params.per = exoplanet_params["period"]                                 #orbital period in days
    params.rp = exoplanet_params["rp"]                                      #planet radius (in units of stellar radii)
    if event_type == "transit":
        params.t0 = exoplanet_params["t0"]                                  #time of inferior conjunction in days
    if event_type == "eclipse":
        params.fp = exoplanet_params["fp"]                                  #planet flux (in units of stellar flux)
        params.t_secondary = exoplanet_params["t_secondary"]                #time of superior conjunction in days
    params.a = exoplanet_params["aoR"]                                      #semi-major axis (in units of stellar radii)
    params.inc = exoplanet_params["inc"]                                    #orbital inclination (in degrees)
    params.ecc = exoplanet_params["ecc"]                                    #eccentricity
    params.w = exoplanet_params["lop"]                                      #longitude of periastron (in degrees)
    params.u = exoplanet_params["LD_coeffs"]                                #limb darkening coefficients [u1, u2]
    params.limb_dark = exoplanet_params["model_type"]                       #limb darkening model

    return params

def build_theta_dict(a1, a2, exoplanet_params, fixed_param):
    theta_guess = {}
    
    theta_guess["a1"] = a1
    theta_guess["a2"] = a2
    
    #if "rp" in list(exoplanet_params.keys()):
    #    theta_guess["rp"] = exoplanet_params["rp"]
    #if "fp" in list(exoplanet_params.keys()):
    #    theta_guess["fp"] = exoplanet_params["fp"]
    
    for parameter in fixed_param.keys():
        if not fixed_param[parameter]:
            theta_guess[parameter] = exoplanet_params[parameter]
    
    return theta_guess

def update_params(theta, params):
    a1 = theta["a1"]
    a2 = theta["a2"]
    
    fit_params = theta.keys()
    if "rp" in fit_params:
        params.rp = theta["rp"]
    if "fp" in fit_params:
        params.fp = theta["fp"]
    if "LD_coeffs" in fit_params:
        params.u = theta["LD_coeffs"]
    if "t0" in fit_params:
        params.t0 = theta["t0"]
    if "t_secondary" in fit_params:
        params.t_secondary = theta["t_secondary"]
    if "period" in fit_params:
        params.per = theta["period"]
    if "aoR" in fit_params:
        params.a = theta["aoR"]
    if "inc" in fit_params:
        params.inc = theta["inc"]
    if "ecc" in fit_params:
        params.ecc = theta["ecc"]
    if "lop" in fit_params:
        params.w = theta["lop"]
    
    return a1, a2, params

def broadcast_array_back_to_dict(theta_arr, theta, modified_keys, using_kipping):
    if "LD_coeff" in modified_keys:
        LD_coeffs = []
    checked = 0
    i = 0
    for theta_arr_item, modified_key in zip(theta_arr, modified_keys):
        if "LD_coeff" not in modified_key:
            theta[modified_key] = theta_arr_item
        else:
            if using_kipping:
                if checked == 0:
                    u1 = 2*np.sqrt(theta_arr[i])*theta_arr[i+1]
                    LD_coeffs.append(u1)
                    checked += 1
                else:
                    u2 = np.sqrt(theta_arr[i-1])*(1-2*theta_arr[i])
                    LD_coeffs.append(u2)
            else:
                LD_coeffs.append(theta_arr_item)
        i += 1
    if "LD_coeff" in modified_keys:
        theta["LD_coeffs"] = LD_coeffs
    return theta

def turn_array_to_dict(modified_keys, fit_theta_arr):
    fit_theta_dict = {}
    if "LD_coeff" in modified_keys:
        fitted_LD_coeffs = []
    for fitted_param, modified_key in zip(fit_theta_arr, modified_keys):
        if "LD_coeff" != modified_key:
            fit_theta_dict[modified_key] = fitted_param
        else:
            fitted_LD_coeffs.append(fitted_param)
    if "LD_coeff" in modified_keys:
        fit_theta_dict["LD_coeffs"] = fitted_LD_coeffs
    return fit_theta_dict

def turn_dict_to_array(theta_dict):
    theta_arr = []
    modified_keys = []
    for key in theta_dict.keys():
        if key=="LD_coeffs":
            for ld_coeff in theta_dict[key]:
                theta_arr.append(ld_coeff)
                modified_keys.append("LD_coeff")
        else:
            theta_arr.append(theta_dict[key])
            modified_keys.append(key)
    return np.array(theta_arr), modified_keys

def make_LSQ_bounds_object(theta_dict, priors_dict, priors_type, ld_lower, ld_upper):
    lower_bounds = []
    upper_bounds = []
    for key in theta_dict.keys():
        if key=="LD_coeffs":
            for ld_coeff in theta_dict[key]:
                lower_bounds.append(ld_lower)
                upper_bounds.append(ld_upper)
        elif key in priors_dict.keys():
            if priors_type == "gaussian":
                mu = priors_dict[key][0]
                sigma = priors_dict[key][1]
                lower_bounds.append(mu-sigma)
                upper_bounds.append(mu+sigma)
            if priors_type == "uniform":
                lower_bounds.append(priors_dict[key][0])
                upper_bounds.append(priors_dict[key][1])
        else:
            lower_bounds.append(-np.inf)
            upper_bounds.append(np.inf)
        #print("Supplying following bounds for key: ", key)
        #print(lower_bounds[-1], upper_bounds[-1])
    bounds=(lower_bounds, upper_bounds)
    return bounds

if exotic_ld_available:
    def get_exotic_coefficients(exoplanet_params, stellar_params, exoticLD, custom=None):
        if stellar_params != None:
            # Get LD coefficients from EXoTiC-LD. If this is None, you had better have a custom model in place!
            M_H, Teff, logg = stellar_params
        # And wavelengths for fitting coefficients
        spectral_range = exoticLD["spectral_range"]
        # And read mode from exoticLD
        mode = exoticLD["instrument_mode"]
        # Check for custom model.
        if exoticLD["ld_grid"] == "custom":
            file_path = exoticLD["custom_model_path"]

            s_wvs = (np.genfromtxt(file_path, skip_header = 2, usecols = [0]).T)*1e4
            s_mus = np.flip(np.genfromtxt(file_path, skip_header = 1, max_rows = 1))
            stellar_intensity = np.flip(np.genfromtxt(file_path, skip_header = 2)[:,1:],axis = 1)

            sld = SLD(ld_data_path=exoticLD["ld_data_path"], ld_model="custom",
                        custom_wavelengths=s_wvs, custom_mus=s_mus, custom_stellar_model=stellar_intensity)
        else:
            sld = SLD(M_H=M_H, Teff=Teff, logg=logg,
                    ld_model=exoticLD["ld_grid"], ld_data_path=exoticLD["ld_data_path"],
                    interpolate_type=exoticLD["ld_interpolate_type"], verbose=True)
        
        if exoplanet_params["model_type"] in ("quadratic", "kipping2013"):
            exoplanet_params["LD_coeffs"] = sld.compute_quadratic_ld_coeffs(wavelength_range=[spectral_range[0], spectral_range[1]],
                                                                            mode=mode, mu_min=0.0)
        if exoplanet_params["model_type"] == "square-root":
            exoplanet_params["LD_coeffs"] = sld.compute_squareroot_ld_coeffs(wavelength_range=[spectral_range[0], spectral_range[1]],
                                                                                mode=mode, mu_min=0.0)
        if exoplanet_params["model_type"] == "nonlinear":
            exoplanet_params["LD_coeffs"] = sld.compute_4_parameter_non_linear_ld_coeffs(wavelength_range=[spectral_range[0], spectral_range[1]],
                                                                                            mode=mode, mu_min=0.0)
        
        return exoplanet_params

def plot_fit_and_res(t, lc, err, model, interp_t, residuals):
    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    axes[0].scatter(t, lc, c="k", s=20, alpha=0.75, edgecolors="k")
    axes[0].errorbar(t, lc, yerr=err, fmt="none", capsize=0, elinewidth=2, ecolor="k")
    axes[0].plot(interp_t, model, color="red", lw=2)
    axes[1].scatter(t, residuals, color="k")
    return fig, axes

def plot_chains(ndim, samples, labels):
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        try:
            ax = axes[i]
        except TypeError:
            # There is only one axis because there was only one sample.
            ax = axes
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        
    #axes[-1].set_xlabel("step number")
    ax.set_xlabel("step number") # will automatically grab last used axis
    return fig, axes

def plot_post(ndim, samples, labels, n):
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=False)
    for i in range(ndim):
        try:
            ax = axes[i]
        except TypeError:
            # There is only one axis because there was only one sample.
            ax = axes
        post = np.reshape(samples[:, :, i], (n))
        ax.hist(post, 100, alpha=0.3)
        ax.set_xlim(min(post), max(post))
        ax.set_ylabel(labels[i])
    return fig, axes

def plot_corner(nrs2_samples, labels):
    fig = corner.corner(nrs2_samples, labels=labels)
    return fig

def write_run_to_text(outdir, outfile, fit_theta, err_theta, fit_err,
                      exoplanet_params, limb_darkening_model,
                      fixed_param, priors_dict, priors_type, N_walkers, N_steps, exoticLD):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, outfile),mode="w") as f:
        f.write("Outputs of fit:\n")
        for key in fit_theta.keys():
            try:
                f.write("{},{},{}\n".format(key,fit_theta[key],err_theta[key]))
            except:
                f.write("{},{},{}\n".format(key,fit_theta[key],"NA"))
        f.write("SDNR of residuals:\n")
        f.write("SDNR(ppm),{}\n".format(fit_err*10**6))
        
        f.write("Depths of fit:\n")
        if "rp" in fit_theta.keys():
            depth, depth_err = compute_depth_rprs(fit_theta, err_theta)
            f.write("(Rp/Rs),{},{}\n".format(depth,depth_err))
            depth, depth_err = compute_depth_rprs2(fit_theta, err_theta)
            f.write("(Rp/Rs)^2,{},{}\n".format(depth,depth_err))
            depth, depth_err = compute_depth_aoverlap(fit_theta, err_theta, exoplanet_params)
            f.write("A_overlap/A*,{},{}\n".format(depth,depth_err))
        if "fp" in fit_theta.keys():
            depth, depth_err = compute_depth_fpfs(fit_theta, err_theta)
            f.write("(Fp/Fs),{},{}\n".format(depth,depth_err))

        f.write("Run parameters - exoplanet_params:\n")
        for key in exoplanet_params.keys():
            f.write("{},{}\n".format(key,exoplanet_params[key]))
        
        f.write("Run parameters - limb_darkening_model:\n")
        for key in limb_darkening_model.keys():
            f.write("{},{}\n".format(key,limb_darkening_model[key]))
        
        f.write("Run parameters - fixed_param:\n")
        for key in fixed_param.keys():
            f.write("{},{}\n".format(key,fixed_param[key]))

        f.write("Run parameters - priors_dict:\n")
        for key in priors_dict.keys():
            f.write("{},{}\n".format(key,priors_dict[key]))
        f.write("Type,{}\n".format(priors_type))

        f.write("Run parameters - MCMC settings:\n")
        f.write("N_walkers,{}\n".format(N_walkers))
        f.write("N_steps,{}\n".format(N_steps))
        
        f.write("Run parameters - exoticLD:\n")
        for key in exoticLD.keys():
            f.write("{},{}\n".format(key,exoticLD[key]))

def get_spectrum(thetas, errs, SDNRs, exoplanet_params, depth_type, ignore_high_SDNR=False, SDNRlimit=12000):
    wavelengths = [i for i in thetas.keys()]
    retained_wavelengths = []
    depths = []
    depth_errs = []

    for l in wavelengths:
        theta = thetas[l]
        theta_err = errs[l]
        SDNR = SDNRs[l]

        # With the thetas acquired, compute the chosen depth type.
        if depth_type == "rprs":
            depth, depth_err = compute_depth_rprs(theta, theta_err)
        if depth_type == "rprs2":
            depth, depth_err = compute_depth_rprs2(theta, theta_err)
        if depth_type == "aoverlap":
            depth, depth_err = compute_depth_aoverlap(theta, theta_err, exoplanet_params)
        if depth_type == "fpfs":
            depth, depth_err = compute_depth_fpfs(theta, theta_err)
        
        if (ignore_high_SDNR and SDNR > SDNRlimit):
            print("Depth at wavelength {} had residuals in excess of {} ppm and will be ignored...".format(l, SDNRlimit))
        else:
            depths.append(depth)
            depth_errs.append(depth_err)
            retained_wavelengths.append(l)
    
    # Now need to convert wavelengths to numbers
    wavelengths = [float(x) for x in retained_wavelengths]
    return wavelengths, depths, depth_errs

def write_spectrum(outdir, outfile, wavelengths, depths, depth_errs):
    with open(os.path.join(outdir, outfile),mode="w") as f:
        f.write("#wavelength[AA]     bin[AA]     depth[na]     err[na]\n")
        for i in range(0, len(wavelengths)):#-1):
            l = wavelengths[i]
            if i == 0:
                hw = (wavelengths[i+1] - wavelengths[i])
            elif i == (len(wavelengths) - 1):
                hw = wavelengths[i]-wavelengths[i-1]
            else:
                hw = (wavelengths[i+1] - wavelengths[i-1])/2
            d = depths[i]
            e = depth_errs[i]
            f.write("{}     {}     {}     {}\n".format(l,hw,d,e))

def plot_transit_spectrum(wavelengths, depths, depth_errs, reference_wavelengths, reference_depths, reference_errs, reference_names, ylim):
    fig, ax = plt.subplots(figsize=(20,7))
    colors = ("red","blue","forestgreen","gold","orange","indigo","violet")
    if reference_depths:
        for reference_wavelength, reference_depth, reference_err, reference_name, color in zip(reference_wavelengths,reference_depths,reference_errs,reference_names,colors):
            print("Plotting reference spectrum with {} points...".format(len(reference_depth)))
            ax.scatter(reference_wavelength, reference_depth, s=20, color=color, marker="o", label=reference_name)
            ax.errorbar(reference_wavelength, reference_depth, yerr=reference_err, fmt="none", capsize=0, ecolor=color, elinewidth=2)
    # Mask NaNs so that plots can always be made even if NaNs are present.
    depths = np.ma.masked_invalid([100*x for x in depths])
    depth_errs = np.ma.masked_array(np.array([100*x for x in depth_errs]), np.ma.getmask(depths))
    wavelengths = np.ma.masked_array(np.array(wavelengths), np.ma.getmask(depths))
    ax.scatter(wavelengths, depths, s=20, color="k", marker="o", label="this reduction")
    ax.errorbar(wavelengths, depths, yerr=depth_errs, fmt="none", capsize=0, ecolor="k", elinewidth=2)

    ax.set_xlabel("wavelength [mu]", fontsize=14)
    ax.set_ylabel("transit depth [%]", fontsize=14)
    ax.set_xlim(np.min(wavelengths)-0.05,np.max(wavelengths)+0.05) # set it so that the new spectrum dominates the FOV
    try:
        ax.set_ylim(100*ylim[0], 100*ylim[1])
    except ValueError:
        # Happens if NaNs or Infs are present, just recalculate limits with masked arrays.
        ylim = (0.95*np.min(depths), 1.05*np.max(depths))
        ax.set_ylim(ylim[0], ylim[1])
    #ax.xaxis.set_major_formatter(fsf("%.0g"))
    ax.yaxis.set_major_formatter(fsf("%.2f"))
    #ax.xaxis.set_minor_formatter(nulf())
    #ax.xaxis.set_minor_locator(aml(4))
    ax.legend()
    
    ax.tick_params(which="both", axis="both", direction="in", pad=5, labelsize=10)
    return fig, ax

def plot_eclipse_spectrum(wavelengths, depths, depth_errs, reference_wavelengths, reference_depths, reference_errs, reference_names, ylim):
    fig, ax = plt.subplots(figsize=(20,7))
    colors = ("red","blue","forestgreen","gold","orange","indigo","violet")
    if reference_depths:
        for reference_wavelength, reference_depth, reference_err, reference_name, color in zip(reference_wavelengths,reference_depths,reference_errs,reference_names,colors):
            print("Plotting reference spectrum with {} points...".format(len(reference_depth)))
            ax.scatter(reference_wavelength, reference_depth, s=20, color=color, marker="o", label=reference_name)
            ax.errorbar(reference_wavelength, reference_depth, yerr=reference_err, fmt="none", capsize=0, ecolor=color, elinewidth=2)
    # Mask NaNs so that plots can always be made even if NaNs are present.
    depths = np.ma.masked_invalid([1e6*x for x in depths])
    depth_errs = np.ma.masked_array(np.array([1e6*x for x in depth_errs]), np.ma.getmask(depths))
    wavelengths = np.ma.masked_array(np.array(wavelengths), np.ma.getmask(depths))
    ax.scatter(wavelengths, depths, s=20, color="k", marker="o", label="this reduction")
    ax.errorbar(wavelengths, depths, yerr=depth_errs, fmt="none", capsize=0, ecolor="k", elinewidth=2)

    ax.set_xlabel("wavelength [mu]", fontsize=14)
    ax.set_ylabel("eclipse depth [ppm]", fontsize=14)
    try:
        ax.set_ylim(1e6*ylim[0], 1e6*ylim[1])
    except ValueError:
        # Happens if NaNs or Infs are present, just recalculate limits with masked arrays.
        ylim = (0.95*np.min(depths), 1.05*np.max(depths))
        ax.set_ylim(ylim[0], ylim[1])
    #ax.xaxis.set_major_formatter(fsf("%.0g"))
    ax.yaxis.set_major_formatter(fsf("%.2f"))
    #ax.xaxis.set_minor_formatter(nulf())
    #ax.xaxis.set_minor_locator(aml(4))
    ax.legend()
    
    ax.tick_params(which="both", axis="both", direction="in", pad=5, labelsize=10)
    return fig, ax

def plot_multiple_spectra(thetas, errs, event_type, SDNRs, exoplanet_params, depth_type):
    fig, ax = plt.subplots(figsize=(20,7))
    orders = list(thetas.keys())
    colors = ("red","blue","forestgreen","gold","orange","indigo","violet")
    d_min = 10
    d_max = -10
    for order, color in zip(orders, colors):
        # Get transit spectrum for that order.
        order_thetas = {order:thetas[order]}
        order_thetaerrs = {order:errs[order]}
        order_SDNRs = {order:SDNRs[order]}
        wavelengths, depths, depth_errs = get_spectrum(order_thetas, order_thetaerrs, order_SDNRs, exoplanet_params, depth_type, ignore_high_SDNR=False)
        # Check mins and maxes.
        if min(depths) < d_min:
            d_min = min(depths)
        if max(depths) > d_max:
            d_max = max(depths)
        # Mask NaNs so that plots can always be made even if NaNs are present.
        depths = np.ma.masked_invalid([100*x for x in depths])
        depth_errs = np.ma.masked_array(np.array([100*x for x in depth_errs]), np.ma.getmask(depths))
        wavelengths = np.ma.masked_array(np.array(wavelengths), np.ma.getmask(depths))
        ax.scatter(wavelengths, depths, s=20, color=color, marker="o", label=order)
        ax.errorbar(wavelengths, depths, yerr=depth_errs, fmt="none", capsize=0, ecolor=color, elinewidth=2)
    
    ax.set_xlabel("wavelength [mu]", fontsize=14)
    ax.set_ylabel("{} depth [%]".format(event_type), fontsize=14)
    ax.set_ylim(100*d_min, 100*d_max)
    ax.yaxis.set_major_formatter(fsf("%.2f"))
    ax.legend()
    
    ax.tick_params(which="both", axis="both", direction="in", pad=5, labelsize=10)
    return fig, ax

def plot_SDNRs(SDNRs):
    fig, ax = plt.subplots(figsize=(10,7))

    # Not every wavelength has an SDNR attached, since some curves don't fit.
    # So have to get wavelengths from keys of SDNR dict itself.
    wavelengths = [float(wav) for wav in list(SDNRs.keys())]
    SDNRs = [SDNRs[key] for key in list(SDNRs.keys())]
    ax.scatter(wavelengths, SDNRs, s=10, color="k",marker="s")
    ax.set_xlabel("wavelength [mu]", fontsize=14)
    ax.set_ylabel("SDNR [ppm]", fontsize=14)
    ax.yaxis.set_major_formatter(fsf("%.0f"))
    ax.tick_params(which="both", axis="both", direction="in", pad=5, labelsize=10)
    return fig, ax

def plot_waterfall_fitres(timestamps, slc, slc_err, event_type, models, reses, offset=0.025):
    '''
    Produces a waterfall plot of the given slc[t,lambda] object, using the wavelengths object to assign colors to each curve.
    Also overplots the fitted light curve. To the right, plots the normalized residuals in a corresponding waterfall.
    
    :param timestamps: np array. The timestamps in days.
    :param slc: np array. The curves[Order#,t,lambda] object storing all spectroscopic transit curves. Must be median-normalized.
    :param slc_err: np array. The error on the curves objects.
    :param event_type: str. "transit" or "eclipse", the type of event observed.
    :param models: dict. Dictionary containing every model for each slc.
    :param reses: dict. Dictionary containing every model's residuals for each slc.
    :param offset: float. The spacing between each plot. Should be roughly equal to the transit depth or a little bigger.
    :return: waterfall plots of curves and reses as a fig, ax object.
    '''
    # Set curve and res offsets and grab wavelengths from models.
    offsets = np.arange(0, -np.shape(slc)[1]*offset, -offset)
    wavelengths = [i for i in list(models.keys())]

    # Open a gridspec.
    fig=plt.figure(figsize=(12, 42))
    gs = grid.GridSpec(1, 2, wspace=0.55, hspace=0.15)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[:, 1])
    # Open objects to store all the offset curves.
    flxs = []
    mods = []
    ress = []
    time = []
    interp_time = []
    wavs = []
    interp_wavs = []
    for l, wavelength in enumerate(wavelengths):
        flx = slc[:,l]+offsets[l]*np.ones_like(slc[:,l])
        res = reses[wavelength]+offsets[l]*np.ones_like(slc[:,l])
        for i, f in enumerate(flx):
            flxs.append(f)
            time.append(timestamps[i])
            wavs.append(float(wavelength))
        for i, r in enumerate(res):
            ress.append(r)
        submod_list = models[wavelength]+offsets[l]*np.ones_like(models[wavelength])
        subit_list = np.linspace(min(time),max(time),len(submod_list))
        subwav_list = [float(wavelength) for i in submod_list]
        mods.append(submod_list)
        interp_time.append(subit_list)
        interp_wavs.append(subwav_list)
        
        #SDNR = np.std(res)*10**6
        #ax1.text(1.5*timestamps[-1]+0.05, np.mean(ress), "{} ".format(wavelength) + r"$\AA$" + ", SDNR = %.0g ppm" % SDNR, color="k",
        #           horizontalalignment="left", verticalalignment="center")
    # With all of the offset curves gathered, it's time to plot them.
    ax0.scatter(time, flxs, s=7, c=wavs, cmap="jet")

    # Create cmap for the lines, which is a more involved process.
    jet = cm = plt.get_cmap('jet')
    float_wavs = [float(x) for x in wavelengths]
    cNorm = mplcolors.Normalize(vmin=min(float_wavs),vmax=max(float_wavs))
    scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=jet)
    ind = 0
    for mod, it, wav in zip(mods, interp_time, interp_wavs):
        colorVal = scalarMap.to_rgba(float_wavs[ind])
        ax0.plot(it, mod, lw=1.5, color=colorVal)
        ind += 1

    ax1.scatter(time, ress, s=7, c=wavs, cmap="jet")
    ax0.set_xlim(timestamps[0]-0.025, timestamps[-1]+0.025)
    ax1.set_xlim(timestamps[0]-0.025, timestamps[-1]+0.025)
    testflx = [x for x in flxs if str(x) != 'nan']
    testres = [x for x in ress if str(x) != 'nan']
    ax0.set_ylim(min(testflx)-offset/6, max(testflx)+offset/6)
    ax1.set_ylim(min(testres)-offset/6, max(testres)+offset/6)
    ax0.set_xlabel("time since mid-{} (days)".format(event_type))
    ax1.set_xlabel("time since mid-{} (days)".format(event_type))
    ax0.set_ylabel("normalized flux (+offsets)")
    ax1.set_ylabel("normalized residuals (+offsets)")
    return fig, ax0, ax1

def compute_depth_rprs(theta, theta_err):
    depth = theta["rp"]
    err = theta_err["rp"]
    return depth, err

def compute_depth_rprs2(theta, theta_err):
    depth = theta["rp"]**2
    err = 2*theta["rp"]*theta_err["rp"]
    return depth, err

def compute_depth_aoverlap(theta, theta_err, exoplanet_params):
    if "inc" in theta.keys():
        inc = theta["inc"]
        inc_err = theta_err["inc"]*np.pi/180
    else:
        inc = exoplanet_params["inc"]
        inc_err = 0
    if "aoR" in theta.keys():
        aoR = theta["aoR"]
        aoR_err = theta_err["aoR"]
    else:
        aoR = exoplanet_params["aoR"]
        aoR_err = 0
    bo = aoR*np.cos(inc*np.pi/180)
    bo_sq = bo**2
    bo_err = np.sqrt((np.cos(inc*np.pi/180)*aoR_err)**2 + (aoR*np.sin(inc*np.pi/180)*inc_err)**2)
    
    rs = 1
    rs_sq = rs**2

    rp = theta["rp"]
    rp_sq = rp**2
    rp_err = theta_err["rp"]

    arg_phi_1 = ((bo_sq * rs_sq) + rp_sq - rs_sq)/(2 * (bo * rs) * rp)
    arg_phi_1_err = np.sqrt((bo_err * (bo_sq * rs_sq - rp_sq + rs_sq) / (2 * bo_sq * rp * rs))**2 + 
                            (rp_err * (-bo_sq * rs_sq + rp_sq + rs_sq) / (2 * bo * rp_sq * rs))**2)

    arg_phi_2 = ((bo_sq * rs_sq) + rs_sq - rp_sq)/(2 * (bo * rs) * rs)
    arg_phi_2_err = np.sqrt((bo_err * (bo_sq * rs_sq + rp_sq - rs_sq) / (2 * bo_sq * rs_sq))**2 + 
                            (rp_err * rp / (bo * rs_sq))**2)

    phi_1 = np.arccos(arg_phi_1)  # Angle at planet centre
    phi_1_err = np.sqrt(arg_phi_1_err**2 / (1 - arg_phi_1))

    phi_2 = np.arccos(arg_phi_2)  # Angle at star centre
    phi_2_err = np.sqrt(arg_phi_2_err**2 / (1 - arg_phi_2))

    # Evaluate the overlapping area analytically
    A_overlap = (rp_sq * (phi_1 - 0.5 * np.sin(2.0 * phi_1)) +
                    rs_sq * (phi_2 - 0.5 * np.sin(2.0 * phi_2)))
    A_s = np.pi*rs_sq
    depth = A_overlap/A_s

    # The tedious process of computing the depth error
    '''
    dphi_1 = (rs_sq*(bo_sq-1)-rp_sq)/(rp*np.sqrt(rs_sq*rs_sq*(-(bo_sq-1)**2)+2*rs_sq*(bo_sq+1)*rp_sq-rp_sq*rp_sq))
    dphi_2 = 2*rp/(np.sqrt(rs_sq*rs_sq*(-(bo_sq-1)**2)+2*rs_sq*(bo_sq+1)*rp_sq-rp_sq*rp_sq))
    err = (2*rp*(phi_1-0.5*np.sin(2 * phi_1))
           + rp_sq*dphi_1*(1-np.cos(2 * phi_1))
           + rs_sq*dphi_2*(1-np.cos(2 * phi_2)))*theta_err["rp"]
    '''
    dAdrp = 2*rp*(phi_1 - 0.5 * np.sin(2.0 * phi_1))/A_s
    dAdphi_1 = rp_sq*(1 - np.cos(2.0 * phi_1))/A_s
    dAdphi_2 = rs_sq*(1 - np.cos(2.0 * phi_2))/A_s

    err = np.sqrt((dAdrp*rp_err)**2 + (dAdphi_1*phi_1_err)**2+(dAdphi_2*phi_2_err)**2)
    return depth, err

def compute_depth_fpfs(theta, theta_err):
    depth = theta["fp"]
    err = theta_err["fp"]
    return depth, err

def NRS_uncal_file_sorter(input_dir:str, output_dir:str):
    """Sort exposures between NRS1 and NRS2

    Parameters
    ----------
    input_dir       string
                    path to parent directory with both NRS1 and NRS2 files 
                    in directory or subdirectories

    output_dir      string
                    path to parent data directory where all uncal files 
                    will be located
    """

    ### Ensure directories for sorted data exists
    uncal = f"{output_dir}/uncal/"
    nrs1 = f"{uncal}/nrs1/"
    nrs2 = f"{uncal}/nrs2/"

    if os.path.exists(uncal) != True:
        os.makedirs(uncal)
    if os.path.exists(nrs1) != True:
        os.makedirs(nrs1)
    if os.path.exists(nrs2) != True:
        os.makedirs(nrs2)

    for files in glob.glob(f"{input_dir}/**/*nrs1_uncal.fits", recursive=True):
        fname = os.path.basename(files)
        shutil.move(files, f"{nrs1}{fname}")

    for files in glob.glob(f"{input_dir}/**/*nrs2_uncal.fits", recursive=True):
        fname = os.path.basename(files)
        shutil.move(files, f"{nrs2}{fname}")