import juniper as Juniper
import numpy as np
import glob
import os
os.environ['CRDS_PATH'] = './crds_cache/jwst_ops' # set path to CRDS cache if you already have one

### HELLO, BEFORE YOU RUN: ###
'''
This is a test script that demos an execution of the Juniper module.
The way this script is organised is also a handy way for you to organise your own script!
'''
### OKAY, LET'S GET TO IT ###
# Select target name and a name for this run to distinguish it from other runs.
target = 'WD1856b'
run_name = 'run1'

# Print to confirm operations have started on the right target.
print("Operating on target {}...".format(target))

# The event type, "transit" or "eclipse". Tells us which type of depth to use.
event_type = "transit"

# The locations of the data files for Stage 1 processing. Adjust to fit your own computer! Also sets the locations for where outputs will be saved to.
stage1_filepaths = ["./path/to/jw02358030001_04101_00001-seg001_nrs1_uncal.fits",]
stage1_outfiles = [target,]
stage1_outdir = "./{}/processed/rateints_{}".format(target, run_name)
        
stage2_filesdir = "./{}/processed/rateints_{}".format(target, run_name)
stage2_outfiles = [target,]
stage2_outdir = "./{}/processed/calints_{}".format(target, run_name)

stage3_filesdir = "./{}/processed/calints_{}".format(target, run_name)
stage3_outdir = "./{}/processed/postprocessed_{}".format(target, run_name)

stage4_filesdir = "./{}/processed/postprocessed_{}".format(target, run_name)
stage4_outdir = "./{}/processed/extraction_{}".format(target, run_name)

stage5_filesdir = "./{}/processed/extraction/output_txts_extraction_{}".format(target, run_name)
stage5_outdir ="./{}/processed/fits" .format(target, run_name)

# Epoch of mid-transit/eclipse.
epoch = 60061.50686491065 # BJD_TDB

# The exoplanet_params dictionary sets up initial guesses and fixed values for fitting with Batman.
# If you are fitting an eclipse, you will need to enter fp and t_secondary. You also need rp, but you do not need t0.
exoplanet_params = {"rp":8,
                    "t0":0,
                    "period":1.407939217,       # Fixed from reference
                    "aoR":330.71854,            # FIREFLy 2023
                    "inc":88.78157208,          # FIREFLy 2023
                    "ecc":0,
                    "lop":90,
                    "offset":0.0}

# The priors dict lists all parameters but if you fix any of them, you can ignore the entry in this dict.
# If you are fitting an eclipse, you will need to enter fp and t_secondary. You also need rp, but you do not need t0.
# If your priors are uniform, this is [lower bound, upper bound]. It is necessary that lower bound < upper bound!
# If your priors are gaussian, this is [mean, sigma]. sigma must be positive!
priors_dict = {"t0":[0.0, 0.001],
               "period":[1.407939217, 0.000000016],
               "aoR":[336, 3*14],
               "inc":[88.778, 0.059],
               "ecc":[0, 0.8],
               "lop":[0, 90],
               "offset":[0.0,0.0]}

# Decide which priors type you use.
priors_type = "gaussian"

# Initialize the systematics (a,b) for linear trend a + b*t.
systematics = (0,1)

# Metallicity, effective T, and log g for your star, which determines which Phoenix model will be used. For WD 1856, a custom model is supplied, so we set this to None.
stellar_params = None # (M_H, Teff, logg)

# If you have a custom model, this is where it is storedd.
path_to_custom_LD_model = "./path/to/blouin_WD1856_LDmodel/Imu_bestfitJWST.txt"

# Set up your LD model, including parameters type (quadratic, nonlinear, etc.), custom model path and stellar params (where applicable), and initial guess (only relevant if you are fitting for LD coeffs).
limb_darkening_model = {"model_type":"quadratic",
                        "custom_model":path_to_custom_LD_model,
                        "stellar_params":stellar_params,
                        "initial_guess":[0.0,0.0,0.0,0.0]}

# Whether or not you are using exoticLD. You might need exoticLD to use Blouin's model.
exoticLD = {"available":True,
            "ld_data_path":"./path/to/exotic_ld_data-3.1.2",    # you may not need this data since we are using Blouin's model.
            "ld_grid":"custom",
            "custom_model_path":path_to_custom_LD_model,
            "ld_interpolate_type":'trilinear'}
    
# Which parameters are held fixed during white light curve (WLC) fitting. Generally, the WLC is used to fit for the mid-transit/eclipse time, a/R*, inclination, etc. as well as a broad-band depth rp.
fixed_param_WLC = {"rp":False,
                   "fp":True,
                   "LD_coeffs":True,
                   "t0":False,
                   "t_secondary":True,
                   "period":True,
                   "aoR":False,
                   "inc":False,
                   "ecc":True,
                   "lop":True,
                   "offset":True}

# Which parameters are held fixed during spectroscopic light curve (SLC) fitting. For the SLCs, you lock in system parameters but you let the radius vary.
fixed_param_SLC = {"rp":False,
                   "fp":True,
                   "LD_coeffs":True,
                   "t0":True,
                   "t_secondary":True,
                   "period":True,
                   "aoR":True,
                   "inc":True,
                   "ecc":True,
                   "lop":True,
                   "offset":True}

# Whether to carry out each stage. The only variable you need to chanfe for each of these dicts is "skip". If you want to do that stage, enter False for "skip".
stage1 = {"skip":True,
          "filepaths":stage1_filepaths,
          "outfiles":stage1_outfiles,
          "outdir":stage1_outdir}

stage2 = {"skip":True,
          "filesdir":stage2_filesdir,
          "outfiles":stage2_outfiles,
          "outdir":stage2_outdir}

stage3 = {"skip":False,
          "filesdir":stage3_filesdir,
          "outdir":stage3_outdir}

stage4 = {"skip":False,
          "filesdir":stage4_filesdir,
          "outdir":stage4_outdir}

stage5 = {"skip":False,
          "filesdir":stage5_filesdir,
          "outdir":stage5_outdir,
          "exoplanet_params":exoplanet_params}


# And now it gets started. The next five "if not" statements will initialize the stages you want to run. They also have tunable parameters, so give these a look as well!
# There *should* be documentation in each StageX.py file as to what each of these variables mean, but I may not have gotten around to writing it all yet.
# I have left it set up to recreate Juniper PRISM 2023, so for the time being you don't yet need to know how this stuff works.
# But obviously for your own data, you will need to know that stuff.

if not stage1["skip"]:
    Juniper.Stage1.doStage1(stage1["filepaths"], stage1["outfiles"], stage1["outdir"],
                            group_scale={"skip":False},
                            dq_init={"skip":False},
                            saturation={"skip":False},
                            superbias={"skip":False},
                            refpix={"skip":False},
                            linearity={"skip":False},
                            dark_current={"skip":False},
                            jump={"skip":True, "rejection_threshold":15},
                            ramp_fit={"skip":False},
                            gain_scale={"skip":False},
                            one_over_f={"skip":False, "bckg_rows":[0,1,2,3,4,5,-1,-2,-3,-4,-5,-6], "sigma":3.0, "kernel":(7,1), "show":True}
                            )

if not stage2["skip"]:
    Juniper.Stage2.doStage2(stage2["filesdir"], stage2["outfiles"], stage2["outdir"],
                            assign_wcs={"skip":False},
                            extract_2d={"skip":False},
                            srctype={"skip":False},
                            wavecorr={"skip":False},
                            flat_field={"skip":True},
                            pathloss={"skip":True},
                            photom={"skip":True},
                            resample_spec={"skip":True},
                            extract_1d={"skip":True}
                            )

if not stage3["skip"]:
    Juniper.Stage3.doStage3(stage3["filesdir"], stage3["outdir"],
                            trace_aperture={"hcut1":0,
                                            "hcut2":32,
                                            "vcut1":0,
                                            "vcut2":431},
                            frames_to_reject = [],
                            identified_bad_pixels=[],
                            loss_stats_step={"skip":False},
                            mask_flagged_pixels={"skip":False},
                            iteration_outlier_removal={"skip":False, "n":2, "sigma":6.5},
                            spatialfilter_outlier_removal={"skip":True, "sigma":5.25, "kernel":(1,41)},
                            laplacianfilter_outlier_removal={"skip":True, "sigma":50},
                            second_bckg_subtract={"skip":False,"bckg_rows":[0,1,2,3,4,5,-6,-5,-4,-3,-2,-1], "sigma":3.0},
                            track_source_location={"skip":True,"reject_disper":True,"reject_spatial":True})

if not stage4["skip"]:
    Juniper.Stage4.doStage4(stage4["filesdir"], stage4["outdir"],
                            trace_aperture={"hcut1":11,
                                            "hcut2":16,
                                            "vcut1":0,
                                            "vcut2":431},
                            mask_unstable_pixels={"skip":True,
                                                  "threshold":1.0},
                            extract_light_curves={"skip":False,
                                                  "binmode":"columns",
                                                  "columns":3,
                                                  "min_wav":0.5,
                                                  "max_wav":5.5,
                                                  "wavbins":[],
                                                  "ext_type":"medframe",
                                                  "badcol_threshold":8.0,
                                                  "omit_columns":[],}
                            median_normalize_curves={"skip":False,
                                                     "event_type":event_type},
                            sigma_clip_curves={"skip":True,
                                                "b":100,
                                                "clip_at":5},
                            fix_mirror_tilts={"skip":True,
                                              "threshold":0.002,
                                              "known_index":270},
                            fix_transit_times={"skip":False,
                                               "epoch":epoch},
                            plot_light_curves={"skip":False,
                                               "event_type":event_type},
                            save_light_curves={"skip":False})

if not stage5["skip"]:
    Juniper.Stage5.doStage5(stage5["filesdir"], stage5["outdir"], event_type=event_type, reject_threshold=3, raise_alarm=10,
                            LSQfit_WLC={"do":True,
                                        "exoplanet_params":exoplanet_params,
                                        "systematics":systematics,
                                        "limb_darkening_model":limb_darkening_model,
                                        "fixed_param":fixed_param_WLC,
                                        "priors_dict":priors_dict,
                                        "priors_type":priors_type,
                                        "exoticLD":exoticLD},
                            MCMCfit_WLC={"do":True,
                                         "exoplanet_params":exoplanet_params,
                                         "systematics":systematics,
                                         "limb_darkening_model":limb_darkening_model,
                                         "fixed_param":fixed_param_WLC,
                                         "priors_dict":priors_dict,
                                         "exoticLD":exoticLD,
                                         "priors_type":priors_type,
                                         "N_walkers":80,
                                         "N_steps":5000,
                                         "est_err":0.001},
                            LSQfit_spec={"do":True,
                                         "exoplanet_params":exoplanet_params,
                                         "systematics":systematics,
                                         "limb_darkening_model":limb_darkening_model,
                                         "fixed_param":fixed_param_SLC,
                                         "priors_dict":priors_dict,
                                         "priors_type":priors_type,
                                         "exoticLD":exoticLD},
                            MCMCfit_spec={"do":True,
                                          "exoplanet_params":exoplanet_params,
                                         "systematics":systematics,
                                         "limb_darkening_model":limb_darkening_model,
                                         "fixed_param":fixed_param_SLC,
                                         "priors_dict":priors_dict,
                                         "exoticLD":exoticLD,
                                         "priors_type":priors_type,
                                         "N_walkers":32,
                                         "N_steps":3000},
                            save_plots={"do":True,
                                        "fit_and_residuals":True,
                                        "posteriors":True,
                                        "corner":True,
                                        "chains":True,
                                        "spectrum":True},
                            reference_spectra={"reference_wavelengths":[],
                                               "reference_depths":[],
                                               "reference_errs":[],
                                               "reference_names":[]}
                           )