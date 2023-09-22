import juniper as Juniper
import os
os.environ['CRDS_PATH'] = './crds_cache/jwst_ops' # set path to CRDS cache if you already have one

### HELLO, BEFORE YOU RUN: ###
'''
This is a test script that demos an execution of the Juniper module.
The way this script is organised is also a handy way for you to organise your own script!
'''
### OKAY, LET'S GET TO IT ###

# All the parameters needed for fitting for WD 1856 b.
epoch = 60061.50686491065 # BJD_TDB
exoplanet_params = {"rp":8,#7.28,
                    "t0":0,
                    "period":1.407939217,
                    "aoR":339.247070534106, #340.00199310598543,
                    "inc":88.74513993900113, #88.74395537223828,
                    "ecc":0,
                    "lop":90,
                    "offset":0.0}
priors_dict = {"t0":[0.0, 0.001],
               "period":[1.407939217, 0.000000016],
               "aoR":[336, 3*14],
               "inc":[88.778, 0.059],
               "ecc":[0, 0.8],
               "lop":[0, 90],
               "offset":[0.2,0.25]}
systematics = (0,1)
stellar_params = None # (M_H, Teff, logg)
path_to_custom_LD_model = "/Users/abby/code_dev/jwst/WD_1856/blouin_WD1856_LDmodel/Imu_bestfitJWST.txt"
limb_darkening_model = {"model_type":"quadratic",
                        "custom_model":path_to_custom_LD_model,
                        "stellar_params":stellar_params,
                        "initial_guess":[0.0,0.0,0.0,0.0]}
exoticLD = {"available":True,
            "ld_data_path":"/Users/abby/opt/anaconda3/exotic_ld_data-3.1.2",
            "ld_grid":"custom",
            "custom_model_path":path_to_custom_LD_model,
            "ld_interpolate_type":'trilinear'}
priors_type = "gaussian"

fixed_param_WLC = {"LD_coeffs":True,
                   "t0":False,
                   "period":True,
                   "aoR":True,
                   "inc":True,
                   "ecc":True,
                   "lop":True,
                   "offset":True}

fixed_param_SLC = {"LD_coeffs":True,
                   "t0":True,
                   "period":True,
                   "aoR":True,
                   "inc":True,
                   "ecc":True,
                   "lop":True,
                   "offset":True}

stage1 = {"skip":False,
          "filepath":"/Users/abby/code_dev/jwst/WD_1856/JWST_WD1856-selected/jw02358030001_04101_00001-seg001_nrs1_uncal.fits",
          "outfile":"WD1856b",
          "outdir":"./processed/rateints"}

stage2 = {"skip":True,
          "filepath":"./processed/rateints/WD1856b_rateints.fits",
          "outfile":"WD1856b",
          "outdir":"./processed/calints"}

stage3 = {"skip":True,
          "filesdir":"./processed/calints",
          "outdir":"./processed/postprocessed"}

stage4 = {"skip":True,
          "filesdir":"./processed/postprocessed",
          "outdir":"./processed/extraction_revised"}

stage5 = {"skip":True,
          "filesdir":"./processed/extraction_revised/output_txts_extraction",
          "outdir":"./processed/fits_revised",
          "exoplanet_params":exoplanet_params}

if not stage1["skip"]:
    Juniper.Stage1.doStage1(stage1["filepath"], stage1["outfile"], stage1["outdir"],
                            group_scale={"skip":False},
                            dq_init={"skip":False},
                            saturation={"skip":False},
                            superbias={"skip":False},
                            refpix={"skip":False},
                            linearity={"skip":False},
                            dark_current={"skip":False},
                            jump={"skip":True},
                            ramp_fit={"skip":False},
                            gain_scale={"skip":False},
                            one_over_f={"skip":False, "bckg_rows":[1,2,3,4,5,6,-1,-2,-3,-4,-5,-6], "sigma":3.0, "kernel":(5,1), "show":False}
                            )

if not stage2["skip"]:
    Juniper.Stage2.doStage2(stage2["filepath"], stage2["outfile"], stage2["outdir"],
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
                            trace_aperture={"hcut1":10,
                                            "hcut2":15,
                                            "vcut1":0,
                                            "vcut2":432},
                            frames_to_reject = [],
                            identified_bad_pixels=[(96,10),],
                            loss_stats_step={"skip":False},
                            mask_flagged_pixels={"skip":False},
                            iteration_outlier_removal={"skip":False, "n":2, "sigma":6.5},
                            spatialfilter_outlier_removal={"skip":True, "sigma":3, "kernel":(1,15)},
                            laplacianfilter_outlier_removal={"skip":True, "sigma":50},
                            second_bckg_subtract={"skip":False,"bckg_rows":[0,1,2,-2,-1], "sigma":3},
                            track_source_location={"skip":True,"reject_disper":True,"reject_spatial":True})
    
if not stage4["skip"]:
    Juniper.Stage4.doStage4(stage4["filesdir"], stage4["outdir"],
                            trace_aperture={"hcut1":9,
                                            "hcut2":16,
                                            "vcut1":0,
                                            "vcut2":432},
                            mask_unstable_pixels={"skip":True,
                                                  "threshold":1.0},
                            extract_light_curves={"skip":False,
                                                  "binmode":"columns",
                                                  "columns":3,
                                                  "min_wav":0.5,
                                                  "max_wav":5.5,
                                                  "wavbins":[],
                                                  "ext_type":"medframe"},
                            median_normalize_curves={"skip":False},
                            sigma_clip_curves={"skip":True,
                                                "b":100,
                                                "clip_at":5},
                            fix_transit_times={"skip":False,
                                                "epoch":epoch},
                            plot_light_curves={"skip":False},
                            save_light_curves={"skip":False})

if not stage5["skip"]:
    Juniper.Stage5.doStage5(stage5["filesdir"], stage5["outdir"], reject_threshold=3, raise_alarm=10,
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
                                         "N_steps":10000,
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