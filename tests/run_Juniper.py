import juniper as Juniper
import glob
import os
os.environ['CRDS_PATH'] = './crds_cache/jwst_ops' # set path to CRDS cache if you already have one

### HELLO, BEFORE YOU RUN: ###
'''
This is a test script that demos an execution of the Juniper module.
The way this script is organised is also a handy way for you to organise your own script!
'''
### OKAY, LET'S GET TO IT ###
target = 'WASP-39b'

if target == 'WD1856b':
    # The event type, "transit" or "eclipse".
    event_type = "transit"
    # The locations of the data files.
    stage1_filepaths = ["/Users/abby/code_dev/jwst/WD_1856/JWST_WD1856-selected/jw02358030001_04101_00001-seg001_nrs1_uncal.fits",]
    stage1_outfiles = ["WD1856b",]
    stage1_outdir = "./processed/rateints"
    
    stage2_filesdir = "./processed/rateints"
    stage2_outfiles = ["WD1856b",]
    stage2_outdir = "./processed/calints"

    stage3_filesdir = "./processed/calints"
    stage3_outdir = "./processed/postprocessed"

    stage4_filesdir = "./processed/postprocessed"
    stage4_outdir = "./processed/extraction"

    stage5_filesdir = "./processed/extraction/output_txts_extraction"
    stage5_outdir ="./processed/fits" 

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
    priors_type = "gaussian"
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
    
if target == 'WASP-39b':
    # The event type, "transit" or "eclipse".
    event_type = "transit"
    # The locations of the data files.
    selecting_for = 'nrs1'
    stage1_filepaths = []
    stage1_outfiles = []
    dirs = os.listdir('/Users/abby/opt/anaconda3/github_repos/juniper/tests/WASP39_ERS')
    for dir in dirs:
        if dir != '.DS_Store':
            path = os.path.join('/Users/abby/opt/anaconda3/github_repos/juniper/tests/WASP39_ERS',dir)
            files = glob.glob(os.path.join(path,'*.fits'))
            for file in files:
                if selecting_for in file:
                    stage1_filepaths.append(file)
                    outfile_name = str.replace(str.split(file,'/')[-1],'_uncal.fits','')
                    stage1_outfiles.append(outfile_name)
    stage1_filepaths = sorted(stage1_filepaths)
    stage1_outfiles = sorted(stage1_outfiles)
    stage1_outdir = "./WASP39_ERS/processed_{}/rateints".format(selecting_for)

    stage2_filesdir = stage1_outdir
    stage2_outfiles = stage1_outfiles
    stage2_outdir = "./WASP39_ERS/processed_{}/calints".format(selecting_for)

    stage3_filesdir = stage2_outdir
    stage3_outdir = "./WASP39_ERS/processed_{}/postprocessed".format(selecting_for)

    stage4_filesdir = stage3_outdir
    stage4_outdir = "./WASP39_ERS/processed_{}/extraction".format(selecting_for)

    stage5_filesdir = os.path.join(stage4_outdir,'output_txts_extraction')
    stage5_outdir ="./WASP39_ERS/processed_{}/fits" .format(selecting_for)

    # All the parameters needed for fitting for WASP-39 b.
    epoch = 59791.115 # BJD_TDB
    exoplanet_params = {"rp":0.1457,
                        "t0":0,
                        "period":4.0552765,
                        "aoR":11.37,
                        "inc":87.75,
                        "ecc":0,
                        "lop":90,
                        "offset":0.0}
    priors_dict = {"t0":[0.0, 0.005],
                "period":[4.0552765, 0.0000035],
                "aoR":[11.37, 0.24],
                "inc":[87.75, 0.30],
                "ecc":[0, 0.8],
                "lop":[0, 90],
                "offset":[0.2,0.25]}
    priors_type = "gaussian"
    systematics = (0,1)
    stellar_params = (-0.12, 5400, 4.45)
    limb_darkening_model = {"model_type":"quadratic",
                            "custom_model":None,
                            "stellar_params":stellar_params,
                            "initial_guess":[0.0,0.0,0.0,0.0]}
    exoticLD = {"available":True,
                "ld_data_path":"/Users/abby/opt/anaconda3/exotic_ld_data-3.1.2",
                "ld_grid":"stagger",
                "custom_model_path":None,
                "ld_interpolate_type":'trilinear'}
    
if target == 'WASP-52b':
    # The event type, "transit" or "eclipse".
    event_type = "eclipse"
    # The locations of the data files.
    stage1_filepaths = []
    stage1_outfiles = []
    dir = '/Users/abby/opt/anaconda3/github_repos/juniper/tests/WASP52_GTO1224/JWST_WASP52_Prism-selected'
    files = glob.glob(os.path.join(dir,'*.fits'))
    for file in files:
        stage1_filepaths.append(file)
        outfile_name = str.replace(str.split(file,'/')[-1],'_uncal.fits','')
        stage1_outfiles.append(outfile_name)
    stage1_filepaths = sorted(stage1_filepaths)
    stage1_outfiles = sorted(stage1_outfiles)
    stage1_outdir = "./WASP52_GTO1224/processed/rateints"

    stage2_filesdir = stage1_outdir
    stage2_outfiles = stage1_outfiles
    stage2_outdir = "./WASP52_GTO1224/processed/calints"

    stage3_filesdir = stage2_outdir
    stage3_outdir = "./WASP52_GTO1224/processed/postprocessed"

    stage4_filesdir = stage3_outdir
    stage4_outdir = "./WASP52_GTO1224/processed/extraction"

    stage5_filesdir = os.path.join(stage4_outdir,'output_txts_extraction')
    stage5_outdir ="./WASP52_GTO1224/processed/fits"

    # All the parameters needed for fitting for WASP-52 b.
    epoch = 59791.115+314.4 # BJD_TDB
    exoplanet_params = {"rp":0.1639,
                        "fp":0.002,
                        "t_secondary":0,
                        "period":1.749779800,
                        "aoR":7.22,
                        "inc":85.17,
                        "ecc":0,
                        "lop":90,
                        "offset":0.0}
    priors_dict = {"t_secondary":[0.0, 0.005],
                "period":[1.749779800, 0.0],
                "aoR":[7.22, 0.07],
                "inc":[85.17, 0.13],
                "ecc":[0, 0.8],
                "lop":[0, 90],
                "offset":[0.2,0.25]}
    priors_type = "gaussian"
    systematics = (0,1)
    stellar_params = (0.03, 5000, 4.5)
    limb_darkening_model = {"model_type":"quadratic",
                            "custom_model":None,
                            "stellar_params":stellar_params,
                            "initial_guess":[0.0,0.0]} # locked to [0.0, 0.0] because this is an eclipse!
    exoticLD = {"available":False,
                "ld_data_path":"/Users/abby/opt/anaconda3/exotic_ld_data-3.1.2",
                "ld_grid":"stagger",
                "custom_model_path":None,
                "ld_interpolate_type":'trilinear'}

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

print("Operating on target {}...".format(target))

stage1 = {"skip":False,
          "filepaths":stage1_filepaths,
          "outfiles":stage1_outfiles,
          "outdir":stage1_outdir}

stage2 = {"skip":False,
          "filesdir":stage2_filesdir,
          "outfiles":stage2_outfiles,
          "outdir":stage2_outdir}

stage3 = {"skip":True,
          "filesdir":stage3_filesdir,
          "outdir":stage3_outdir}

stage4 = {"skip":True,
          "filesdir":stage4_filesdir,
          "outdir":stage4_outdir}

stage5 = {"skip":True,
          "filesdir":stage5_filesdir,
          "outdir":stage5_outdir,
          "exoplanet_params":exoplanet_params}

if not stage1["skip"]:
    Juniper.Stage1.doStage1(stage1["filepaths"], stage1["outfiles"], stage1["outdir"],
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
                            one_over_f={"skip":False, "bckg_rows":[0,1,2,3,4,5,-1,-2,-3,-4,-5,-6], "sigma":2.5, "kernel":(3,1), "show":False}
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

# For G395H, 15-7, 15+7+1, 0, 1271/2047.
# For PRISM, 7-4, 7+4+1, 0, 431.
if not stage3["skip"]:
    Juniper.Stage3.doStage3(stage3["filesdir"], stage3["outdir"],
                            trace_aperture={"hcut1":15-7,
                                            "hcut2":15+7+1,
                                            "vcut1":0,
                                            "vcut2":1271},
                            frames_to_reject = [],
                            identified_bad_pixels=[],#[(96,10),],
                            loss_stats_step={"skip":False},
                            mask_flagged_pixels={"skip":False},
                            iteration_outlier_removal={"skip":False, "n":2, "sigma":3.5},
                            spatialfilter_outlier_removal={"skip":True, "sigma":3, "kernel":(1,15)},
                            laplacianfilter_outlier_removal={"skip":True, "sigma":50},
                            second_bckg_subtract={"skip":False,"bckg_rows":[0,1,2,-3,-2,-1], "sigma":3},
                            track_source_location={"skip":True,"reject_disper":True,"reject_spatial":True})
    
if not stage4["skip"]:
    Juniper.Stage4.doStage4(stage4["filesdir"], stage4["outdir"],
                            trace_aperture={"hcut1":15-7,
                                            "hcut2":15+7+1,
                                            "vcut1":0,
                                            "vcut2":1271},
                            mask_unstable_pixels={"skip":True,
                                                  "threshold":1.0},
                            extract_light_curves={"skip":False,
                                                  "binmode":"columns",
                                                  "columns":10,
                                                  "min_wav":0.5,
                                                  "max_wav":5.5,
                                                  "wavbins":[],
                                                  "ext_type":"medframe",
                                                  "badcol_threshold":8.0},
                            median_normalize_curves={"skip":False,
                                                     "event_type":event_type},
                            sigma_clip_curves={"skip":False,
                                                "b":100,
                                                "clip_at":3},
                            fix_mirror_tilts={"skip":False,
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