import os

def write_config(config_dict, run_name, stage, outdir):
    """Unpacks a dictionary and writes it out to a config file.
    Credit V.A. Boehm from ExoTiC-UVIS.

    Args:
        config_dict (dict): The dictionary used to guide the execution of a Stage of Juniper.
        run_name (str): The name of this run.
        stage (int): From 1 to 6. Which Stage was executed, which sets the template of the config file.
        outdir (str): The path to where the config file is to be stored.
    """
    # Get correct print info.
    if stage == 1:
        header, subsection_headers, subsection_keys, subsection_comments = Stage1_info()
    if stage == 2:
        header, subsection_headers, subsection_keys, subsection_comments = Stage2_info()
    if stage == 3:
        header, subsection_headers, subsection_keys, subsection_comments = Stage3_info()
    if stage == 4:
        header, subsection_headers, subsection_keys, subsection_comments = Stage4_info()
    if stage == 5:
        header, subsection_headers, subsection_keys, subsection_comments = Stage5_info()
    if stage == 6:
        header, subsection_headers, subsection_keys, subsection_comments = Stage6_info()
    
    # And write.
    with open(os.path.join(outdir,"stage_{}_{}_juniper.berry".format(stage, run_name)), mode='w') as f:
        print("Writing config file for Stage {}...".format(stage))
        # First, write the overall file header.
        f.write(header)
        f.write('\n\n')

        # Then, start parsing each step out (Setup, Step 1, Step 2, etc.).
        subsections = list(subsection_keys.keys())
        for i, subsection in enumerate(subsections):
            # Write the step name.
            f.write(subsection_headers[i])
            f.write('\n')

            # For every keyword and comment in that step...
            for j, keyword in enumerate(subsection_keys[subsection]):
                # Write the keyword, its stringed value, and the comment.
                try:
                    value = str(config_dict[keyword])
                    f.write("{0:<15} {1:<100} {2:}\n".format(keyword, value, subsection_comments[subsection][j]))
                except IndexError:
                    print(subsection, subsection_keys[subsection])
                    print("Index error here!")
            # A space between this step and the next step.
            f.write('\n')
        # Declare the file over.
        f.write("# ENDPARSE")
    print("Config file written.")
            

def Stage1_info():
    '''
    Specific keys and subsections for Stage 1.
    '''
    header = "# Juniper config file for launching Stage 1: JWST Detector1Pipeline and 1/f Corrections"

    subsection_headers = ["# Setup for Stage 1",
                          "# Step 1: JWST Detector1Pipeline front-end",
                          "# Step 2: Group-level background subtraction",
                          "# Step 3: JWST Detector1Pipeline back-end",
                          "# Step 4: NSClean",
                          ]
    
    subsection_keys = {"Setup":["toplevel_dir",
                                "verbose",
                                "show_plots",
                                "save_plots"],
                       "Step 1":["files_loc",
                                 "files_rename",],
                       "Step 2":["do_glbs",],
                       "Step 3":[],
                       "Step 4":["do_NSCle"],
                       }
    
    subsection_comments = {"Setup":["# Directory where you want your files to be stored after Stage 1 has run. This is where the *rateints.fits files will be stored.",
                                    "# Int from 0 to 2. 0 = print nothing. 1 = print some statements. 2 = print every action.",
                                    "# Int from 0 to 2. 0 = show nothing. 1 = show some plots. 2 = show all plots.",
                                    "# Int from 0 to 2. 0 = save nothing. 1 = save some plots. 2 = save all plots.",],
                           "Step 1":["# str. This is where the *uncal.fits files that you want to operate on are. Download them before Stage 1!",
                                     "# str. What you want to rename the files to. Can be None to keep the default name. Suggestions include naming the files after the target planet and program ID (e.g. 'PLANET-B_ID12345').",],
                           "Step 2":["# bool. Whether to perform this step. Strongly recommend you do - glbs can reduce scatter by thousands of ppm!",],
                           "Step 3":[],
                           "Step 4":["# bool. Whether to perform this step. Cuts down further on systematic noise.",],
                           }
    return header, subsection_headers, subsection_keys, subsection_comments

def Stage2_info():
    '''
    Specific keys and subsections for Stage 1.
    '''
    header = "# ExoTiC-UVIS config file for launching Stage 1: Reduction"

    subsection_headers = ["# Setup for Stage 1",
                          "# Step 1: Read in the data",
                          "# Step 2: Reject cosmic rays with time iteration\n# Step 2a: Fixed iteration parameters",
                          "# Step 2b: Free iteration parameters",
                          "# Step 2c: Sigma clip parameters",
                          "# Step 3: Reject hot pixels with spatial detection\n# Step 3a: Laplacian Edge Detection parameters",
                          "# Step 3b: Spatial smoothing parameters",
                          "# Step 4: Background subtraction\n# Step 4a: full frame mode or median background subtraction",
                          "# Step 4b: corners mode or median background subtraction",
                          "# Step 4c: Column-by-column background subtraction",
                          "# Step 4d: Pagul+ 2023 background subtraction",
                          "# Step 5: Displacement estimation\n# Step 5a: Source center-of-mass tracking",
                          "# Step 5b: Background star tracking",
                          "# Step 6: Save outputs",]
    
    subsection_keys = {"Setup":["toplevel_dir",
                                "run_name",
                                "verbose",
                                "show_plots",
                                "save_plots"],
                       "Step 1":["skip_first_fm",
                                 "skip_first_or",],
                       "Step 2a":["do_fixed_iter",
                                 "fixed_sigmas",
                                 "replacement",],
                       "Step 2b":["do_free_iter",
                                  "free_sigma"],
                       "Step 2c":["do_sigma_clip",],
                       "Step 3a":["do_led",
                                  "led_threshold",
                                  "led_factor",
                                  "led_n",
                                  "fine_structure",
                                  "contrast_factor",],
                       "Step 3b":["do_smooth",],
                       "Step 4a":["do_full_frame",
                                  "full_value",],
                       "Step 4b":["do_corners",
                                  "bounds",
                                  "corners_value"],
                       "Step 4c":["do_column",
                                  "rows",
                                  "col_sigma",],
                       "Step 4d":["do_Pagul23",
                                  "path_to_Pagul23",
                                  "mask_parameter",
                                  "median_columns",],
                       "Step 5a":["do_0thtracking",],
                       "Step 5b":["do_bkg_stars",
                                  "bkg_stars_loc",],
                       "Step 6":["do_save",],
                       }
    
    subsection_comments = {"Setup":["# Directory where your Stage 0 files are stored. This folder should contain the specimages/, directimages/, etc. folders with your data.",
                                    "# Str. This is the name of the current run. It can be anything that does not contain spaces or special characters (e.g. $, %, @, etc.).",
                                    "# Int from 0 to 2. 0 = print nothing. 1 = print some statements. 2 = print every action.",
                                    "# Int from 0 to 2. 0 = show nothing. 1 = show some plots. 2 = show all plots.",
                                    "# Int from 0 to 2. 0 = save nothing. 1 = save some plots. 2 = save all plots.",],
                           "Step 1":["# Bool. If True, ignores all first frames in each orbit.",
                                     "# Bool. If True, ignores all frames in the first orbit.",],
                           "Step 2a":["# Bool. Whether to use fixed iteration rejection to clean the timeseries.",
                                      "# lst of float. The sigma to reject outliers at in each iteration. The length of the list is the number of iterations.",
                                      "# int or None. If int, replaces flagged outliers with the median of values within +/-replacement indices of the outlier. If None, uses the median of the whole timeseries instead.",],
                           "Step 2b":["# Bool. Whether to use free iteration rejection to clean the timeseries.",
                                      "# float. The sigma to reject outliers at in each iteration. Iterates over each pixel's timeseries until no outliers at this sigma level are found.",],
                           "Step 2c":["# Bool. Whether to use sigma clipping rejection to clean the timeseries.",],
                           "Step 3a":["# Bool. Whether to use Laplacian Edge Detection rejection to clean the frames.",
                                      "# Float. The threshold parameter at which to kick outliers in LED. The lower the number, the more values will be replaced.",
                                      "# Int. The subsampling factor. Minimum value 2. Higher values increase computation time but aren't expected to yield much improvement in rejection.",
                                      "# Int. Number of times to do LED on each frame.",
                                      "# Bool. Whether to build a fine structure model, which can protect narrow bright features like traces from LED.",
                                      "# Float. If fine_structure is True, acts as the led_threshold for the fine structure step.",],
                           "Step 3b":["# Bool. Whether to use spatial smoothing rejection to clean the frames.",],
                           "Step 4a":["# Bool. Whether to subtract the background using the full frame mode or median.",
                                      "# Str. The value to extract from the histogram. Options are 'coarse', 'Gaussian', or 'median'.",],
                           "Step 4b":["# Bool. Whether to subtract the background using the frame corners mode or median.",
                                      "# Lst of lst of float. The region from which the background values will be extracted.",
                                      "# Str. The value to extract from the histogram. Options are 'coarse', 'Gaussian', or 'median'.",],
                           "Step 4c":["# Bool. Whether to subtract the background using a column-by-column method.",
                                      "# lst of int. The indices defining the rows used as background.",
                                      "# float. How aggressively to mask outliers in the background region.",],
                           "Step 4d":["# Bool. Whether to subtract the background using the scaled Pagul+ 2023 G280 sky image.",
                                      "# Str. The absolute path to where the Pagul+ 2023 G280 sky image is stored.",
                                      "# Float. How strong the trace masking should be. Values of 0.001 or less recommended.",
                                      "# Bool. If True, takes the median value of each column in the Pagul+ 2023 sky image as the background. As the Pagul+ 2023 image is undersampled, this helps to suppress fluctuations in the image.",],
                           "Step 5a":["# Bool. Whether to track frame displacements by centroiding the 0th order.",],
                           "Step 5b":["# Bool. Whether to track frame displacements by centroiding background stars.",
                                      "# Lst of lst of float. Every list should indicate the estimated location of every background star",],
                           "Step 6":["# Bool. If True, saves the output xarray to be used in Stage 2.",],
                           }
    return header, subsection_headers, subsection_keys, subsection_comments

def Stage3_info():
    '''
    Specific keys and subsections for Stage 2.
    '''
    header = "# ExoTiC-UVIS config file for launching Stage 2: Extraction"

    subsection_headers = ["# Setup for Stage 2",
                          "# Step 1: Read in the data",
                          "# Step 2: Trace configuration",
                          "# Step 3: 1D spectral extraction",
                          "# Step 3a: Box extraction parameters",
                          "# Step 3b: Optimum extraction parameters",
                          "# Step 4: 1D spectral cleaning and aligning",
                          ]
    
    subsection_keys = {"Setup":["toplevel_dir",
                                "run_name",
                                "verbose",
                                "show_plots",
                                "save_plots"],
                       "Step 1":[],
                       "Step 2":["config",
                                 "path_to_config",
                                 "traces_to_conf",
                                 "refine_fit"],
                       "Step 3":["method",
                                  "subtract_contam",
                                  "sens_correction",],
                       "Step 3a":["determine_hw",
                                  "indices",
                                  "halfwidths_box",],
                       "Step 3b":["aperture_type",
                                  "halfwidths_opt",],
                       "Step 4":["outlier_sigma",
                                 "align",],
                       }
    
    subsection_comments = {"Setup":["# Directory where your current project files are stored. This folder should contain the specimages/, directimages/, etc. folders with your data as well as the outputs folder.",
                                    "# Str. This is the name of the current run. It can be anything that does not contain spaces or special characters (e.g. $, %, @, etc.).",
                                    "# Int from 0 to 2. 0 = print nothing. 1 = print some statements. 2 = print every action.",
                                    "# Int from 0 to 2. 0 = show nothing. 1 = show some plots. 2 = show all plots.",
                                    "# Int from 0 to 2. 0 = save nothing. 1 = save some plots. 2 = save all plots.",],
                           "Step 1":[],
                           "Step 2":["# Str. The type of configuration you are using. Options are 'aXe' or 'GRISMCONF'.",
                                      "# Str. The absolute path to the .conf file used by aXe or GRISMCONF.",
                                      "# Lst of str. The traces you want to configure and extraction from.",
                                      "# Bool. If True, uses Gaussian fitting to refine the trace solution.",],
                           "Step 3":["# Str. Options are 'box' (draw a box around the trace and sum without weights) or 'optimum' (weight using Horne 1986 methods).",
                                     "# Bool. Whether to model the contaminating orders and subtract them from your trace during extraction. Sometimes works, sometimes just adds lots of scatter.",
                                     "# Bool. Whether to correct for the G280's changing sensitivity as a function of wavelength. Since absolute calibrated spectra aren't needed in exoplanetary sciences, you can skip this safely.",],
                           "Step 3a":["# Bool. If True, automatically determines preferred half-width for each order by minimizing out-of-transit/eclipse residuals.",
                                      "# Lst of lsts of int. If determine_hw, these are the indices used to estimate the out-of-transit/eclipse residuals.",
                                      "# Lst of ints. The half-width of extraction aperture to use for each order. Input here is ignored if 'determine_hw' is True.",],
                           "Step 3b":["# Str. Type of aperture to draw. Options are 'row_polyfit', 'column_polyfit', 'column_gaussfit', 'column_moffatfit', 'median', 'smooth'.",
                                      "# Lst of ints. The half-width of extraction aperture to use for each order. For optimum extraction, you should make this big (>12 pixels at least). There is no 'preferred' half-width in optimum extraction due to the weights.",],
                           "Step 4":["# Float. Sigma at which to reject spectral outliers in time. Outliers are replaced with median of timeseries. Enter 'None' to skip this step.",
                                     "# Bool. If True, uses cross-correlation to align spectra to keep wavelength solution consistent.",],
                           }
    return header, subsection_headers, subsection_keys, subsection_comments

def Stage4_info():
    '''
    Specific keys and subsections for Stage 3.
    '''
    header = "# ExoTiC-UVIS config file for launching Stage 3: Binning"

    subsection_headers = ["# Setup for Stage 3",
                          "# Step 1: Read in the data",
                          "# Step 2: Light curve extraction",
                          ]
    
    subsection_keys = {"Setup":["toplevel_dir",
                                "run_name",
                                "verbose",
                                "show_plots",
                                "save_plots"],
                       "Step 1":[],
                       "Step 2":["bin_method",
                                 "wavelength_bins",
                                 "N_columns",
                                 "time_binning",
                                 "sigma_clip",
                                 "normalize",],
                       }
    
    subsection_comments = {"Setup":["# Directory where your current project files are stored. This folder should contain the specimages/, directimages/, etc. folders with your data as well as the outputs folder.",
                                    "# Str. This is the name of the current run. It can be anything that does not contain spaces or special characters (e.g. $, %, @, etc.).",
                                    "# Int from 0 to 2. 0 = print nothing. 1 = print some statements. 2 = print every action.",
                                    "# Int from 0 to 2. 0 = show nothing. 1 = show some plots. 2 = show all plots.",
                                    "# Int from 0 to 2. 0 = save nothing. 1 = save some plots. 2 = save all plots.",],
                           "Step 1":[],
                           "Step 2":["# Str. How to bin the light curves. Options are 'columns' (bin N columns at a time) or 'wavelengths' (bin from wavelength1 to wavelength2).",
                                     "# Lst of floats or numpy array. If bin_method is 'wavelengths', defines edges of each wavelength bin.",
                                     "# Int. If bin_method is 'columns', how many columns go into each bin.",
                                     "# Int or None. If int, how many frames in time should be binned. Reduces computation time but degrades time resolution.",
                                     "# Float or None. If float, the sigma at which to mask outliers in sigma clipping.",
                                     "# Bool. If True, normalizes curves by out-of-transit/eclipse flux.",],
                           }
    return header, subsection_headers, subsection_keys, subsection_comments

def Stage5_info():
    '''
    Specific keys and subsections for Stage 1.
    '''

def Stage6_info():
    '''
    Specific keys and subsections for Stage 1.
    '''