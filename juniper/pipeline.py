# This is going to be the big wrapper for all stages.
import os
import glob
import numpy as np

from juniper.config import *
from juniper.util import *
from juniper.stage1 import *
from juniper.stage2 import *
from juniper.stage3 import *
from juniper.stage4 import *
from juniper.stage5 import *
from juniper.stage6 import *

def run_pipeline(config_folder,stages=(1,2,3,4,5,6,)):
    """Wrapper for the Juniper pipeline.

    Args:
        config_folder (str): path to the folder that stores all config files.
        stages (tuple, optional): tuple of ints which specify the stages you want to run. Defaults to (1,2,3,4,5,6,).
    """
    ### Run Juniper Stage 1: JWST Detector1Pipeline and 1/f Corrections
    if 1 in stages:
        # Open the config dictionary.
        s1_config = glob.glob(os.path.join(config_folder,'s1_*'))[0]
        s1_config = read_config.read_config(s1_config)

        # Process Stage 1.
        for f in glob.glob(s1_config["files_loc"]):
            s1_result = wrap_stage1jwst.wrap_front_end(f,
                                                       inpt_dict=s1_config["wrap_front_end"])
            
            if s1_config["do_glbs"]:
                s1_result.data = group_level_bckg_sub.glbs_all(s1_result.data,
                                                               inpt_dict=s1_config["glbs"])
            
            wrap_stage1jwst.wrap_back_end(s1_result,
                                          inpt_dict=s1_config["wrap_back_end"],
                                          outfile=s1_config["files_rename"],
                                          outdir=s1_config["toplevel_dir"])
            
            if s1_config["do_NSClean"]:
                # WIP! Needs to open the wrap_back_end files and NSClean them, then save them again.
                pass