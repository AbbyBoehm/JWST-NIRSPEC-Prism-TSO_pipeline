# This is going to be the big wrapper for all stages.
import os
import glob
import shutil
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
        s1_config_path = glob.glob(os.path.join(config_folder,'s1_*'))[0]
        s1_config = read_config.read_config(s1_config_path)

        # Define directories.
        project_dir = s1_config["toplevel_dir"]
        input_dir = os.path.join(project_dir,s1_config["input"])
        output_dir = os.path.join(project_dir,s1_config["output"])

        # Find files.
        files = sorted(glob.glob(os.path.join(input_dir,"*uncal.fits")))

        # Process Stage 1.
        do_stage1(files,
                  ['{}{}'.format(s1_config["rename"],i) for i, f in enumerate(files)],
                  output_dir,
                  s1_config)
        
        # Write the config dictionary out as a copy.
        config_outdir = os.path.join(output_dir,"configuration")
        if not os.path.exists(config_outdir):
            os.makedirs(config_outdir)
        config_name = "s1_juniper.berry"
        shutil.copy(s1_config_path,os.path.join(config_outdir,config_name))
    
    ### Run Juniper Stage 2: ??