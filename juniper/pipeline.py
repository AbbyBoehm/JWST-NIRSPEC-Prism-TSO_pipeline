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
'''
from juniper.stage5 import *
from juniper.stage6 import *
'''

def run_pipeline(config_folder,stages=(1,2,3,4,5,6,)):
    """Wrapper for the Juniper pipeline.

    Args:
        config_folder (str): path to the folder that stores all config files.
        stages (tuple, optional): tuple of ints which specify the stages you want to run. Defaults to (1,2,3,4,5,6,).
    """
    print("Juniper pipeline running with following stages:", stages)
    print(".berry configuration files will be sourced from:", config_folder)
    ### Run Juniper Stage 1: JWST Detector1Pipeline and 1/f Corrections
    if 1 in stages:
        # Open the config dictionary.
        s1_config_path = glob.glob(os.path.join(config_folder,'s1_*'))[0]
        s1_config = read_config.read_config(s1_config_path)

        # Set up run name and define directories.
        run_name = s1_config["run_name"] # set up run_name for the .berry file
        project_dir = s1_config["toplevel_dir"]
        input_dir = os.path.join(project_dir,s1_config["input"])
        output_dir = os.path.join(project_dir,s1_config["output"])
        if run_name:
            # Add an extra sub-folder to separate this run from other runs.
            output_dir = os.path.join(project_dir,os.path.join(s1_config["output"],run_name))
        diagnosticplots_dir = os.path.join(project_dir, s1_config["diagnostics"]) # FIX: currently unused

        # Find files.
        files = sorted(glob.glob(os.path.join(input_dir,"*uncal.fits")))
        outfiles = [None for i in files] # use default names and change uncal to rateints, that's all.
        if s1_config["rename"]:
            # Set up new outfile names and also change uncal to rateints.
            outfiles = ['{}_f{}_rateints'.format(s1_config["rename"],i) for i, f in enumerate(files)]

        # Set up crds cache.
        os.environ["CRDS_PATH"] = "crds_cache" # if no path to crds_cache defined, set it up in the cwdir
        if s1_config["CRDS_PATH"]:
            os.environ["CRDS_PATH"] = s1_config["CRDS_PATH"] # set path to CRDS cache if you already have on
        elif not os.path.exists("crds_cache"):
            os.makedirs("crds_cache")

        # Process Stage 1.
        do_stage1(files,outfiles,output_dir,s1_config)
        
        # Write the config dictionary out as a copy.
        config_outdir = os.path.join(output_dir,"configuration")
        if not os.path.exists(config_outdir):
            os.makedirs(config_outdir)
        config_name = "s1_{}_juniper.berry".format(run_name)
        shutil.copy(s1_config_path,os.path.join(config_outdir,config_name))
    
    ### Run Juniper Stage 2: Wavelength Solution and Curvature Correction
    if 2 in stages:
        # Open the config dictionary.
        s2_config_path = glob.glob(os.path.join(config_folder,'s2_*'))[0]
        s2_config = read_config.read_config(s2_config_path)

        # Set up run name and define directories.
        run_name = s2_config["run_name"] # set up run_name for the .berry file
        project_dir = s2_config["toplevel_dir"]
        input_dir = os.path.join(project_dir,s2_config["input"])
        output_dir = os.path.join(project_dir,s2_config["output"])
        if run_name:
            # Add an extra sub-folder to separate this run from other runs.
            output_dir = os.path.join(project_dir,os.path.join(s2_config["output"],run_name))
        diagnosticplots_dir = os.path.join(project_dir, s2_config["diagnostics"])

        # Find files.
        files = sorted(glob.glob(os.path.join(input_dir,"*rateints.fits")))
        fnames = [str.split(f,sep='/')[-1] for f in files]
        outfiles = [str.replace(f,'_rateints.fits','_calints') for f in fnames] # use default names and change rateints to calints, that's all.
        if s2_config["rename"]:
            # Set up new outfile names and also change rateints to calints.
            outfiles = ['{}_f{}_calints'.format(s2_config["rename"],i) for i, f in enumerate(files)]

        # Set up crds cache.
        os.environ["CRDS_PATH"] = "crds_cache/jwst_ops/" # if no path to crds_cache defined, set it up in the cwdir
        if s2_config["CRDS_PATH"]:
            os.environ["CRDS_PATH"] = s2_config["CRDS_PATH"] # set path to CRDS cache if you already have on
        elif not os.path.exists("crds_cache/jwst_ops/"):
            os.makedirs("crds_cache/jwst_ops/")

        # Process Stage 2.
        do_stage2(files,outfiles,output_dir,s2_config,diagnosticplots_dir)
        
        # Write the config dictionary out as a copy.
        config_outdir = os.path.join(output_dir,"configuration")
        if not os.path.exists(config_outdir):
            os.makedirs(config_outdir)
        config_name = "s2_juniper.berry"
        if run_name:
            config_name = "s2_{}_juniper.berry".format(run_name)
        shutil.copy(s2_config_path,os.path.join(config_outdir,config_name))
    ### Run Juniper Stage 3: Reduction
    if 3 in stages:
        # Open the config dictionary.
        s3_config_path = glob.glob(os.path.join(config_folder,'s3_*'))[0]
        s3_config = read_config.read_config(s3_config_path)

        # Set up run name and define directories.
        run_name = s3_config["run_name"] # set up run_name for the .berry file
        project_dir = s3_config["toplevel_dir"]
        input_dir = os.path.join(project_dir,s3_config["input"])
        output_dir = os.path.join(project_dir,s3_config["output"])
        if run_name:
            # Add an extra sub-folder to separate this run from other runs.
            output_dir = os.path.join(project_dir,os.path.join(s3_config["output"],run_name))
        diagnosticplots_dir = os.path.join(project_dir, s3_config["diagnostics"])

        # Find files.
        files = sorted(glob.glob(os.path.join(input_dir,"*calints.fits")))
        fnames = [str.split(f,sep='/')[-1] for f in files]
        outfiles = [str.replace(f,'_calints.fits','_reduced') for f in fnames] # use default names and change rateints to calints, that's all.
        if s3_config["rename"]:
            # Set up new outfile names and also change rateints to calints.
            outfiles = ['{}_f{}_reduced'.format(s3_config["rename"],i) for i, f in enumerate(files)]

        # Process Stage 3.
        do_stage3(files,outfiles,output_dir,s3_config,diagnosticplots_dir)

        # Write the config dictionary out as a copy.
        config_outdir = os.path.join(output_dir,"configuration")
        if not os.path.exists(config_outdir):
            os.makedirs(config_outdir)
        config_name = "s3_juniper.berry"
        if run_name:
            config_name = "s3_{}_juniper.berry".format(run_name)
        shutil.copy(s3_config_path,os.path.join(config_outdir,config_name))
    ### Run Juniper Stage 4: Reduction
    if 4 in stages:
        # Open the config dictionary.
        s4_config_path = glob.glob(os.path.join(config_folder,'s4_*'))[0]
        s4_config = read_config.read_config(s4_config_path)

        # Set up run name and define directories.
        run_name = s4_config["run_name"] # set up run_name for the .berry file
        project_dir = s4_config["toplevel_dir"]
        input_dir = os.path.join(project_dir,s4_config["input"])
        output_dir = os.path.join(project_dir,s4_config["output"])
        if run_name:
            # Add an extra sub-folder to separate this run from other runs.
            output_dir = os.path.join(project_dir,os.path.join(s4_config["output"],run_name))
        diagnosticplots_dir = os.path.join(project_dir, s4_config["diagnostics"])

        # Find files.
        files = sorted(glob.glob(os.path.join(input_dir,"*reduced.nc")))
        fnames = [str.split(f,sep='/')[-1] for f in files]
        outfile = [str.replace(f,'_reduced.nc','_1Dspec') for f in fnames][0] # use default names and change reduced to 1Dspec, that's all.
        if s4_config["rename"]:
            # Set up new outfile names and also change reduced to 1Dspec.
            outfile = '{}_1Dspec'.format(s4_config["rename"])

        # Process Stage 4.
        do_stage4(files,outfile,output_dir,s4_config,diagnosticplots_dir)

        # Write the config dictionary out as a copy.
        config_outdir = os.path.join(output_dir,"configuration")
        if not os.path.exists(config_outdir):
            os.makedirs(config_outdir)
        config_name = "s4_juniper.berry"
        if run_name:
            config_name = "s4_{}_juniper.berry".format(run_name)
        shutil.copy(s4_config_path,os.path.join(config_outdir,config_name))