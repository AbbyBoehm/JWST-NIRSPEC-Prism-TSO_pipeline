import shlex
import numpy as np

def read_config(path_to_config_file):
    """Parses config files to create a dictionary of inputs.
    Credit V.A. Boehm from ExoTiC-UVIS.

    Args:
        path_to_config_file (str): Path to the .berry file that is being read.

    Returns:
        dict: instructions for pipeline.py to follow.
    """
    # Open the dictionary.
    config = {}

    # Define certain keys as special.
    special_keys = ["rp","fp","t_prim","t_seco","period",
                    "aor","incl","ecc","longitude"]
    prior_keys = [key+"_prior" for key in special_keys]
    for key in prior_keys:
        special_keys.append(key)
    seen_this_key = {}
    for key in special_keys:
        seen_this_key[key] = 0

    # Read out all lines.
    with open(path_to_config_file,mode='r') as f:
        lines = f.readlines()

    # Process all lines.
    for line in lines:
        line = shlex.split(line, posix=False)
        # Check if it is empty line or comment line and pass.
        if len(line) == 0:
            continue
        if line[0] == '#':
            continue
        key = line[0]

        # Handle special keys.
        if key in special_keys:
            # For planets and flares, the same key name may appear many times. 
            # e.g. if you fit two planets in transit, rp and rp_prior will both appear twice.
            # So we have to add numbers to keep them distinct.
            seen_this_key[key] += 1 # keep track of how many times we've seen the special key.
            key = key + str(seen_this_key[key])

        # Param may have spaces, so we need to keep going with it.
        param = line[1]
        i = 2
        while "#" not in line[i]:
            param = ''.join([param,line[i]])
            i += 1
        try:
            param = eval(param)
        except:
            pass
        # If we are reading planet or flare parameters, there are special rules! They have duplicates!
        
        config[key] = param

    return config