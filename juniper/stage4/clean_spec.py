import time
from tqdm import tqdm

import numpy as np

from juniper.util.diagnostics import tqdm_translate, plot_translate, timer

def clean_spectra(oneD_spec, inpt_dict):
    """Compares all 1D spectra to a median spectra and replaces outliers
    with the median of that spectral point in time.

    Args:
        oneD_spec (np.array): 1D spectra to have outliers trimmed from.
        inpt_dict (dict): instructions for running this step.

    Returns:
        np.array: 1D spectra with outliers cleaned.
    """
    # Log.
    if inpt_dict["verbose"] >= 1:
        print("Cleaning 1D spectrum for outliers...")

    # Check tqdm and plotting requests.
    time_step, time_ints = tqdm_translate(inpt_dict["verbose"])
    # FIX : i'll figure this out later
    plot_step, plot_ints = plot_translate(inpt_dict["show_plots"])
    save_step, save_ints = plot_translate(inpt_dict["save_plots"])

    # Time step, if asked.
    if time_step:
        t0 = time.time()

    # Track cleaned spectra and load sigma.
    cleaned_specs = []
    sigma = inpt_dict["sigma"]
    
    # Iterate over spectra.
    for i in tqdm(range(oneD_spec.shape[0]),
                  desc='Cleaning spectral outliers...',
                  ):
        # Track outliers removed.
        bad_spex_removed = 0

        # Iteration stop condition. As long as outliers are being found, we have to keep iterating.
        outlier_found = True
        while outlier_found:
            # Define median spectrum in time and extend its size to include all time.
            med_spec = np.median(oneD_spec,axis=0)
            med_spec = np.array([med_spec,]*oneD_spec.shape[0])
            # Get standard deviation of each point.
            std_spec = np.std(oneD_spec,axis=0)
            std_spec = np.array([std_spec,]*oneD_spec.shape[0])

            # Flag outliers.
            S = np.where(np.abs(oneD_spec-med_spec) > sigma*std_spec, 1, 0)

            # Count outliers found.
            bad_spex_this_step = np.count_nonzero(S)
            bad_spex_removed += bad_spex_this_step

            if bad_spex_this_step == 0:
                # No more outliers found! We can break the loop now.
                outlier_found = False
            
            # Correct outliers and loop once more.
            oneD_spec = np.where(S == 1, med_spec, oneD_spec)
        print("1D spectral cleaning complete. Removed %.0f spectral outliers." % bad_spex_removed)
        cleaned_specs.append(oneD_spec)
    return np.array(cleaned_specs)