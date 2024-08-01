def mask_flags(segments, replacement=None, spare_flags=[]):
    """Uses the jwst pipeline data quality flags to mask bad pixels.

    Args:
        segments (xarray): segments.data contains the integrations and segments.dq contains the data quality flags.
        replacement (, optional): how to handle replacing the flagged values. Defaults to None.
        spare_flags (list, optional): the integer values of any jwst pipeline flags you don't want to mask. Defaults to None.

    Returns:
        xarray: segments with the flagged pixels masked.
    """
    
    return segments