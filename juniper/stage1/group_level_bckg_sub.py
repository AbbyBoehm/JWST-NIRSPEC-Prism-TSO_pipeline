import numpy as np
import time

import matplotlib.pyplot as plt

from revised_utils import clean, timer

def do(datamodel, inpt_dict):
    '''
    Performs group-level background subtraction on every group in the datamodel according to the instructions in inpt_dict.

    :param datamodel: JWST datamodel. A datamodel containing attribute .data, which is an np array of shape nints x ngroups x nrows x ncols, produced during wrap_front_end.
    :param inpt_dict: dict. A dictionary containing instructions for performing this step.
    :return: datamodel with updated cleaned .data attribute.
    '''
    datamodel.data = glbs(datamodel.data,
                          bckg_kernel=inpt_dict["kernel"],
                          bckg_sigma=inpt_dict["sigma"],
                          time_step=inpt_dict["timer"],
                          show=inpt_dict["show"])
    return datamodel

def glbs(data, bckg_kernel, bckg_sigma, time_step, show):
    '''
    Performs 1/f subtraction on the given array.
    Adapted from routine developed by Trevor Foote (tof2@cornell.edu).
    
    :param data: 4D array. Array of integrations x groups x rows x cols.
    :param bckg_kernel: tuple of odd int. Kernel to use when cleaning the background region. Should be a column like (5,1) so that columns do not contaminate adjacent columns.
    :param bckg_sigma: float. Threshold to reject outliers from the background region.
    :param time_step: bool. Whether to run the timer on this step.
    :param show: bool. Whether to show diagnostic frames. For inspection of whether this step is working properly.
    :return: 4D array that has undergone 1/f subtraction.
    '''
    # Time this step if asked.
    if time_step:
        t0 = time.time()

    # Begin processing.
    for i in range(np.shape(data)[0]): # for each integration
        trace_mask = np.ma.getmask(clean(np.copy(data[i,-1,:,:]),bckg_sigma,bckg_kernel)) # obtain the mask that hides the trace using a cleaned version of the very last group in this integration.
        if (i == 0 and show):
            plt.imshow(trace_mask)
            plt.title('1/f trace mask')
            plt.show()
            plt.close()
        for g in range(np.shape(data)[1]): # for each group
            # Define the background region.    
            background_region = np.ma.masked_array(data=np.copy(data[i, g, :, :]),
                                                   mask=trace_mask)
            
            # Define the median background in each column and extend to a full-size array.
            #backgroun = np.ma.mean(background_region,axis=0)
            background = np.ma.median(background_region,axis=0)
            background = np.array([background,]*np.shape(data)[2])

            # And remove background from data.
            data[i, g, :, :] = data[i, g, :, :] - background

            del(background_region) # recover memory by deleting the copied array.
        
        if (time_step and i%1000 == 0 and i != 0):
            timer(time.time()-t0,i+1,i,np.shape(data)[0] - i)
    print("1/f subtraction completed in %.3f seconds." % (time.time()-t0))
    return data

def mask_trace(group):
    '''
    Masks anywhere in the final group that is determined to be part of the trace.

    :param group: 2D array. Final group in an integration, cleaned for outliers.
    :return: np.ma mask for obscuring traces.
    '''
    # We choose 10000 as the upper limit for bckg values, but this may need tuned by dataset!
    mu = np.median(group[group<10000])
    sig = np.std(group[group<10000])
    masked_fg = np.ma.masked_where(np.abs(group - mu) > sig, group)
    return np.ma.getmask(masked_fg)