def timer(elapsed_time,current_iter,iters_completed,iters_remaining):
    '''
    Simple util for reporting the time a step is taking.

    :param elapsed time: float. time.time() - t0.
    :param current_step: int. The number of the current iteration, usually index+1.
    :param steps_competed: int. The number of iterations completed up to now, or index.
    :param steps_remaining: int. The number of iterations yet to be done, or N_total - index.
    :return: simple print-out of the status.
    '''
    if current_iter != None:
        iterrate = iters_completed/elapsed_time
        print("On iteration %.0f. Elapsed process time: %.3f seconds." % (current_iter, elapsed_time))
        print("Average rate of processing: %.3f iter/s." % iterrate)
        print("Estimated time remaining: %.3f seconds.\n" % (iters_remaining/iterrate))
    else:
        print("Step completed in %.3f seconds = %.3f minutes." % (elapsed_time, elapsed_time/60))

def tqdm_translate(integer):
    """Simple util for checking whether tqdm should run.

    Args:
        integer (int): 0, 1, or 2. If 0, time nothing. If 1, time full steps. If 2, time each iteration in the step.

    Returns:
        bool, bool: bool statement on whether to time the full step and the iterations.
    """
    if integer == 2:
        return True, True
    elif integer == 1:
        return True, False
    else:
        return False, False
    
def plot_translate(integer):
    """Simple util for checking whether images should be displayed/saved.

    Args:
        integer (int): 0, 1, or 2. If 0, plot nothing. If 1, plot diagnostic plots of full steps. If 2, plot everything.

    Returns:
        bool, bool: bool statement on whether to plot the full step and the iterations.
    """
    if integer == 2:
        return True, True
    elif integer == 1:
        return True, False
    else:
        return False, False