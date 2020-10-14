""" List of utils functions

"""
# encoding: utf-8
import numpy as np

def datetime_to_nightsec(d_time):
    try:
        new_time = d_time.hour * 3600 + d_time.minute * 60 + d_time.second
        if new_time > 16 * 3600:
            new_time -= 24 * 3600
        return new_time

    except:
        return float('nan')


def array_to_times(binary_array):
    """ Translate a binary array into a list of index and durations.

    It detects when value of data switch from 0 to 1 and how long
    the value stays to 1.

    Parameters
    ----------
    binary_array : 1D numpy.array
        Binary array to convert.

    Returns
    -------
        up_times
            Index where binary array switching from 0 to 1
        durations
            Durations of 1 phase

    Example
    -------
    >>> array_to_times([1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1])
    ([0, 6, 9], [4, 1, 3])

    """
    # compute when a one appears
    data = np.array(binary_array)
    up_times = (np.where(np.diff(data) == 1)[0] + 1).tolist()
    if data[0] == 1:  # if we start with a one
        up_times = [0] + up_times
    # compute when a zero appears
    down_times = (np.where(np.diff(data) == -1)[0] + 1).tolist()
    if data[-1] == 1:  # if we end with a one
        down_times = down_times + [data.shape[0]]
    # compute the durations
    durations = []
    for up, down in zip(up_times, down_times):
        durations.append(down - up)
    return up_times, durations
