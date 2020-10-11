""" Hypnogram utils functions.

hypnogram_to_epoch
find_stages_period
merge_close_periods
intersect_periods

"""
# encoding: utf-8
import numpy as np


def hypnogram_to_epoch(hypnogram,
                       binsize=30
                       ):
    """
    Convert Dreemnogram to sleep stages epoch

    Parameters
    ----------

    hypnogram : 1D numpy.array
        List of sleep stages
    binsize : float
        Epoch duration

    sleep_stades : list
            list of [start, end, stages]

    """

    if len(hypnogram) == 0:
        return []

    hypnogram = np.asarray(hypnogram)
    change_stade = np.append([True], np.diff(hypnogram) != 0)
    change_stade = np.where(change_stade)[0]

    starts = change_stade * binsize
    ends = np.append(starts[1:], len(hypnogram) * binsize)
    stades = hypnogram[change_stade]

    sleep_stades = np.dstack((starts, ends, stades)).tolist()[0]

    return sleep_stades


def find_stages_period(hypnogram,
                       stages,
                       binsize=30
                       ):
    """
    Find Specific stage periods in Dreemnogram

    Parameters
    ----------

    hypnogram : 1D numpy.array
        List of sleep stages
    stages : 1D numpy.array or int
        List of sleep stages or only one sleep stage
    binsize : float
        Epoch duration

    sleep_periods : list
            list of [start, end]

    """

    # int to list enventually
    if isinstance(stages, int):
        stages = [stages]

    # hypnogram Sleep/NonSleep only
    new_hypnogram = np.array(hypnogram)
    idx_stage = np.isin(new_hypnogram, stages)
    new_hypnogram[idx_stage] = 1
    new_hypnogram[np.logical_not(idx_stage)] = 0

    sleep_stades = hypnogram_to_epoch(new_hypnogram, binsize=binsize)
    stage_periods = [[sst[0], sst[1]] for sst in sleep_stades if sst[2] == 1]

    return stage_periods


def merge_close_periods(periods,
                        merge_threshold
                        ):
    """
    Merge close periods

    Parameters
    ----------

    periods : list
        List of [start, end], non overlapping
    merge_threshold : float
        threshold below which periods are merged


    merged_periods : list
            list of [start, end]

    """

    # if less than 2 periods
    if len(periods) < 2:
        return periods

    # sort periods
    starts = np.array([p[0] for p in periods])
    ends = np.array([p[1] for p in periods])
    idx_sort = np.argsort(starts)
    starts = starts[idx_sort]
    ends = ends[idx_sort]

    # merge close periods - only keep spaced enough periods
    long_gaps = np.where(starts[1:] - ends[:-1] > merge_threshold)[0]
    if long_gaps.size > 0:
        merged_end = np.append(ends[long_gaps], ends[-1])
        merged_start = np.append(starts[0], starts[long_gaps + 1])
        merged_periods = np.dstack((merged_start, merged_end)).tolist()[0]
    else:
        merged_periods = [[starts[0], ends[-1]]]

    return merged_periods


def intersect_periods(period1, period2):
    """
    Intersection between two list of epochs

    Parameters
    ----------

    period1 : list
        List of [int:start, int:end], non overlapping, time in second
    period2 : list
        List of [int:start, int:end], non overlapping


    intersection : list
            list of [start, end]

    """

    if len(period1) == 0 or len(period2) == 0:
        return []

    # timestamps of period1 and period2 -> Intersection
    tmp_period1 = np.sort(np.unique(np.hstack([np.arange(p[0], p[1] + 1) for p in period1])))
    tmp_period2 = np.sort(np.unique(np.hstack([np.arange(p[0], p[1] + 1) for p in period2])))
    tmp_both_period = np.intersect1d(tmp_period1, tmp_period2)

    if len(tmp_both_period) == 0:
        return []

    # find epochs from tmp_both_period
    change_epoch = np.where(np.diff(tmp_both_period) != 1)[0]
    intersect_start = tmp_both_period[np.append(0, change_epoch + 1)]
    intersect_end = np.append(tmp_both_period[change_epoch], tmp_both_period[-1])
    intersection = np.dstack((intersect_start, intersect_end)).tolist()[0]

    return intersection


class Hypnogram:
    """ Hypnogram """

    def __init__(self, hypnogram, increment_duration=30):
        self.list_hypno = hypnogram
        self.increment_duration = increment_duration

    def get_hypnogram(self):
        return self.list_hypno

    def get_stage_epochs(self, stages=[]):
        if not stages:
            return hypnogram_to_epoch(self.list_hypno, binsize=self.increment_duration)
        else:
            return find_stages_period(self.list_hypno, stages=stages, binsize=self.increment_duration)
