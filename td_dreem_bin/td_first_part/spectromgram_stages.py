""" Spectrogram on EEG signals

"""
import numpy as np
import matplotlib.pyplot as plt
from mne import create_info, EpochsArray
from mne.baseline import rescale
from mne.time_frequency import (tfr_array_multitaper, tfr_array_stockwell, tfr_array_morlet)
from mne.viz import centers_to_edges

frequency_interval = np.arange(0.5, 15, 0.2)


def compute_spectrogram_dataset(dataset, fs=50., algorithm="multitaper", freqs=frequency_interval):
    """ Spectrogram on all epochs"""

    # add channel dimension for 2D-arrays
    if dataset.ndim == 2:
        dataset = np.expand_dims(dataset, axis=1)

    # parameters
    n_cycles = 10
    time_bandwidth = 3

    # time-frequency computation
    if algorithm == "multitaper":
        power = tfr_array_multitaper(dataset, sfreq=fs, freqs=freqs, n_cycles=n_cycles,
                                     time_bandwidth=time_bandwidth,output='power')
    elif algorithm == "stockwell":
        power = tfr_array_stockwell(dataset, sfreq=fs, freqs=freqs, n_cycles=n_cycles,
                                    output='power')
    elif algorithm == "morlet":
        power = tfr_array_morlet(dataset, sfreq=fs, freqs=freqs, n_cycles=n_cycles,
                                 output='power')

    return power


if __name__ == "__main__":
    from td_dreem_bin.load_data.data_kaggle import load_xtrain_dataset, load_ytrain

    # sleep stages
    stage_correspondance = {0: "WAKE", 1: "N1", 2: "N2", 3: "DEEP", 4: "REM"}
    sleep_stages = ['N1', 'N2', 'N3', 'REM', 'Wake']
    freq = frequency_interval[:-1]

    # data
    derivation = 3  # eeg_4
    fs = 50.
    x_train = load_xtrain_dataset()[:, derivation, :]
    y_train = load_ytrain()

    # spectrogram_dataset = compute_spectrogram_dataset(x_train,
    #                                                   fs=fs,
    #                                                   algorithm="multitaper",
    #                                                   freqs=frequency_interval)

    # data
    dataset = np.expand_dims(x_train, axis=1)
    n_cycles = freqs / 2.
    time_bandwidth = 3
    freqs = frequency_interval

    power = tfr_array_multitaper(dataset[0:100], sfreq=fs, freqs=freqs, n_cycles=n_cycles,
                                 time_bandwidth=time_bandwidth, output='power')
