""" Spectrum on EEG signals

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from mne import create_info, EpochsArray
from mne.baseline import rescale
from mne.time_frequency import (tfr_array_multitaper, tfr_array_stockwell, tfr_array_morlet)
from mne.viz import centers_to_edges

frequency_interval = np.arange(0.5, 15, 0.2)


def spectral_power_one_channel(signal, fs=250., frequency_interval=frequency_interval,
                               stft_duration=5, stft_overlap=3.5, epoch_duration=30, window='hamming'):
    nperseg, noverlap, nperepoch = int(stft_duration * fs), int(stft_overlap * fs), int(epoch_duration * fs)
    N_epoch = signal.shape[0]

    freq, _, signal_stft = stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    signal_stft = np.abs(signal_stft)
    if frequency_interval is not None:
        PSD = []
        for i in range(len(frequency_interval)-1):
            idx_to_sum = np.where(np.logical_and(freq >= frequency_interval[i], freq <= frequency_interval[i + 1]))[0]
            if len(idx_to_sum) > 0:
                PSD += [np.sum(signal_stft[:, idx_to_sum, :], axis=2, keepdims=True)]
        PSD = np.concatenate(PSD, 2)

    return PSD.squeeze()


def average_spectrum_stage(spectrum_array, sleep_stages_list, stages):

    sleep_stages_array = np.asarray(sleep_stages_list)[:, 1]
    idx = np.isin(sleep_stages_array, stages)

    all_spectromgram = spectrum_array[idx, :]
    spectrum = np.mean(all_spectromgram, 0)

    return spectrum


if __name__ == "__main__":
    from td_dreem_bin.load_data.data_kaggle import load_xtrain_dataset, load_ytrain

    # sleep stages
    stage_correspondance = {0: "WAKE", 1: "N1", 2: "N2", 3: "DEEP", 4: "REM"}
    sleep_stages = ['N1', 'N2', 'N3', 'REM', 'Wake']
    freq = frequency_interval[:-1]

    # data
    derivation = 3  # eeg_4
    x_train = load_xtrain_dataset()[:, derivation, :]
    y_train = load_ytrain()

    spectrum_train = spectral_power_one_channel(x_train)

    # Normalized spectrum
    normalized_spectrum = {}
    # whole night
    night_spectrum = average_spectrum_stage(spectrum_train, y_train, range(4))
    normalized_spectrum['all'] = freq * night_spectrum
    normalized_spectrum['all'] = np.log10(night_spectrum)

    # by stages
    for stage, stage_label in stage_correspondance.items():
        spectrum = average_spectrum_stage(spectrum_train, y_train, stage)
        normalized_spectrum[stage_label] = np.log10(spectrum)

    # plot
    fig, axs = plt.subplots(3, 2, figsize=(9, 7))
    axs = np.ravel(axs)

    axs[0].plot(freq, normalized_spectrum['all'])
    axs[0].set_xlim(0, 15)

    for stage, stage_label in stage_correspondance.items():
        ax = axs[stage + 1]
        ax.plot(freq, normalized_spectrum[stage_label])
        ax.set_xlim(0, 15)
        ax.set_title(stage_label)

    fig.show()
