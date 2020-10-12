""" Spectrogram on EEG signals

"""
import numpy as np
from scipy.signal import decimate
from mne.time_frequency import (tfr_array_multitaper, tfr_array_morlet)

frequency_interval = np.arange(0.5, 18, 0.5)


def compute_spectrogram_dataset(dataset, fs=50., algorithm="multitaper",
                                freqs=frequency_interval, split_epochs=1000):
    """ Spectrogram on all epochs"""

    # add channel dimension for 2D-arrays
    if dataset.ndim == 2:
        dataset = np.expand_dims(dataset, axis=1)

    data_limit = np.append(np.arange(0, len(dataset), int(split_epochs)), len(dataset))
    all_power = []

    for i in range(len(data_limit) - 1):
        k_min = data_limit[i]
        k_max = data_limit[i + 1]

        # time-frequency computation
        if algorithm == "multitaper":
            n_cycles = freqs / 2.
            time_bandwidth = 4
            power = tfr_array_multitaper(dataset[k_min:k_max], sfreq=fs, freqs=freqs, n_cycles=n_cycles,
                                         time_bandwidth=time_bandwidth, output='power')
        elif algorithm == "morlet":
            n_cycles = freqs / 2.
            power = tfr_array_morlet(dataset[k_min:k_max], sfreq=fs,  freqs=freqs, n_cycles=n_cycles,
                                     output='power')

        power = power.squeeze()
        power = decimate(power, 50)
        print(k_max)
        all_power += [power]

    all_power = np.concatenate(all_power)

    return all_power


if __name__ == "__main__":
    from td_dreem_bin.load_data.data_kaggle import load_xtrain_dataset, load_ytrain

    # sleep stages
    stage_correspondance = {0: "WAKE", 1: "N1", 2: "N2", 3: "DEEP", 4: "REM"}
    sleep_stages = ['N1', 'N2', 'N3', 'REM', 'Wake']
    freq = frequency_interval[:-1]

    # save folder
    import os
    from td_dreem_bin import path_repo
    save_folder = os.path.join(path_repo, "data", "processed/")

    # data
    derivation = 3  # eeg_4
    fs = 50.
    x_train = load_xtrain_dataset()[:, derivation, :]
    freqs = frequency_interval
    # normalize data
    x_train = np.clip(x_train, -150, 150)

    # Time-frequency representation(spectrogram)
    multitaper_power = compute_spectrogram_dataset(x_train, fs=fs, algorithm="multitaper",
                                                   freqs=frequency_interval, split_epochs=1000)
    morlet_power = compute_spectrogram_dataset(x_train, fs=fs, algorithm="morlet",
                                                   freqs=frequency_interval, split_epochs=1000)

    # save
    save_path = os.path.join(save_folder, "spectrogram_multitaper1.npy")
    np.save(save_path, multitaper_power)
    save_path = os.path.join(save_folder, "spectrogram_morlet1.npy")
    np.save(save_path, morlet_power)
    save_path = os.path.join(save_folder, "frequencies1.npy")
    np.save(save_path, freqs)
