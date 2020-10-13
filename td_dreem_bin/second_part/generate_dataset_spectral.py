"""
Generate two datasets from spectrogram
"""
from datetime import datetime
import numpy as np

from td_dreem_bin.utils.spectrogram import compute_spectrogram
from td_dreem_bin.load_data.data_records import (
    get_one_record,
    train_records,
    test_records
)


def create_dataset_spectral(record_list):
    """
    generate x and y data from record_list for classification
    """
    # params
    fs = 250.  # Hz
    epoch_size = 30.  # sec
    fmin, fmax = 0.5, 18

    # generate dataset
    x_data = []
    y_data = []

    for record in record_list:
        # get data
        results = get_one_record(record)
        eeg_1 = results['eeg_1']
        eeg_2 = results['eeg_2']
        hypnogram = results['hypnogram']
        # accelerometer = results['accelerometer']

        # compute spectrogram
        specg1, t1, freq1 = compute_spectrogram(eeg_1, fs, win_sec=epoch_size, fmin=fmin, fmax=fmax)
        specg2, t2, freq2 = compute_spectrogram(eeg_2, fs, win_sec=epoch_size, fmin=fmin, fmax=fmax)

        # same size hypno and spectro
        if len(t1) > len(hypnogram):
            specg1 = specg1[:, :len(hypnogram)]
            t1 = t1[:len(hypnogram)]
        elif len(t1) < len(hypnogram):
            hypnogram = hypnogram[:len(t1)]

        # dataset
        x_data += [np.transpose(np.concatenate([specg1, specg2], axis=0))]
        y_data += [hypnogram]

    x_data = np.concatenate(x_data)
    y_data = np.concatenate(y_data)

    # remove non labelled epochs
    idx_keep = y_data >= 0
    x_data = x_data[idx_keep]
    y_data = y_data[idx_keep]

    return x_data, y_data


def get_train_dataset_spectral():
    return create_dataset_spectral(train_records)


def get_test_dataset_spectral():
    return create_dataset_spectral(test_records)


if __name__ == "__main__":

    x_train, y_train = get_train_dataset_spectral()
    x_test, y_test = get_test_dataset_spectral()

    # save
    import os
    from td_dreem_bin import path_repo
    save_folder = os.path.join(path_repo, "data", "processed/")

    save_path = os.path.join(save_folder, "record_datatrain_spectrogram")
    np.savez(save_path, x_train=x_train, y_train=y_train)
    save_path = os.path.join(save_folder, "record_datatest_spectrogram")
    np.savez(save_path, x_test=x_test, y_test=y_test)
