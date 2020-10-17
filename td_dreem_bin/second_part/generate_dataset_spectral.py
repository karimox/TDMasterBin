"""
Generate two datasets from spectrogram
"""
from datetime import datetime
import numpy as np
import pandas as pd

from td_dreem_bin.utils.spectrogram import compute_spectrogram
from td_dreem_bin.load_data.data_records import (
    get_one_record,
    train_records,
    test_records
)


def get_feature_representation(eeg_data, hypnogram, fs=250., epoch_size=30):
    """
    generate a feature vactor from sleep data
    """
    features_vector = []
    features_name = []

    # params
    fmin, fmax = 0.5, 18
    downsampling_factor = 30

    # index
    features_name += ['index_window']
    features_vector += [np.arange(len(hypnogram))]

    # compute spectrogram
    specg, t, freq = compute_spectrogram(eeg_data, fs, win_sec=epoch_size, fmin=fmin, fmax=fmax)
    # downsample frequency bins
    df_spec = pd.DataFrame(specg)
    spectrogram = df_spec.groupby(df_spec.index // downsampling_factor).mean().to_numpy()
    freq = freq[::30] + np.diff(freq[::30])[0]/2

    # same size hypno and spectro
    if len(t) > len(hypnogram):
        spectrogram = spectrogram[:, :len(hypnogram)]
        t = t[:len(hypnogram)]

    features_name += list(freq)
    features_vector += [spectrogram]

    # concatenate and return
    features_vector = np.transpose(np.vstack(features_vector))
    spectral_feature_df = pd.DataFrame(features_vector, columns=features_name)
    return spectral_feature_df


def create_dataset_spectral(record_list):
    """
    generate x and y data from record_list for classification
    """
    # generate dataset
    x_data = []
    y_data = []

    for i, record in enumerate(record_list):
        # get data
        results = get_one_record(record)
        eeg_data = results['eeg_1']
        hypnogram = results['hypnogram']

        # extract features
        spectral_feature_df = get_feature_representation(eeg_data, hypnogram)
        # add column with record
        spectral_feature_df.insert(loc=0, column='record', value=i)

        # dataset
        x_data += [spectral_feature_df]
        y_data += [hypnogram]

    x_data = pd.concat(x_data)
    y_data = np.concatenate(y_data)

    # remove non labelled epochs
    idx_keep = y_data >= 0
    x_data = x_data[idx_keep]
    y_data = y_data[idx_keep]

    return x_data, y_data


def get_train_dataset_spectral():
    return create_dataset_spectral(train_records)


def get_test_dataset_spectral(output_freq=False):
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





