"""
Generate two datasets using feature extraction
"""
from datetime import datetime
import numpy as np
import pandas as pd

from td_dreem_bin.utils.spectrogram import compute_spectral_features, compute_spectrogram
from td_dreem_bin.utils.event_detections import compute_spindles_features, compute_slowwaves_features
from td_dreem_bin.utils.accelerometer import compute_accelerometer_features
from td_dreem_bin.load_data.data_records import (
    get_one_record,
    train_records,
    test_records
)


def get_feature_representation(eeg_data, x_y_z, epochs, fs=250.):
    """
    generate a feature vactor from sleep data
    """
    features_vector = []
    features_name = []

    # index
    features_name += ['index']
    features_vector += [np.arange(len(epochs) - 1)]

    # spectral band
    win_sec = np.mean(np.diff(epochs))
    spectrogram, t1, freq = compute_spectrogram(eeg_data, fs, win_sec=win_sec)
    features_specg = compute_spectral_features(spectrogram, freq)
    for k, v in features_specg.items():
        features_name += [k]
        features_vector += [v]

    # spindles
    ss_count, ss_magnitude, ss_durations = compute_spindles_features(eeg_data, fs, epochs)
    features_name += ['Nb spindles', 'spindles magnitude', 'spindles duration']
    features_vector += [ss_count, ss_magnitude, ss_durations]
    # slow waves
    sw_count, sw_magnitude, sw_durations = compute_slowwaves_features(eeg_data, fs, epochs)
    features_name += ['Nb slow waves', 'slow waves magnitude', 'slow waves duration']
    features_vector += [sw_count, sw_magnitude, sw_durations]

    # movement
    x, y, z = x_y_z
    mov_variance, little_movement, strong_movement, _ = compute_accelerometer_features(
        x, y, z, epochs)
    mov_variance = np.max(mov_variance, 1)
    features_name += ['AccelerometerVar', 'little movement', 'strong movement']
    features_vector += [mov_variance, little_movement, strong_movement]

    # concatenate and return
    features_vector = np.transpose(np.vstack(features_vector))
    features_df = pd.DataFrame(features_vector, columns=features_name)
    return features_df


def create_dataset_features(record_list):
    """
    generate x and y data from record_list for classification
    """
    # params
    fs = 250.  # Hz
    epoch_size = 30.  # sec

    # generate dataset
    x_data = []
    y_data = []

    for record in record_list:
        # get data
        results = get_one_record(record)
        eeg_data = results['eeg_1']
        hypnogram = results['hypnogram']
        x, y, z = results['x_acc'], results['y_acc'], results['z_acc']

        # extract features
        epochs = np.arange(0, len(hypnogram) + 1) * epoch_size
        features_df = get_feature_representation(eeg_data, [x, y, z], epochs)

        # dataset
        x_data += [features_df]
        y_data += [hypnogram]

    x_data = pd.concat(x_data)
    y_data = np.concatenate(y_data)

    # remove non labelled epochs
    idx_keep = y_data >= 0
    x_data = x_data[idx_keep]
    y_data = y_data[idx_keep]

    return x_data, y_data


def get_train_dataset_features():
    return create_dataset_features(train_records)


def get_test_dataset_features():
    return create_dataset_features(test_records)


if __name__ == "__main__":

    x_train, y_train = get_train_dataset_features()
    x_test, y_test = get_test_dataset_features()

    # save
    import os
    from td_dreem_bin import path_repo
    save_folder = os.path.join(path_repo, "data", "processed/")

    save_path = os.path.join(save_folder, "record_datatrain_features")
    np.savez(save_path, x_train=x_train, y_train=y_train)
    save_path = os.path.join(save_folder, "record_datatest_features")
    np.savez(save_path, x_test=x_test, y_test=y_test)