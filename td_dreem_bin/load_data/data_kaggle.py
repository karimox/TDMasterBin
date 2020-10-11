""" Load project data

"""
import os
import h5py
import numpy as np
import pandas as pd

from td_dreem_bin import path_repo

data_path = os.path.join(path_repo, "data", "kaggle/")

file_xtrain = data_path + "X_train.h5"
file_xtest = data_path + "X_test.h5"
file_ytrain = data_path + "y_train.csv"
file_ytest = data_path + "y_test.csv"

derivation_list = ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'eeg_5', 'eeg_6', 'eeg_7']


def load_x_dataset(filename, derivations=[]):
    """ Load x_train or x_test dataset for selected derivation"""
    if not derivations:
        derivations = derivation_list

    x_data = []
    with h5py.File(filename, "r") as fi:
        for derivation in derivations:
            x_data.append([fi[derivation][()]])
    x_data = np.stack(x_data, axis=2)[0]

    return x_data


def load_xtrain_dataset(derivations=[]):
    """ Load x_train dataset for selected derivation"""
    return load_x_dataset(file_xtrain, derivations=[])


def load_xtest_dataset(derivations=[]):
    """ Load x_train dataset for selected derivation"""
    return load_x_dataset(file_xtest, derivations=[])


def load_ytrain():
    return pd.read_csv(file_ytrain)


def load_ytest():
    return pd.read_csv(file_ytest)


if __name__ == "__main__":
    x_train = load_xtrain_dataset()
    x_test = load_xtest_dataset()
    y_train = load_xtrain_dataset()
    y_test = load_ytest()
