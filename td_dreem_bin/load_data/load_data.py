""" Load project data

"""
import os
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from td_dreem_bin import path_repo
data_path = os.path.join(path_repo, "data/")

file_xtrain = data_path + "X_train.h5"
file_xtest = data_path + "X_test.h5"
file_ytrain = data_path + "y_train.csv"
file_ytest = data_path + "y_test.csv"


def load_one_signal(electrode, index_absolute, dataset='train'):
    # load one EEG signal
    if dataset == 'train':
        filename = file_xtrain
    else:
        filename = file_xtest

    with h5py.File(filename, "r") as fi:
        eeg_signal = fi[electrode][()][index_absolute]
    return eeg_signal


def h5_to_data(filename):
    # function for loading X input data
    results = {}
    fields = ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'eeg_5', 'eeg_6', 'eeg_7']
    fields += ['x', 'y', 'z']
    fields += ['index', 'index_absolute', 'index_window']

    with h5py.File(filename, "r") as fi:
        for data in fields:
            results[data] = fi[data][()]

    return results


if __name__ == "__main__":

    res = h5_to_data(file_xtrain)

    # y_train = pd.read_csv(file_ytrain)
    # y_test = pd.read_csv(file_ytest)
    #
    # x_train = h5_to_data(file_xtrain)
    # x_test = h5_to_data(file_xtest)





