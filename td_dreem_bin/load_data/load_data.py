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


class EegEpochDataset(Dataset):
    """EEG Epochs dataset."""

    def __init__(self, x_h5file, y_csv_file, transform=None):
        """
        Args:
            x_h5file (string): Path to the h5 file with EEG signals.
            y_csv_file (string): Path to the csv file with sleep stages.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.y_stages = pd.read_csv(y_csv_file)
        self.x_filename = x_h5file
        self.transform = transform

        self.fields = ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4', 'eeg_5', 'eeg_6', 'eeg_7']
        self.fields += ['x', 'y', 'z']

        self.x_train = {}
        with h5py.File(x_h5file, "r") as fi:
            for field in self.fields:
                self.x_train[field] = fi[field][()]
        self.signal_length = len(self.x_train[self.fields[0]][0])

    def __len__(self):
        return len(self.y_stages)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        signal = torch.empty(len(self.fields), self.signal_length)
        for field in self.fields:
            signal[field] = self.x_train[field][idx]
        stage = self.y_stages['sleep_stage'][idx]

        if self.transform:
            signal = self.transform(signal)

        return signal, stage


def get_train_dataset(batch_size=32, num_workers=2):
    """Return train dataset in Dataloader format """
    eeg_dataset = EegEpochDataset(x_h5file=file_xtrain, y_csv_file=file_ytrain)
    dataloader = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


def get_test_dataset(batch_size=32, num_workers=2):
    """Return test dataset in Dataloader format """
    eeg_dataset = EegEpochDataset(x_h5file=file_xtest, y_csv_file=file_ytest)
    dataloader = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


if __name__ == "__main__":

    eeg_dataset = EegEpochDataset(x_h5file=file_xtrain, y_csv_file=file_ytrain)
    trainloader = DataLoader(eeg_dataset, batch_size=8, shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    signals, labels = dataiter.next()

    # # read csv
    # y_train = pd.read_csv(file_ytrain)
    # y_test = pd.read_csv(file_ytest)
    #
    # x_train = h5_to_data(file_xtrain)
    # x_test = h5_to_data(file_xtest)





