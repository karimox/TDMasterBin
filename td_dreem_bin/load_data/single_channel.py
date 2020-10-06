""" Load project data
    DataLoader and Dataset for signle-channel EEG

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


def normalize_data(eeg_array):
    """normalize signal between 0 and 1"""

    normalized_array = np.clip(eeg_array, -150, 150)
    normalized_array = normalized_array / 150

    return normalized_array


class EegEpochDataset(Dataset):
    """EEG Epochs dataset."""

    def __init__(self, x_h5file, y_csv_file, derivation, transform=None):
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
            self.x_train = normalize_data(fi[derivation][()])

    def __len__(self):
        return len(self.y_stages)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        signal = np.expand_dims(self.x_train[idx], axis=0)
        stage = self.y_stages['sleep_stage'][idx]

        if self.transform:
            signal = self.transform(signal)

        return signal, stage


def get_train_dataset(derivation, batch_size=32, num_workers=2):
    """Return train dataset in Dataloader format """
    eeg_dataset = EegEpochDataset(x_h5file=file_xtrain, y_csv_file=file_ytrain, derivation=derivation)
    dataloader = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


def get_test_dataset(derivation, batch_size=32, num_workers=2):
    """Return test dataset in Dataloader format """
    eeg_dataset = EegEpochDataset(x_h5file=file_xtest, y_csv_file=file_ytest, derivation=derivation)
    dataloader = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


if __name__ == "__main__":

    trainloader = get_train_dataset('eeg_4')
    dataiter = iter(trainloader)
    signals, labels = dataiter.next()
