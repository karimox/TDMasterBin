""" Load project data

"""
import os
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from td_dreem_bin import path_repo
data_path = os.path.join(path_repo, "data", "kaggle/")

file_xtrain = data_path + "X_train.h5"
file_xtest = data_path + "X_test.h5"
file_ytrain = data_path + "y_train.csv"
file_ytest = data_path + "y_test.csv"


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

        # shape
        with h5py.File(self.x_filename, "r") as fi:
            signal = fi[self.fields[0]][()]
            self.nb_samples = signal.shape[0]
            self.signal_length = signal.shape[1]
        self.nb_channels = len(self.fields)

        self.x_train = []
        with h5py.File(self.x_filename, "r") as fi:
            for field in self.fields:
                self.x_train.append([fi[field][()]])
        self.x_train = np.stack(self.x_train, axis=2)[0]

    def __len__(self):
        return len(self.y_stages)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        signal = self.x_train[idx]
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

    trainloader = get_train_dataset()
    dataiter = iter(trainloader)
    signals, labels = dataiter.next()
