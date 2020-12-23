""" Load project data
    DataLoader and Dataset for single-channel EEG

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
file_ytrain = data_path + "y_train.csv"


def normalize_data(eeg_array):
    """normalize signal between 0 and 1"""

    normalized_array = np.clip(eeg_array, -150, 150)
    normalized_array = normalized_array / 150

    return normalized_array


class EegEpochDataset(Dataset):
    """EEG Epochs dataset."""

    def __init__(self, x_data, y_data, transform=None):
        """
        Args:
            x_data (numpy array): Numpy array of input data.
            y_data (list of numpy array): Sleep Stages
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.y_data = y_data
        self.x_data = x_data
        self.transform = transform

        self.x_data = normalize_data(x_data)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        signal = np.expand_dims(self.x_data[idx], axis=0)
        stage = self.y_data[idx]

        if self.transform:
            signal = self.transform(signal)

        return signal, stage


def get_train_validation_dataset(derivation, batch_size=32, validation_ratio=0.2):
    """
    Return train and validation datasets in Dataloader format
    :param derivation: EEG derivation, from eeg_1 to eeg_7
    :param batch_size: size of the batch, usually 16, 3é or 64
    :param validation_ratio:

    :return:
    train_dataloader
    validation_dataloader
    """

    with h5py.File(file_xtrain, "r") as fi:
        x_data = fi[derivation][()]
    y_data = pd.read_csv(file_ytrain)['sleep_stage'].to_numpy()

    # Creating data indices for training and validation splits:
    dataset_size = len(y_data)
    indices = list(range(dataset_size))
    split = int((1 - validation_ratio) * dataset_size)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    x_train, x_validation = x_data[train_indices], x_data[val_indices]
    y_train, y_validation = y_data[train_indices], y_data[val_indices]

    # torch dataset
    train_dataset = EegEpochDataset(x_data=x_train, y_data=y_train)
    val_dataset = EegEpochDataset(x_data=x_validation, y_data=y_validation)

    # to dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader


if __name__ == "__main__":

    train_dataloader, validation_dataloader = get_train_validation_dataset('eeg_4', batch_size=32)
    dataiter = iter(train_dataloader)
    signals, labels = dataiter.next()
