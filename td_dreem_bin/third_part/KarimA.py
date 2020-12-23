""" Karim A
    %run td_dreem_bin/third_part/KarimA.py
"""
import os
import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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


class KarimANet(nn.Module):

    def __init__(self):
        super(KarimANet, self).__init__()
        self.conv_a = nn.Conv1d(1, 128, 7, stride=2, padding=6, padding_mode='zeros')
        self.conv_b = nn.Conv1d(128, 128, 7, stride=2, padding=6, padding_mode='zeros')
        self.conv_c = nn.Conv1d(128, 256, 7, stride=2, padding=6, padding_mode='zeros')
        self.conv_d = nn.Conv1d(256, 256, 5, stride=2, padding=4, padding_mode='zeros')
        self.conv_e = nn.Conv1d(256, 256, 3, stride=2, padding=2, padding_mode='zeros')

        self.pool = nn.MaxPool1d(2)

        self.activfunc_a = nn.LeakyReLU(negative_slope=0.1)

        self.fc1 = nn.Linear(3 * 256, 100)
        self.fc2 = nn.Linear(100, 5)

    def forward(self, x):

        x = self.activfunc_a(self.conv_a(x))
        for _ in range(5):
            x = self.activfunc_a(self.conv_b(x))
        x = self.activfunc_a(self.conv_c(x))
        for _ in range(3):
            x = self.activfunc_a(self.conv_d(x))
        x = self.activfunc_a(self.conv_e(x))
        x = self.activfunc_a(self.conv_e(x))

        x = x.view(-1, self.num_flat_features(x))
        x = self.activfunc_a(self.fc1(x))
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    import os
    import torch.optim as optim

    from td_dreem_bin import path_repo
    from td_dreem_bin.load_data.single_channel import get_train_dataset
    from td_dreem_bin.load_data.kaggle_loader import get_train_validation_dataset

    #datasets
    get_train_validation_dataset('eeg_4', batch_size=32)
    trainloader = get_train_dataset('eeg_4', batch_size=32)
    trainloader, _ = get_train_validation_dataset('eeg_4', batch_size=32)

    # neural network and co
    net = SorsNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    print('training...')
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    save_path = os.path.join(path_repo, "predictors/karimA_net1.pth")
    torch.save(net.state_dict(), save_path)
