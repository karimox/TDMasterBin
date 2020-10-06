""" Inspired from Sors et al. 2017
    Full convolutional network from raw single-channel EEG
"""
import torch
import torch.nn as nn


class SorsmultiNet(nn.Module):

    def __init__(self):
        super(SorsmultiNet, self).__init__()
        self.conv_a = nn.Conv1d(7, 128, 7, stride=2, padding=6, padding_mode='zeros')
        self.conv_b = nn.Conv1d(128, 128, 7, stride=2, padding=6, padding_mode='zeros')
        self.conv_c = nn.Conv1d(128, 256, 7, stride=2, padding=6, padding_mode='zeros')
        self.conv_d = nn.Conv1d(256, 256, 5, stride=2, padding=4, padding_mode='zeros')
        self.conv_e = nn.Conv1d(256, 256, 3, stride=2, padding=2, padding_mode='zeros')

        self.pool = nn.MaxPool1d(2)

        self.activfunc_a = nn.LeakyReLU(negative_slope=0.1)

        self.fc1 = nn.Linear(3 * 256, 100)
        self.fc2 = nn.Linear(100, 5)

        self.drop_layer = nn.Dropout(p=0.5)

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
        x = self.drop_layer(x)
        x = self.activfunc_a(self.fc1(x))
        x = self.drop_layer(x)
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
    from td_dreem_bin.load_data.multi_channel import get_train_dataset

    #datasets
    trainloader = get_train_dataset(batch_size=32)

    # neural network and co
    net = SorsmultiNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    print('training...')
    for epoch in range(4):  # loop over the dataset multiple times

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

    save_path = os.path.join(path_repo, "predictors/sorsmulti_net1.pth")
    torch.save(net.state_dict(), save_path)
