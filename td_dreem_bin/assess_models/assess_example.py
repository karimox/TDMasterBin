""" Test CNN network
"""

if __name__ == "__main__":
    import os
    import torch

    from td_dreem_bin import path_repo
    from td_dreem_bin.load_data.single_channel import get_test_dataset
    from td_dreem_bin.models.Sors2017 import SorsNet
    from td_dreem_bin.utils.scores import score_functions

    # params
    classes = ['Wake', 'N1', 'N2', 'N3', 'REM']
    compute_f1 = score_functions['f1']
    compute_cohen_kappa = score_functions['cohen_kappa']
    compute_accuracy = score_functions['accuracy']

    # datasets
    testloader = get_test_dataset('eeg_4', batch_size=32)
    save_path = os.path.join(path_repo, "predictors/sors_net1.pth")

    net = SorsNet()
    net.load_state_dict(torch.load(save_path))

    with torch.no_grad():
        prediction_list = torch.empty(0)
        label_list = torch.empty(0)
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(5):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))