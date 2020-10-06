""" Test CNN network

"""

if __name__ == "__main__":
    import os
    import torch

    from td_dreem_bin import path_repo
    from td_dreem_bin.load_data.load_data import get_test_dataset
    from td_dreem_bin.models.CNN_example import Net

    classes = ['Wake', 'N1', 'N2', 'N3', 'REM']
    # datasets
    testloader = get_test_dataset()
    save_path = os.path.join(path_repo, "predictors/karim_net1.pth")

    net = Net()
    net.load_state_dict(torch.load(save_path))

    class_correct = list(0. for i in range(5))
    class_total = list(0. for i in range(5))
    with torch.no_grad():
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