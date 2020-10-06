""" Test Sors 2017 network with multi channel

"""

if __name__ == "__main__":
    import os
    import torch

    from td_dreem_bin import path_repo
    from td_dreem_bin.load_data.multi_channel import get_test_dataset
    from td_dreem_bin.models.Sors2017_multi import SorsmultiNet
    from td_dreem_bin.utils.scores import score_functions

    # params
    classes = ['Wake', 'N1', 'N2', 'N3', 'REM']
    compute_f1 = score_functions['f1']
    compute_cohen_kappa = score_functions['cohen_kappa']
    compute_accuracy = score_functions['accuracy']

    # datasets
    testloader = get_test_dataset(batch_size=32)
    save_path = os.path.join(path_repo, "predictors/sorsmulti_net1.pth")

    net = SorsmultiNet()
    net.load_state_dict(torch.load(save_path))

    with torch.no_grad():
        prediction_list = torch.empty(0)
        label_list = torch.empty(0)
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            prediction_list = torch.cat([prediction_list, predicted])
            label_list = torch.cat([label_list, labels])

    f1_score = compute_f1(label_list, prediction_list)
    cohen_kappa = compute_cohen_kappa(label_list, prediction_list)
    accuracy = compute_accuracy(label_list, prediction_list)

    print('F1-score = %2d %%' % (f1_score*100))
    print('Cohen Kappa = %2d %%' % (cohen_kappa*100))
    print('Accuracy = %2d %%' % (accuracy*100))

