""" Plot Spectrogram on EEG signals

"""
import numpy as np
from mne.baseline import rescale
from mne.viz import centers_to_edges


def process_spectrograms_data(raw_spectrograms, frequencies):
    """ From raw data to data to plot"""

    timestamps = np.linspace(0, 30, raw_spectrograms.shape[2], endpoint=False)

    spectrograms = np.log10(np.abs(raw_spectrograms))

    dt = np.mean(np.diff(timestamps))
    df = np.mean(np.diff(frequencies))
    edge_times, edge_frequencies = centers_to_edges(timestamps + dt/2, frequencies + df/2)

    return spectrograms, edge_times, edge_frequencies


def average_spectrogram_stage(spectrogram_array, sleep_stages_list, stages):

    sleep_stages_array = np.asarray(sleep_stages_list)[:, 1]
    idx = np.isin(sleep_stages_array, stages)

    all_spectromgram = spectrogram_array[idx]
    spectrogram = np.mean(all_spectromgram, 0)

    return spectrogram


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    from td_dreem_bin.load_data.data_kaggle import load_ytrain
    from td_dreem_bin import path_repo
    save_folder = os.path.join(path_repo, "data", "processed/")

    # sleep stages
    stage_correspondance = {0: "WAKE", 1: "N1", 2: "N2", 3: "DEEP", 4: "REM"}
    sleep_stages = ['N1', 'N2', 'N3', 'REM', 'Wake']

    # load data
    y_train = load_ytrain()
    raw_spectrograms = np.load(os.path.join(save_folder, "spectrogram_morlet1.npy"))
    frequencies = np.load(os.path.join(save_folder, "frequencies1.npy"))

    # Process and normalize spectrogram
    spectrograms, edge_times, edge_frequencies = process_spectrograms_data(raw_spectrograms, frequencies)

    normalized_spectrogram = {}
    # whole night
    normalized_spectrogram['all'] = average_spectrogram_stage(spectrograms, y_train, range(4))
    # by stages
    for stage, stage_label in stage_correspondance.items():
        normalized_spectrogram[stage_label] = average_spectrogram_stage(spectrograms, y_train, stage)

    # plot
    fig, axs = plt.subplots(3, 2, figsize=(9, 7))
    axs = np.ravel(axs)

    ax = axs[0]
    mesh = ax.pcolormesh(edge_times, edge_frequencies, normalized_spectrogram['all'], cmap='jet')
    ax.set_title('TFR all')
    ax.set(ylim=frequencies[[0, -1]], xlabel='Time (s)')

    for stage, stage_label in stage_correspondance.items():
        ax = axs[stage + 1]
        mesh = ax.pcolormesh(edge_times, edge_frequencies,  normalized_spectrogram[stage_label], cmap='jet')
        ax.set_title(stage_label)
        ax.set(ylim=frequencies[[0, -1]], xlabel='Time (s)')

    fig.show()


    # # plot
    # fig, ax = plt.subplots()
    # mesh = ax.pcolormesh(edge_times, edge_frequencies, spectrograms[14], cmap='jet')
    # ax.set_title('TFR all')
    # ax.set(ylim=frequencies[[0, -1]], xlabel='Time (s)')
    # fig.colorbar(mesh)
    # plt.tight_layout()
    #
    # plt.show()

