""" Spectrum on EEG signals

"""
import numpy as np
import matplotlib.pyplot as plt
from td_dreem_bin.utils.hypnogram import Hypnogram


def average_spectrum_stage(spectrogram, timestamps, hypnogram_list, stages, binsize=30):

    hypno = Hypnogram(hypnogram_list, increment_duration=binsize)
    stage_epochs = np.asarray(hypno.get_stage_epochs(stages=stages))

    idx = [i for (i, t) in enumerate(timestamps) for epoch in stage_epochs if epoch[0] <= t < epoch[1]]
    all_spectromgram = spectrogram[:, idx]
    spectrum = np.mean(all_spectromgram, 1)

    return spectrum


if __name__ == "__main__":
    from td_dreem_bin.load_data.data_records import load_one_record_spectrogram, get_one_record_hypnogram

    # sleep stages
    stage_correspondance = {0: "WAKE", 1: "N1", 2: "N2", 3: "DEEP", 4: "REM"}
    sleep_stages = ['N1', 'N2', 'N3', 'REM', 'Wake']


    # data
    list_record = [1979683, 1980547, 1994683, 1994698, 1994755]
    record = list_record[2]
    t1, freq1, specg1, t2, freq2, specg2 = load_one_record_spectrogram(record)
    hypnogram_list, start_time = get_one_record_hypnogram(record)

    # Normalized spectrum
    normalized_spectrum = {}
    # whole night
    night_spectrum1 = np.mean(specg1, 1)
    normalized_spectrum['all'] = freq1 * night_spectrum1
    normalized_spectrum['all'] = np.log10(night_spectrum1)

    # by stages
    for stage, stage_label in stage_correspondance.items():
        spectrum = average_spectrum_stage(specg1, t1, hypnogram_list, stage, binsize=30)
        normalized_spectrum[stage_label] = np.log10(spectrum)

    # plot
    fig, axs = plt.subplots(3, 2, figsize=(9, 7))
    axs = np.ravel(axs)

    axs[0].plot(freq1, normalized_spectrum['all'])
    axs[0].set_xlim(0, 15)

    for stage, stage_label in stage_correspondance.items():
        ax = axs[stage + 1]
        ax.plot(freq1, normalized_spectrum[stage_label])
        ax.set_xlim(0, 15)
        ax.set_title(stage_label)

    fig.show()
