""" Work on hypnograms and spectrograms

"""
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    from td_dreem_bin.load_data.data_records import get_one_record
    from td_dreem_bin.utils.accelerometer import plot_accelerometer
    from td_dreem_bin.utils.hypnogram import plot_hypnogram
    from td_dreem_bin.utils.spectrogram import compute_spectrogram, plot_spectrogram

    from datetime import datetime

    # data
    list_record = [1979683, 1980547, 1994683, 1994698, 1994755]
    record = list_record[2]
    results = get_one_record(list_record[1])
    eeg_1 = results['eeg_1']
    eeg_2 = results['eeg_2']
    hypnogram = results['hypnogram']
    accelerometer = results['accelerometer']
    start_time = datetime.fromtimestamp(results['start_time'])
    fs = 250.

    #spectrogram
    specg1, t1, freq1 = compute_spectrogram(eeg_1, fs)
    specg2, t2, freq2 = compute_spectrogram(eeg_2, fs)

    # plot
    fig, axs = plt.subplots(3, 1, figsize=(18, 14))
    axs = np.ravel(axs)
    rescale = 3600

    # spectrogram channel 1
    img1 = plot_spectrogram(specg1, t1, freq1, axe_plot=axs[0], rescale=rescale,
                            start_time=start_time, title='Channel 1 - F7-01')

    # accelerometer
    mov = plot_accelerometer(accelerometer, axe_plot=axs[1], fs=50.,
                             rescale=rescale, start_time=start_time, title='movement')
    axs[1].set_xlim(axs[0].get_xlim())

    # hypnogram
    hyp = plot_hypnogram(hypnogram, axe_plot=axs[2], binsize=30,
                         rescale=rescale, start_time=start_time, title='Hypnogram')
    hyp.set_xlabel('Time [h]')

    fig.show()
