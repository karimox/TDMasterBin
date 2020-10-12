""" Work on hypnograms and spectrograms

"""
import numpy as np
import matplotlib.pyplot as plt
from td_dreem_bin.utils.hypnogram import datetime_to_nightsec

def plot_spectrogram(
        spectrogram,
        timestamps,
        frequencies,
        vborders=None,
        axe_plot=None,
        rescale=3600,
        start_time=0,
        title='Hypnogram',
):
    # data
    start_hour = datetime_to_nightsec(start_time)
    if np.isnan(start_hour):
        start_hour = 0

    MatSpectrogram = np.log10(spectrogram)
    real_x = (timestamps + start_hour) / rescale
    real_y = frequencies
    dx = (real_x[1] - real_x[0]) / 2.
    dy = (real_y[1] - real_y[0]) / 2.
    extent = [real_x[0] - dx, real_x[-1] + dx, real_y[0] - dy, real_y[-1] + dy]

    # plot
    if axe_plot is None:
        fig, axs = plt.subplots(1, 1, figsize=(9, 7))
        ax = np.ravel(axs)[0]
    else:
        ax = axe_plot

    ax.set_title(title)
    if vborders:
        vmin = vborders[0]
        vmax = vborders[1]
        ax.imshow(MatSpectrogram, origin="lower", cmap='jet', extent=extent, aspect='auto', vmin=vmin,  vmax=vmax)
    else:
        ax.imshow(MatSpectrogram, origin="lower", cmap='jet', extent=extent, aspect='auto')

    tmp = range(-8, 24, 2)
    ax.set_xticks(tmp)
    ax.set_xticklabels([t % 24 for t in tmp])
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], 18)

    if axe_plot is None:
        fig.show()

    return ax


if __name__ == "__main__":
    from td_dreem_bin.load_data.data_records import load_one_record_spectrogram, get_one_record_hypnogram
    from td_dreem_bin.utils.hypnogram import plot_hypnogram
    from datetime import datetime

    # data
    list_record = [1979683, 1980547, 1994683, 1994698, 1994755]
    record = list_record[1]
    t1, freq1, specg1, t2, freq2, specg2 = load_one_record_spectrogram(record)
    hypnogram, start_time = get_one_record_hypnogram(record)
    start_time = datetime.fromtimestamp(start_time)

    # plot
    fig, axs = plt.subplots(3, 1, figsize=(9, 7))
    axs = np.ravel(axs)
    rescale = 3600

    # spectrogram channel 1
    img1 = plot_spectrogram(specg1, t1, freq1, axe_plot=axs[0], vborders=[0, 4],
                            rescale=rescale, start_time=start_time, title='Channel 1 - F7-01')
    # spectrogram channel 2
    img2 = plot_spectrogram(specg2, t2, freq2, axe_plot=axs[1],  vborders=[0, 4],
                            rescale=rescale, start_time=start_time, title='Channel 2 - F8-02')

    # hypnogram
    hyp = plot_hypnogram(hypnogram, axe_plot=axs[2], binsize=30,
                         rescale=rescale, start_time=start_time, title='Hypnogram')

    fig.show()
