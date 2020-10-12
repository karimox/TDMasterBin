"""
Plotting functions of YASA.
"""
import numpy as np
import matplotlib.pyplot as plt
from td_dreem_bin.utils.hypnogram import datetime_to_nightsec


def plot_accelerometer(accelerometer_array,
                       axe_plot=None,
                       fs=50.,
                       rescale=3600.,
                       start_time=0,
                       title='movement'):
    """
    plot accelerometer norm
    """

    start_hour = datetime_to_nightsec(start_time)
    if np.isnan(start_hour):
        start_hour = 0
    t = np.arange(0, len(accelerometer_array)) / fs
    t = (t + start_hour) / rescale

    if axe_plot is None:
        fig.show()

    # axes
    if axe_plot is None:
        fig, axs = plt.subplots(1, 1, figsize=(12, 4))
        ax = np.ravel(axs)[0]
    else:
        ax = axe_plot

    ax.plot(t, accelerometer_array, color='k')

    tmp = range(-8, 24, 2)
    ax.set_xticks(tmp)
    ax.set_xticklabels([t % 24 for t in tmp])
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel('accelero norm')
    ax.set_title(title)

    return ax


if __name__ == "__main__":
    from td_dreem_bin.load_data.data_records import get_one_record
    from td_dreem_bin.utils.hypnogram import plot_hypnogram

    from datetime import datetime

    # data
    list_record = [1979683, 1980547, 1994683, 1994698, 1994755]
    record = list_record[2]
    results = get_one_record(list_record[1])
    hypnogram = results['hypnogram']
    accelerometer = results['accelerometer']
    start_time = datetime.fromtimestamp(results['start_time'])

    # plot
    fig, axs = plt.subplots(2, 1, figsize=(18, 14))
    axs = np.ravel(axs)
    rescale = 3600

    # accelerometer
    mov = plot_accelerometer(accelerometer, axe_plot=axs[0], fs=50.,
                             rescale=rescale, start_time=start_time, title='movement')

    # hypnogram
    hyp = plot_hypnogram(hypnogram, axe_plot=axs[1], binsize=30,
                         rescale=rescale, start_time=start_time, title='Hypnogram')
    hyp.set_xlabel('Time [h]')

    fig.show()