"""
Plotting functions of YASA.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from td_dreem_bin.utils.utils import datetime_to_nightsec, array_to_times


def compute_movement_variance(data, fs=50., buff_len_sec=1, computation_delay=0.1):
    """
    variance from accelerometer data
    """

    data = pd.DataFrame(data, columns=['x', 'y', 'z'])
    npts_var = int(fs * buff_len_sec)
    mov_variance = data.rolling(npts_var).var()
    npts_compute = int(fs * computation_delay)
    mov_variance = mov_variance.groupby(mov_variance.index // npts_compute).max()
    mov_variance = mov_variance.to_numpy()

    sampling_frequency = 1. / computation_delay

    return mov_variance, sampling_frequency


def movement_from_variance(mov_variance, thresh_var_light=0.000186, thresh_var_strong=0.02):

    little_movement = np.zeros(mov_variance.shape[0])
    strong_movement = np.zeros(mov_variance.shape[0])

    max_variance = np.max(mov_variance, 1)
    idx_little = np.logical_and(max_variance >= thresh_var_light, max_variance <= thresh_var_strong)
    idx_strong = max_variance >= thresh_var_strong

    little_movement[idx_little] = 1
    strong_movement[idx_strong] = 1

    return little_movement, strong_movement


def compute_accelerometer_features(x, y, z, epochs, fs=50.,
                                   buff_len_sec=1, computation_delay=0.1,
                                   thresh_var_light=0.000186, thresh_var_strong=0.02):
    """
    Compute accelerometer features - movement
    """

    data = np.transpose(np.vstack([x, y, z]))
    mov_variance, sampling_frequency = compute_movement_variance(data,
                                                                 fs=fs,
                                                                 buff_len_sec=buff_len_sec,
                                                                 computation_delay=computation_delay)
    little_movement, strong_movement = movement_from_variance(mov_variance,
                                                              thresh_var_light=thresh_var_light,
                                                              thresh_var_strong=thresh_var_strong)

    # to epochs
    timestamps = epochs[:-1]
    n_epochs = len(timestamps)
    epochsize = np.mean(np.diff(epochs))
    nperwin = epochsize * sampling_frequency

    df = pd.DataFrame(mov_variance, columns=['x', 'y', 'z'])
    mov_variance = df.groupby(df.index // nperwin).mean().to_numpy()[:n_epochs]
    df = pd.Series(little_movement)
    little_movement = df.groupby(df.index // nperwin).sum().to_numpy()[:n_epochs]
    df = pd.Series(strong_movement)
    strong_movement = df.groupby(df.index // nperwin).sum().to_numpy()[:n_epochs]

    return mov_variance, little_movement, strong_movement, timestamps


def plot_accelerometer(accelerometer_array,
                       axe_plot=None,
                       fs=50.,
                       rescale=3600.,
                       start_time=0,
                       color='k',
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

    ax.plot(t, accelerometer_array, color=color)

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
    start_time = datetime.fromtimestamp(results['start_time'])
    hypnogram = results['hypnogram']
    epoch_size = 30.
    epochs = np.arange(0, len(hypnogram) + 1) * epoch_size

    accelerometer, x, y, z = results['accelerometer'], results['x_acc'], results['y_acc'], results['z_acc']
    mov_variance, little_movement, strong_movement, timestamps = compute_accelerometer_features(
        x, y, z, epochs, fs=50.,
        buff_len_sec=1, computation_delay=0.1,
        thresh_var_light=0.000186, thresh_var_strong=0.02)

    # plot
    fig, axs = plt.subplots(3, 1, figsize=(18, 14))
    axs = np.ravel(axs)
    rescale = 3600
    sampling_frequency = 1. / np.mean(np.diff(timestamps))

    # accelerometer
    mov1 = plot_accelerometer(accelerometer, axe_plot=axs[0], fs=50.,
                              rescale=rescale, start_time=start_time, title='movement')

    mov2 = plot_accelerometer(little_movement, axe_plot=axs[1], fs=sampling_frequency, color='b',
                              rescale=rescale, start_time=start_time, title='movement variance')
    mov3 = plot_accelerometer(strong_movement * 2, axe_plot=axs[1], fs=sampling_frequency,  color='r',
                              rescale=rescale, start_time=start_time, title='movement variance')

    # hypnogram
    hyp = plot_hypnogram(hypnogram, axe_plot=axs[2], binsize=30,
                         rescale=rescale, start_time=start_time, title='Hypnogram')
    hyp.set_xlabel('Time [h]')

    fig.show()
