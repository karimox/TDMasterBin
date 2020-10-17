"""
Spectrogram functions.
"""
import numpy as np
import matplotlib.pyplot as plt
from lspopt import spectrogram_lspopt
from matplotlib.colors import Normalize
from td_dreem_bin.utils.utils import datetime_to_nightsec

# Set default font size to 12
plt.rcParams.update({'font.size': 12})

frequency_bands = {
    'delta': [0.5, 4],
    'theta': [4, 8],
    'lowfreq': [0.5, 8],
    'alpha': [8, 12],
    'sigma': [11, 15],
    'beta': [15, 18],
    'kcomp': [0.9, 11]
}


def compute_spectrogram(eeg_data, fs, win_sec=30, fmin=0.5, fmax=18):
    """
    Compute spectrogram from EEG 1D-array
    """
    # Calculate multi-taper spectrogram
    nperseg = int(win_sec * fs)
    assert eeg_data.size > 2 * nperseg, 'Data length must be at least 2 * win_sec.'
    f, t, Sxx = spectrogram_lspopt(eeg_data, fs, nperseg=nperseg, noverlap=0)
    Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

    # Select only relevant frequencies (up to 30 Hz)
    good_freqs = np.logical_and(f >= fmin, f <= fmax)
    Sxx = Sxx[good_freqs, :]
    f = f[good_freqs]

    return Sxx, t, f


def plot_spectrogram(spectrogram_array,
                     times,
                     frequencies,
                     trimperc=2.5,
                     cmap='RdBu_r',
                     axe_plot=None,
                     rescale=3600.,
                     start_time=0,
                     title='spectrogram',
                     colourbar=None):
    """
    plot spectrogram
    """
    # data
    t, f, Sxx = times, frequencies, spectrogram_array
    start_hour = datetime_to_nightsec(start_time)
    if np.isnan(start_hour):
        start_hour = 0
    t = (t + start_hour) / rescale

    # Normalization
    vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
    norm = Normalize(vmin=vmin, vmax=vmax)

    # axes
    if axe_plot is None:
        fig, axs = plt.subplots(1, 1, figsize=(12, 4))
        ax = np.ravel(axs)[0]
    else:
        ax = axe_plot

    im = ax.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True, shading='auto')
    tmp = range(-8, 24, 2)
    ax.set_xticks(tmp)
    ax.set_xticklabels([t % 24 for t in tmp])
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel('Frequency [Hz]')
    ax.set_ylim(0, 15)
    ax.set_title(title)

    # Add colorbar
    if colourbar:
        cbar = plt.colorbar(im, ax=ax, cax=colourbar, shrink=0.95, fraction=0.1, aspect=25)
        cbar.ax.set_ylabel('Log Power (dB / Hz)', rotation=270, labelpad=20)

    if axe_plot is None:
        fig.show()

    return ax


def compute_spectral_features(spectrogram, freq):
    """
    Compute spectral features - frequency bands
    """
    n_bins = spectrogram.shape[1]

    # normalize between 0 and 1
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))

    features_value = {}
    total_power = np.sum(spectrogram, 0)
    for nameband, fband in frequency_bands.items():
        fmin, fmax = fband[0], fband[1]
        idx_f = np.logical_and(freq >= fmin, freq <= fmax)
        power = np.sum(spectrogram[idx_f, :], 0)
        ratio = [value / total_power[i] for i, value in enumerate(power)]

        features_value[nameband] = power
        features_value[nameband + '_r'] = ratio

    # spectral edges
    cumsum_power = np.cumsum(spectrogram / total_power, axis=0)
    SC = freq[[(cumsum_power[:, i] > 0.5).argmax() for i in range(n_bins)]]
    SEF90 = freq[[(cumsum_power[:, i] > 0.9).argmax() for i in range(n_bins)]]
    SEF95 = freq[[(cumsum_power[:, i] > 0.95).argmax() for i in range(n_bins)]]

    features_value['SC'] = SC
    features_value['SEF90'] = SEF90
    features_value['SEF95'] = SEF95

    return features_value


if __name__ == "__main__":
    from datetime import datetime

    from td_dreem_bin.load_data.data_records import get_one_record
    from td_dreem_bin.utils.hypnogram import plot_hypnogram

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
    specg1, t1, freq1 = compute_spectrogram(eeg_1, fs, win_sec=30)
    # specg2, t2, freq2 = compute_spectrogram(eeg_2, fs)

    features_value = compute_spectral_features(specg1, freq1)


    # # plot
    # fig, axs = plt.subplots(3, 1, figsize=(18, 14))
    # axs = np.ravel(axs)
    # rescale = 3600
    #
    # # spectrogram channel 1
    # img1 = plot_spectrogram(specg1, t1, freq1, axe_plot=axs[0], rescale=rescale,
    #                         start_time=start_time, title='Channel 1 - F7-01')
    #
    # # accelerometer
    # mov = plot_accelerometer(accelerometer, axe_plot=axs[1], fs=50.,
    #                          rescale=rescale, start_time=start_time, title='movement')
    # axs[1].set_xlim(axs[0].get_xlim())
    #
    # # hypnogram
    # hyp = plot_hypnogram(hypnogram, axe_plot=axs[2], binsize=30,
    #                      rescale=rescale, start_time=start_time, title='Hypnogram')
    # hyp.set_xlabel('Time [h]')
    #
    # fig.show()
