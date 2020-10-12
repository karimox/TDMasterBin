"""
Plotting functions of YASA.
"""
import numpy as np
import matplotlib.pyplot as plt
from lspopt import spectrogram_lspopt
from matplotlib.colors import Normalize

# Set default font size to 12
plt.rcParams.update({'font.size': 12})


def compute_spectrogram(eeg_data, fs, win_sec=30, fmin=0.5, fmax=18):
    """
    Compute spectrogram from EEG 1D-array
    """
    # Calculate multi-taper spectrogram
    nperseg = int(win_sec * fs)
    assert eeg_data.size > 2 * nperseg, 'Data length must be at least 2 * win_sec.'
    f, t, Sxx = spectrogram_lspopt(eeg_data, sf, nperseg=nperseg, noverlap=0)
    Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

    # Select only relevant frequencies (up to 30 Hz)
    good_freqs = np.logical_and(f >= fmin, f <= fmax)
    Sxx = Sxx[good_freqs, :]
    f = f[good_freqs]

    return Sxx, t, f


def plot_spectrogram(spectrogram_array, times, frequencies, trimperc=2.5, cmap='RdBu_r'):
    """
    plot spectrogram
    """
    t, f, Sxx = times, frequencies, spectrogram_array

    # Normalization
    vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(nrows=1, figsize=(12, 4))
    im = ax.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True)
    ax.set_xlim(0, t.max())
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.95, fraction=0.1, aspect=25)
    cbar.ax.set_ylabel('Log Power (dB / Hz)', rotation=270, labelpad=20)
    return fig


if __name__ == "__main__":
    from td_dreem_bin.load_data.data_records import get_one_record
    from datetime import datetime

    # data
    list_record = [1979683, 1980547, 1994683, 1994698, 1994755]
    record = list_record[1]
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