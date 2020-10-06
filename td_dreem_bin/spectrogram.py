""" Spectrogram functions

"""

from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
from lspopt.lsp import spectrogram_lspopt

def plot_spectrogram(eeg_signal, fs=250):
    # plot spectrogram of EEG signal
    f, t, spectrogram = spectrogram_lspopt(eeg_signal, fs, c_parameter=20.0)


    plt.pcolormesh(t, f, spectrogram, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    return 1


def plot_spectrum(eeg_signal, fs=250):
    # plot spectrogram of EEG signal
    f, spectrum = signal.welch(eeg_signal, fs, nperseg=1024)

    plt.semilogy(f, spectrum)
    plt.ylim([0.1e-4, 1000])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

    return 1


if __name__ == "__main__":

    import os
    import h5py
    import pandas as pd
    from td_dreem_bin import path_repo

    from td_dreem_bin.load_data import load_one_signal

    eeg_signal = load_one_signal('eeg_4', 1000)
    plot_spectrum(eeg_signal)
