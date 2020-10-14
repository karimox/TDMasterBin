"""
Sleep event detections functions.
"""
import numpy as np
from yasa import spindles_detect, sw_detect


def detect_spindles(eeg_data, fs):
    """
    Detect sleep spindles from EEG 1D-array
    """
    spindles_event = spindles_detect(eeg_data, sf=fs, remove_outliers=False, freq_sp=[10, 16])
    spindles_df = spindles_event.summary()
    spindles_tmp = np.array(spindles_df['Peak']).astype(dtype=int)
    spindles_magnitude = np.array(spindles_df['RMS'])
    spindles_duration = np.array(spindles_df['Duration'])

    return spindles_tmp, spindles_magnitude, spindles_duration


def detect_slow_waves(eeg_data, fs):
    """
    Detect slow wavesfrom EEG 1D-array
    """
    slowwaves_event = sw_detect(eeg_data, sf=fs, remove_outliers=True, freq_sw=[1, 4])
    slowwaves_df = slowwaves_event.summary()
    slowwaves_tmp = np.array(slowwaves_df['Start']).astype(dtype=int)
    slowwaves_magnitude = np.array(slowwaves_df['ValNegPeak'])
    slowwaves_duration = np.array(slowwaves_df['Duration'])

    return slowwaves_tmp, slowwaves_magnitude, slowwaves_duration


def compute_spindles_features(eeg_data, fs, epochs):
    """
    Compute spindles feature representation on each epochs
    count, mean amplitude, total duration
    """
    spindles_tmp, spindles_magnitude, spindles_duration = detect_spindles(eeg_data, fs)
    ss_count, ss_magnitude, ss_durations = [], [], []
    for k in range(len(epochs) - 1):
        idx = np.logical_and(spindles_tmp >= epochs[k], spindles_tmp < epochs[k + 1])
        ss_count += [np.sum(idx)]
        if np.any(idx):
            ss_magnitude += [np.mean(spindles_magnitude[idx])]
            ss_durations += [np.sum(spindles_duration[idx])]
        else:
            ss_magnitude += [0]
            ss_durations += [0]

    return ss_count, ss_magnitude, ss_durations


def compute_slowwaves_features(eeg_data, fs, epochs):
    """
    Compute slow waves feature representation on each epochs
    count, mean amplitude, total duration
    """
    slowwaves_tmp, slowwaves_magnitude, slowwaves_duration = detect_slow_waves(eeg_data, fs)
    sw_count, sw_magnitude, sw_durations = [], [], []
    for k in range(len(epochs) - 1):
        idx = np.logical_and(slowwaves_tmp >= epochs[k], slowwaves_tmp < epochs[k + 1])
        sw_count += [np.sum(idx)]
        if np.any(idx):
            sw_magnitude += [np.mean(slowwaves_magnitude[idx])]
            sw_durations += [np.sum(slowwaves_duration[idx])]
        else:
            sw_magnitude += [0]
            sw_durations += [0]

    return sw_count, sw_magnitude, sw_durations


if __name__ == "__main__":
    from datetime import datetime

    from td_dreem_bin.load_data.data_records import get_one_record

    # data
    list_record = [1979683, 1980547, 1994683, 1994698, 1994755]
    results = get_one_record(list_record[0])
    eeg_1 = results['eeg_1']
    eeg_2 = results['eeg_2']
    hypnogram = results['hypnogram']
    accelerometer = results['accelerometer']
    start_time = datetime.fromtimestamp(results['start_time'])
    fs = 250.
    epoch_size = 30.

    # spindles and slow waves features
    epochs = np.arange(0, len(hypnogram) + 1) * epoch_size
    ss_count, ss_magnitude, ss_durations = compute_spindles_features(eeg_1, fs, epochs)
    sw_count, sw_magnitude, sw_durations = compute_slowwaves_features(eeg_1, fs, epochs)


