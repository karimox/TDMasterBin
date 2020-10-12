""" Load sleep records

"""
import os
import h5py
import numpy as np

from td_dreem_bin import path_repo

path_records = os.path.join(path_repo, "data", "Records/")
list_record = [1979683, 1980547, 1994683, 1994698, 1994755]


def get_one_record(record):
    """ load one Dreem record"""
    filename = path_records + str(record) + '.h5'
    fields = {
        'eeg_1': 'channel1/visualization',
        'eeg_2': 'channel2/visualization',
        'hypnogram':'algo/dreemnogram',
        'accelerometer': 'accelerometer/norm'
    }

    results = {}
    with h5py.File(filename, "r") as fi:
        for key, field in fields.items():
            results[key] = fi[field][()]
        results['start_time'] = fi.attrs['start_time']
    return results


def get_one_record_hypnogram(record):
    """ load hypnogram for one record: computed with Dreem algorithm"""
    filename = path_records + str(record) + '.h5'
    with h5py.File(filename, "r") as fi:
        hypnogram = fi['algo/dreemnogram'][:]
        start_time = fi.attrs['start_time']

    return hypnogram, start_time


def load_one_record_spectrogram(record):
    """ load spectrograms for one record: channel 1 & 2"""
    file_spectrogram = path_records + str(record) + '_spectrogram.h5'

    with h5py.File(file_spectrogram, "r") as fi:
        t1 = fi['/channel1/time'][()]
        freq1 = fi['/channel1/freq'][()]
        spectrogram1 = fi['/channel1/spectrogram'][()]

        t2 = fi['/channel2/time'][()]
        freq2 = fi['/channel2/freq'][()]
        spectrogram2 = fi['/channel2/spectrogram'][()]

    return t1, freq1, spectrogram1, t2, freq2, spectrogram2


if __name__ == "__main__":
    results = get_one_record(list_record[0])
