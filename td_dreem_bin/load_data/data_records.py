""" Load sleep records

"""
import os
import h5py
import numpy as np

from td_dreem_bin import path_repo

path_records = os.path.join(path_repo, "data", "Records/")
test_records = [1908193, 1990201, 1999154, 2010577, 2015858]
train_records = [1955925, 1960988, 1975358, 1979683, 1980547, 1981722, 1982067, 1987838,
                 1990207, 1994683, 1994698, 1994755, 2011881, 2014079, 2014658]
list_record = test_records + train_records


def get_one_record(record):
    """ load one Dreem record"""
    filename = path_records + str(record) + '.h5'
    fields = {
        'eeg_1': 'channel1/visualization',
        'eeg_2': 'channel2/visualization',
        'hypnogram': 'algo/dreemnogram',
        'accelerometer': 'accelerometer/norm',
        'x_acc': 'accelerometer/x',
        'y_acc': 'accelerometer/y',
        'z_acc': 'accelerometer/z'
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
