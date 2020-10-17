"""
    Raw h5 to TP data

"""
import os
import h5py
import numpy as np

from td_dreem_bin.utils.spectrogram import compute_spectrogram
from td_dreem_bin import path_repo

path_records = os.path.join(path_repo, "data", "Records/")
path_outputs = os.path.join(path_repo, "data", "h5_tp/")

list_record = [1934791, 1980547, 1989860, 1994755, 2020368]

fs = 250.


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


if __name__ == "__main__":

    for record in list_record:
        # from h5
        results = get_one_record(record)
        x, y, z = results['x_acc'], results['y_acc'], results['z_acc']
        accelerometer = np.vstack([x, y, z])
        spectrogram, t, freq = compute_spectrogram(results['eeg_1'], fs)

        # write
        name_output = os.path.join(path_outputs, str(record) + '.h5')
        hf = h5py.File(name_output, 'w')

        hf.create_dataset('sampling_frequency', data=fs)
        hf.create_dataset('start_time', data=results['start_time'])
        hf.create_dataset('eeg', data=results['eeg_1'])
        hf.create_dataset('hypnogram', data=results['hypnogram'])
        hf.create_dataset('accelerometer', data=accelerometer)
        hf.create_dataset('spectrogram/spectrogram', data=spectrogram)
        hf.create_dataset('spectrogram/t', data=t)
        hf.create_dataset('spectrogram/freq', data=freq)

        hf.close()
