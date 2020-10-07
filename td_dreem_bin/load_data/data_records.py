""" Load project data
    DataLoader and Dataset for single-channel EEG

"""
import os
import h5py
import numpy as np

from td_dreem_bin import path_repo
data_path = os.path.join(path_repo, "data", "records/")
list_record = [1979683, 1980547, 1994683, 1994698, 1994755]


def get_one_record_spectrogram(record):
    res = {}
    return res


if __name__ == "__main__":
    spectrogram = get_one_record_spectrogram(list_record[0])



