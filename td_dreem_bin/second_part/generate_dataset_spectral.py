"""
Generate two datasets from spectrogram
"""

if __name__ == "__main__":
    from td_dreem_bin.utils.spectrogram import compute_spectrogram
    from td_dreem_bin.load_data.data_records import get_one_record

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

