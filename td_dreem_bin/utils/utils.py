""" List of utils functions

"""
# encoding: utf-8


def datetime_to_nightsec(d_time):
    try:
        new_time = d_time.hour * 3600 + d_time.minute * 60 + d_time.second
        if new_time > 16 * 3600:
            new_time -= 24 * 3600
        return new_time

    except:
        return float('nan')
