import numpy as np
from scipy.signal import butter, iirnotch, filtfilt


def highpass_filter(data, fs, cutoff, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, data)

def lowpass_filter(data, fs, cutoff=2000, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)

def bandpass_filter(data, fs, lowcut, highcut, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)

def notch_filter(data, fs, freq=50.0, Q=30.0):
    b, a = iirnotch(freq / (fs / 2), Q)
    return filtfilt(b, a, data)

def remove_hum_harmonics(data, fs, base_freq=50.0, max_freq=500.0, Q=30.0):
    out = data.copy()
    for k in range(1, int(max_freq // base_freq) + 1):
        out = notch_filter(out, fs, freq=base_freq * k, Q=Q)
    return out