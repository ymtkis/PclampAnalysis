import numpy as np
from scipy.signal import butter, filtfilt

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
