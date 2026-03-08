import numpy as np
from scipy.signal import find_peaks
from .filters import bandpass_filter

def compute_onset_offset(t, v, peak_idx, dt, dvdt_onset_thr=30.0, window_ms=5.0):
    window = 4
    diff_vec = np.zeros_like(v)
    diff_vec[window:] = (v[window:] - v[:-window]) / (window * dt) / 1000.0
    dvdt = diff_vec

    Vpeak = v[peak_idx]
    onset_idx = None
    for i in range(peak_idx - 200, peak_idx):
        if v[i] < Vpeak - 10.0 and dvdt[i] >= dvdt_onset_thr:
            onset_idx = i
            break

    offset_idx = None
    if onset_idx is not None:
        target_v = v[onset_idx]
        win_samples = int((window_ms / 1000) / dt)
        for j in range(peak_idx + 1, min(len(v), peak_idx + win_samples)):
            if v[j] <= target_v:
                offset_idx = j
                break

    return onset_idx, offset_idx


def extract_holding_currents(abf_path, step_start=0.18, step_duration=1.0, command_channel=1):
    import pyabf
    abf = pyabf.ABF(abf_path)
    fs = abf.dataRate
    start = int(step_start * fs)
    end = int((step_start + step_duration) * fs)

    currents = []
    for i in range(abf.sweepCount):
        abf.setSweep(i, channel=command_channel)
        currents.append(round(np.mean(abf.sweepY[start:end])))

    if currents[0] > 1000:
        currents = [round(i / 5) for i in currents]

    return currents


def ICstep_detect_events(sweep_data, holding_currents, fs, sweep_start, sweep_duration):
    event_counts = []
    event_times_all = []
    amp, amp_onset, half_dur, rise, fall = [], [], [], [], []

    sweep_start_idx = int(sweep_start * fs)
    sweep_end_idx = int((sweep_start + sweep_duration) * fs)
    t = np.arange(sweep_data.shape[1]) / fs
    dt = 1.0 / fs

    rmp = np.mean(sweep_data[:, :sweep_start_idx])

    for sweep in sweep_data:
        segment = sweep[sweep_start_idx:sweep_end_idx]
        filtered = bandpass_filter(segment, fs, 0.1, 1000)
        baseline = bandpass_filter(segment, fs, 2000, 4000)
        delta = filtered - baseline

        peaks, _ = find_peaks(delta, height=40)
        spikes = [(sweep_start_idx + p) for p in peaks if (sweep_start_idx + p) / fs >= 0.185]

        event_counts.append(len(spikes))
        event_times_all.append([i / fs for i in spikes])

        for peak_idx in spikes:
            onset, offset = compute_onset_offset(t, sweep, peak_idx, dt)
            if onset is None:
                continue

            amp.append(sweep[peak_idx] - rmp)
            amp_onset.append(sweep[peak_idx] - sweep[onset])
            half_dur.append((t[peak_idx] - t[onset]) * 1000)

            rise_t = t[peak_idx] - t[onset]
            rise.append(amp_onset[-1] / rise_t / 1000 if rise_t > 0 else np.nan)

            if offset and offset > peak_idx:
                fall_t = t[offset] - t[peak_idx]
                fall.append((sweep[offset] - sweep[peak_idx]) / fall_t / 1000)
            else:
                fall.append(np.nan)

    return event_counts, event_times_all, {
        "Peak amplitude (mV)": amp,
        "Peak amplitude from onset (mV)": amp_onset,
        "Half duration (ms)": half_dur,
        "Rise slope (mV/ms)": rise,
        "Fall slope (mV/ms)": fall,
    }
