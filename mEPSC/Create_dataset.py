"""
Create CNN training dataset from manual mEPSC annotations.

This script:
- Reads manual event-interval annotations (JSON) produced by the annotation GUI.
- Loads the corresponding ABF recordings and applies the shared preprocessing filters.
- Extracts fixed-length positive segments around annotated events.
- Samples negative (non-event) segments from low-risk regions, avoiding annotated events.
- Saves NumPy arrays (X, y) and group IDs for leak-free train/val/test splitting.

Inputs
------
- results/<protocol>/<cnn_annotation_subdir>/events_*.json

Outputs
-------
- results/<protocol>/<cnn_dataset_subdir>/X.npy
- results/<protocol>/<cnn_dataset_subdir>/y.npy
- results/<protocol>/<cnn_dataset_subdir>/groups.npy
"""

import json
import yaml
from pathlib import Path

import numpy as np
import pyabf
from tqdm import tqdm
from scipy.signal import iirnotch, filtfilt, butter
from .config import CFG


base_path = Path(CFG["paths"]["base_path"])
protocol = CFG["protocol"]["mepsc"]
results_root = base_path / CFG["results"]["results_subdir"] / protocol

ANNOTATION_DIR = results_root / CFG["results"]["cnn_annotation_subdir"]
OUTPUT_DIR = results_root / CFG["results"]["cnn_dataset_subdir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Preprocessing filter settings (shared across scripts)
# ------------------------------------------------------------
filter_cfg = CFG["filter"]
NOTCH_MAX_HARMONIC = filter_cfg["notch_max_harmonic"]
NOTCH_Q = filter_cfg["notch_Q"]
LOWPASS_HZ = filter_cfg["lowpass_hz"]
LOWPASS_ORDER = filter_cfg["lowpass_order"]

# ------------------------------------------------------------
# CNN dataset parameters (shared with training/prediction)
# ------------------------------------------------------------
cnn_cfg = CFG["cnn"]
WINDOW_SEC = cnn_cfg["window_sec"]
CENTER_ON = cnn_cfg.get("center_on", "start")
NEG_POS_RATIO = cnn_cfg.get("neg_pos_ratio", 2.0)
EDGE_MARGIN_SEC = cnn_cfg.get("edge_margin_sec", 0.02)
VALID_START_SEC = cnn_cfg.get("valid_start_sec", 0.4)
VALID_END_SEC = cnn_cfg.get("valid_end_sec", 10.4)


# ============================================================
#                        PREPROCESSING
# ============================================================
def preprocess_signal(
    signal,
    fs,
    notch_harmonics=NOTCH_MAX_HARMONIC,
    notch_Q=NOTCH_Q,
    lowpass_hz=LOWPASS_HZ,
    lowpass_order=LOWPASS_ORDER,
):
    """
    Apply the shared preprocessing filters to a raw trace.

    Steps
    -----
    1) 50 Hz notch filter and its harmonics
    2) Butterworth low-pass filter

    Notes
    -----
    Keep this function consistent across GUI / training / prediction to avoid
    train-test distribution shifts.

    Parameters
    ----------
    signal : array-like
        Raw signal (one sweep).
    fs : float
        Sampling rate (Hz).

    Returns
    -------
    np.ndarray
        Filtered signal (float32).
    """
    signal = np.asarray(signal, dtype=float)

    # Notch filters at 50 Hz and harmonics (stop when reaching Nyquist)
    for k in range(1, notch_harmonics + 1):
        f0 = 50.0 * k
        if f0 >= fs / 2:
            break
        b, a = iirnotch(w0=f0, Q=notch_Q, fs=fs)
        signal = filtfilt(b, a, signal)

    # Low-pass filter
    nyq = fs / 2
    cutoff = lowpass_hz / nyq
    b, a = butter(lowpass_order, cutoff, btype="low")
    signal = filtfilt(b, a, signal)

    return signal.astype(np.float32)


# ============================================================
#                      INPUT / OUTPUT IO
# ============================================================
def load_annotation_files(annotation_dir: Path):
    """
    List annotation JSON files.

    Parameters
    ----------
    annotation_dir : Path
        Directory containing annotation files.

    Returns
    -------
    list[Path]
        Sorted list of paths matching "events_*.json".
    """
    return sorted(annotation_dir.glob("events_*.json"))


def load_abf_all_sweeps(abf_path: Path):
    """
    Load an ABF file and preprocess all sweeps.

    Parameters
    ----------
    abf_path : Path
        Path to the ABF file.

    Returns
    -------
    signals : list[np.ndarray]
        Preprocessed signal for each sweep.
    fs : float
        Sampling rate (Hz).
    sweep_lengths : np.ndarray
        Duration of each sweep (sec).
    """
    abf = pyabf.ABF(str(abf_path))
    fs = abf.dataRate
    n_sweeps = abf.sweepCount

    signals = []
    sweep_lengths = []
    for si in range(n_sweeps):
        abf.setSweep(si)
        raw = abf.sweepY.astype(np.float32)
        sig = preprocess_signal(raw, fs)
        signals.append(sig)
        sweep_lengths.append(abf.sweepLengthSec)

    signals = [np.asarray(sig) for sig in signals]
    sweep_lengths = np.asarray(sweep_lengths, dtype=float)
    return signals, fs, sweep_lengths


# ============================================================
#                    SEGMENT EXTRACTION
# ============================================================
def extract_event_segment(signal, fs, start_sec, end_sec, window_sec, center_on="start"):
    """
    Extract a fixed-length segment around an annotated event.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed sweep signal.
    fs : float
        Sampling rate (Hz).
    start_sec, end_sec : float
        Annotated event interval (sec).
    window_sec : float
        Segment length (sec).
    center_on : {"start", "center"}
        - "start": window starts at start_sec
        - "center": window is centered at (start_sec + end_sec) / 2

    Returns
    -------
    np.ndarray | None
        Segment of shape (window_sec * fs,) or None if out of bounds.
    """
    if center_on == "center":
        center = 0.5 * (start_sec + end_sec)
        t_start = center - window_sec / 2
    else:
        t_start = start_sec

    idx0 = int(round(t_start * fs))
    idx1 = idx0 + int(round(window_sec * fs))

    if idx0 < 0 or idx1 > len(signal):
        return None

    return signal[idx0:idx1]


def sample_negative_times_in_range(
    valid_start,
    valid_end,
    events,
    n_neg,
    window_sec,
    edge_margin_sec,
    total_sec,
    rng,
):
    """
    Sample negative (non-event) window start times for a single sweep.

    The sampler:
    - operates within [valid_start, valid_end]
    - avoids annotated event intervals expanded by `edge_margin_sec`
    - allows disjoint allowed regions (split OK)

    Parameters
    ----------
    valid_start, valid_end : float
        Allowed time range for sampling (sec).
    events : list[tuple[float, float]]
        Event intervals (start_sec, end_sec) within the same sweep.
    n_neg : int
        Number of negative windows to sample.
    window_sec : float
        Window length (sec).
    edge_margin_sec : float
        Safety margin added around each event interval (sec).
    total_sec : float
        Total sweep duration (sec).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    list[float]
        List of negative window start times (sec).
    """
    negatives = []
    if n_neg <= 0:
        return negatives

    base_start = max(valid_start, 0.0)
    base_end = min(valid_end, total_sec)
    if base_end - base_start <= window_sec + 2 * edge_margin_sec:
        return []

    # Forbidden intervals = event intervals expanded by a margin
    forbidden = [(s - edge_margin_sec, e + edge_margin_sec) for s, e in events]

    # Clip and merge forbidden intervals
    forbidden_merged = []
    for s, e in sorted(forbidden):
        s = max(s, base_start)
        e = min(e, base_end)
        if e <= s:
            continue
        if not forbidden_merged:
            forbidden_merged.append([s, e])
        else:
            ps, pe = forbidden_merged[-1]
            if s <= pe:
                forbidden_merged[-1][1] = max(pe, e)
            else:
                forbidden_merged.append([s, e])

    # Allowed intervals = [base_start, base_end] minus forbidden_merged
    allowed = []
    cur = base_start + edge_margin_sec
    for s, e in forbidden_merged:
        if s - cur > 2 * edge_margin_sec:
            allowed.append((cur, s))
        cur = max(cur, e)
    if base_end - edge_margin_sec - cur > 0:
        allowed.append((cur, base_end - edge_margin_sec))

    if not allowed:
        return []

    # Uniformly sample within allowed intervals (oversample attempts for robustness)
    for _ in range(n_neg * 5):
        if len(negatives) >= n_neg:
            break
        seg = allowed[rng.integers(len(allowed))]
        t = rng.uniform(seg[0], seg[1])
        if t + window_sec > base_end - edge_margin_sec:
            continue
        negatives.append(t)

    return negatives


# ============================================================
#                             MAIN
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_pos = []
    X_neg = []
    y_pos = []
    y_neg = []
    groups_pos = []
    groups_neg = []

    rng = np.random.default_rng(123)

    ann_files = load_annotation_files(ANNOTATION_DIR)
    print(f"Found {len(ann_files)} annotation files.")

    for file_idx, ann_path in enumerate(tqdm(ann_files)):
        # One annotation file corresponds to one ABF file → use file_idx as group ID
        group_id = int(file_idx)

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        abf_path = Path(ann["abf_path"])
        events_raw = ann.get("events", [])
        if not events_raw:
            continue

        # Annotation JSON schema:
        # {"sweep": int, "start_sec": float, "end_sec": float}
        events_by_sweep = {}
        for ev in events_raw:
            sw = int(ev["sweep"])
            s = float(ev["start_sec"])
            e = float(ev["end_sec"])
            events_by_sweep.setdefault(sw, []).append((s, e))

        signals, fs, sweep_lengths = load_abf_all_sweeps(abf_path)
        n_sweeps = len(signals)
        win_sec = WINDOW_SEC

        # Build samples sweep-by-sweep to preserve per-sweep validity bounds
        for sw, events in events_by_sweep.items():
            if sw < 0 or sw >= n_sweeps:
                continue
            signal = signals[sw]
            total_sec = float(sweep_lengths[sw])

            use_start = max(VALID_START_SEC, 0.0)
            use_end = min(VALID_END_SEC, total_sec)
            if use_end - use_start <= win_sec:
                continue

            # Keep only events fully inside the valid range
            events_in_range = [(s, e) for (s, e) in events if (s >= use_start) and (e <= use_end)]
            if not events_in_range:
                continue

            # Positive samples
            for s, e in events_in_range:
                seg = extract_event_segment(signal, fs, s, e, win_sec, center_on=CENTER_ON)
                if seg is None:
                    continue
                X_pos.append(seg)
                y_pos.append(1)
                groups_pos.append(group_id)

            # Negative samples (count proportional to positives in the same sweep)
            n_pos = len(events_in_range)
            n_neg = int(n_pos * NEG_POS_RATIO)
            neg_times = sample_negative_times_in_range(
                valid_start=use_start,
                valid_end=use_end,
                events=events_in_range,
                n_neg=n_neg,
                window_sec=win_sec,
                edge_margin_sec=EDGE_MARGIN_SEC,
                total_sec=total_sec,
                rng=rng,
            )

            for t in neg_times:
                idx0 = int(round(t * fs))
                idx1 = idx0 + int(round(win_sec * fs))
                if idx1 > len(signal):
                    continue
                seg = signal[idx0:idx1]
                if len(seg) != int(round(win_sec * fs)):
                    continue
                X_neg.append(seg)
                y_neg.append(0)
                groups_neg.append(group_id)

    if not X_pos:
        print("No positive segments found. Check your annotations or VALID_*_SEC.")
        return

    X = np.concatenate([np.stack(X_pos), np.stack(X_neg)], axis=0)
    y = np.array(y_pos + y_neg, dtype=np.int64)
    groups = np.array(groups_pos + groups_neg, dtype=np.int64)

    # Shuffle dataset (X, y, groups stay aligned)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]
    groups = groups[idx]

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)
    np.save(OUTPUT_DIR / "groups.npy", groups)

    print(f"Saved dataset: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Saved groups: shape = {groups.shape}, #unique groups = {len(np.unique(groups))}")


if __name__ == "__main__":
    main()
