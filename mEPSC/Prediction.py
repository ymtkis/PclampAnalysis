"""
Prediction stage for the mEPSC CNN pipeline.

This script loads a trained 1D CNN model and applies it to ABF recordings to:
1) Compute window-level event probabilities across a valid time range.
2) Convert high-probability windows into merged event intervals.
3) Refine peaks within each interval using baseline + noise (SD) estimation.
4) Save interval events and refined events to JSON.

Inputs
------
- ABF files selected via GUI dialog, or discovered automatically with --auto.
- Trained model checkpoint: results/<protocol>/<cnn_model_subdir>/mEPSC_cnn.pt
- config.yaml (paths, filters, CNN parameters, post-processing parameters)

Outputs
-------
- results/<protocol>/<cnn_detection_subdir>/<condition>/<abf_stem>_cnn.json
  containing:
  - cnn_intervals: merged high-probability intervals
  - refined_events: peak-level events after SNR-based refinement

Notes
-----
Comment and docstring formatting is standardized intentionally; runtime behavior
and numeric logic are unchanged.
"""

import json
import yaml
from pathlib import Path
import sys

import numpy as np
import pyabf
from scipy.signal import iirnotch, filtfilt, butter, find_peaks

import torch
from models.cnn_simple import Simple1DCNN

from tqdm import tqdm
from config import CFG


# ============================================================
# PREPROCESSING                                              
# ============================================================
def preprocess_signal(signal, fs, notch_harmonics, notch_Q, lowpass_hz, lowpass_order):
    """
    Preprocess a raw sweep trace before CNN inference.
    
    Steps
    -----
    1) Notch-filter 50 Hz and harmonics (up to notch_harmonics).
    2) Low-pass filter to remove high-frequency noise.
    
    Parameters
    ----------
    signal : array-like
        Raw trace values.
    fs : float
        Sampling rate (Hz).
    notch_harmonics : int
        Number of 50 Hz harmonics to apply notch filtering to.
    notch_Q : float
        Quality factor for notch filters.
    lowpass_hz : float
        Low-pass cutoff (Hz).
    lowpass_order : int
        Butterworth low-pass order.
    
    Returns
    -------
    np.ndarray
        Filtered trace (float32).
    """
    sig = np.asarray(signal, float)

    for k in range(1, notch_harmonics + 1):
        f0 = 50.0 * k
        if f0 >= fs / 2:
            break
        b, a = iirnotch(f0, notch_Q, fs=fs)
        sig = filtfilt(b, a, sig)

    b, a = butter(lowpass_order, lowpass_hz / (fs / 2), btype="low")
    sig = filtfilt(b, a, sig)

    return sig.astype(np.float32)


def highpass(sig, fs, cutoff, order):
    """
    High-pass filter used for peak refinement.
    
    Parameters
    ----------
    sig : array-like
        Input signal.
    fs : float
        Sampling rate (Hz).
    cutoff : float
        High-pass cutoff (Hz).
    order : int
        Butterworth filter order.
    
    Returns
    -------
    np.ndarray
        High-pass filtered signal.
    """
    b, a = butter(order, cutoff / (fs / 2), btype="high")
    return filtfilt(b, a, sig)


# ============================================================
# CNN INTERVAL DETECTION                                     
# ============================================================
def detect_events_in_trace(
    signal,
    fs,
    total_sec,
    model,
    device,
    window_sec,
    step_sec,
    proba_thresh,
    min_gap_between_events,
    valid_start,
    valid_end,
):
    """
    Convert window-level CNN probabilities into merged event intervals.

    Sliding windows with p(event) >= proba_thresh are converted to intervals, then
    neighboring intervals are merged when the gap is smaller than min_gap_between_events.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed sweep signal.
    fs : float
        Sampling rate (Hz).
    total_sec : float
        Sweep duration (sec).
    model : torch.nn.Module
        Trained CNN model.
    device : torch.device
        Inference device.
    window_sec : float
        Sliding window length (sec).
    step_sec : float
        Sliding window stride (sec).
    proba_thresh : float
        Probability threshold for event window selection.
    min_gap_between_events : float
        Merge gap threshold (sec).
    valid_start : float
        Start of analysis range (sec).
    valid_end : float
        End of analysis range (sec).

    Returns
    -------
    list[tuple[float, float, float]]
        (start_sec, end_sec, max_proba) for each merged interval.
    """
    win_len = int(window_sec * fs)
    step_len = int(step_sec * fs)

    use_start = max(valid_start, 0.0)
    use_end = min(valid_end, total_sec)
    i = int(use_start * fs)
    max_i = int((use_end - window_sec) * fs)

    starts, probs = [], []

    while i + win_len <= len(signal) and i <= max_i:
        seg = signal[i:i + win_len]
        seg = (seg - seg.mean()) / (seg.std() + 1e-6)

        x = torch.from_numpy(seg[None, None, :]).to(device)
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)[0, 1].item()

        if p >= proba_thresh:
            starts.append(i / fs)
            probs.append(p)

        i += step_len

    if not starts:
        return []

    events = []
    s0 = starts[0]
    e0 = s0 + window_sec
    p0 = probs[0]

    for t, p in zip(starts[1:], probs[1:]):
        if t <= e0 + min_gap_between_events:
            e0 = max(e0, t + window_sec)
            p0 = max(p0, p)
        else:
            events.append((s0, e0, p0))
            s0, e0, p0 = t, t + window_sec, p

    events.append((s0, e0, p0))
    return events


def compute_proba_trace(
    signal, fs, total_sec, model, device,
    window_sec, step_sec, valid_start, valid_end
):
    """
    Compute window-level CNN event probability trace without thresholding.
    
    A sliding window is applied over [valid_start, valid_end]. For each window, the
    CNN returns p(event). The reported time corresponds to the window center.
    
    Returns
    -------
    (np.ndarray, np.ndarray)
        times (sec) and probabilities, aligned to the window centers.
    """
    win_len = int(window_sec * fs)
    step_len = int(step_sec * fs)

    use_start = max(valid_start, 0.0)
    use_end = min(valid_end, total_sec)

    i = int(use_start * fs)
    max_i = int((use_end - window_sec) * fs)

    times = []
    probs = []

    while i + win_len <= len(signal) and i <= max_i:
        seg = signal[i:i + win_len]
        seg = (seg - seg.mean()) / (seg.std() + 1e-6)

        x = torch.from_numpy(seg[None, None, :]).to(device)
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)[0, 1].item()

        center_t = (i + win_len / 2.0) / fs
        times.append(center_t)
        probs.append(p)

        i += step_len

    if not times:
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.array(times, dtype=float), np.array(probs, dtype=float)


# ============================================================
# PEAK + SNR REFINEMENT                                      
# ============================================================
import numpy as np
from scipy.signal import find_peaks


def compute_baseline_around_peak(
    sig_hp, fs, t_peak,
    proba_times, proba_values,
    window_sec, step_sec,
    total_baseline_sec,
    proba_thresh,
    valid_start, valid_end,
    return_window=False,
):
    """
    Estimate baseline near a candidate peak using low-probability regions.
    
    Baseline is computed as the mean of samples collected from windows whose center
    time satisfies p(event) < proba_thresh and lies within [valid_start, valid_end].
    Windows may be disjoint; windows nearest to t_peak are prioritized until roughly
    total_baseline_sec coverage is reached (approximated via window count).
    
    Parameters
    ----------
    sig_hp : np.ndarray
        High-pass filtered signal used for peak refinement.
    fs : float
        Sampling rate (Hz).
    t_peak : float
        Candidate peak time (sec).
    proba_times, proba_values : np.ndarray
        Window-center times (sec) and corresponding p(event).
    window_sec, step_sec : float
        Window length and stride (sec).
    total_baseline_sec : float
        Target baseline coverage (sec).
    proba_thresh : float
        Threshold for considering a window 'non-event'.
    valid_start, valid_end : float
        Allowed time range (sec).
    return_window : bool
        If True, also return the list of sample-index bounds used for baseline/SD.
    
    Returns
    -------
    float | (float, list[tuple[int,int]]) | None
        Baseline mean (and bounds if return_window=True).
    """
    # Candidate centers where event probability is low and within valid region
    mask = (
        (proba_values < proba_thresh) &
        (proba_times >= valid_start) &
        (proba_times <= valid_end)
    )
    if not np.any(mask):
        return (None, []) if return_window else None

    t_cand = proba_times[mask]
    # nearest centers to the peak time first
    order = np.argsort(np.abs(t_cand - t_peak))

    # How many windows to collect (approximate coverage)
    # Use step_sec to match your original intent ("approximated by window count").
    n_needed = max(1, int(np.ceil(float(total_baseline_sec) / float(step_sec))))

    bounds = []
    segs = []
    used = 0

    half = float(window_sec) / 2.0
    n = len(sig_hp)

    for i in order:
        if used >= n_needed:
            break

        center = float(t_cand[i])
        t0 = center - half
        t1 = center + half

        # Keep window inside valid region (important)
        t0 = max(float(valid_start), t0)
        t1 = min(float(valid_end), t1)
        if t1 <= t0:
            continue

        j0 = int(np.floor(t0 * fs))
        j1 = int(np.ceil(t1 * fs))
        j0 = max(0, j0)
        j1 = min(n, j1)
        if j1 <= j0 + 2:
            continue

        segs.append(sig_hp[j0:j1])
        bounds.append((j0, j1))
        used += 1

    if not segs:
        return (None, []) if return_window else None

    x = np.concatenate(segs)
    if x.size < 3:
        return (None, []) if return_window else None

    bl = float(x.mean())
    return (bl, bounds) if return_window else bl


def refine_peaks(
    sig, fs, intervals,
    min_gap_sec, min_amp_pA,
    snr_k,
    sd_win_sec, hp_hz, hp_order,
    p_t, p_v,
    win_sec, step_sec,
    baseline_sec=1.0,
    proba_thresh=0.90,  
    valid_start=0.4,
    valid_end=10.4,
    # Suppress small peaks on the decay of a large event
    big_snr=6.0,
    shadow_ms=30.0,
    keep_snr_ratio=0.80,
    keep_snr_shadow=8.0,
    rise_win_ms=2.0,
    rise_eps_ms=0.2,
    keep_rise_abs=2.5,
    keep_rise_ratio=0.5,
):
    """
    Refine peak candidates within detected intervals using baseline + SNR rules.
    
    Within each detected interval, negative peaks are found on a high-pass filtered
    signal. Candidates are filtered by minimum amplitude and SNR, where baseline and
    local noise (SD) are computed from the same low-probability baseline windows.
    
    Notes
    -----
    - Windows used for baseline/SD may be disjoint (split OK).
    - Additional suppression is applied to small peaks occurring shortly after a
      large event (shadow window).
    """
    hp = highpass(sig, fs, hp_hz, hp_order)
    min_dist = int(min_gap_sec * fs)

    shadow_sec = float(shadow_ms) / 1000.0
    rise_win = max(1, int(round((float(rise_win_ms) / 1000.0) * fs)))
    rise_eps = max(0, int(round((float(rise_eps_ms) / 1000.0) * fs)))

    def _baseline_and_bounds(t_pk):
        bl, bounds = compute_baseline_around_peak(
            hp, fs, t_pk,
            p_t, p_v,
            win_sec, step_sec,
            baseline_sec,
            proba_thresh,
            valid_start,
            valid_end,
            return_window=True,
        )
        if bl is None or not bounds:
            return None, []
        return bl, bounds

    def _local_sd(bl, bounds):
        if bl is None or not bounds:
            return None

        segs = []
        for j0, j1 in bounds:
            if j1 > j0 + 2:
                segs.append(hp[j0:j1])
        if not segs:
            return None

        w = np.concatenate(segs)
        if w.size < 5:
            return None

        aw = bl - w
        m = aw < float(min_amp_pA)
        ww = w[m] if np.any(m) else w
        if ww.size < 5:
            return None
        sd = float(ww.std())
        return sd if sd > 0 else None

    def _rise_amp(idx):
        j1 = max(0, idx - rise_eps)
        j0 = max(0, j1 - rise_win)
        if j1 <= j0 + 2:
            return 0.0
        return float(np.max(hp[j0:j1]) - hp[idx])

    def _keep_after_big(prev, cur):
        dt = cur["t"] - prev["t"]
        if prev["snr"] < big_snr or dt > shadow_sec:
            return True
        if cur["snr"] >= keep_snr_ratio * prev["snr"]:
            return True
        if (cur["rise"] >= keep_rise_abs) and (cur["rise"] >= keep_rise_ratio * cur["amp"]):
            return True
        return False

    out = []
    n_sig = len(hp)

    for itv in intervals:
        s, e = float(itv["start_sec"]), float(itv["end_sec"])
        i0, i1 = int(s * fs), int(e * fs)
        i0 = max(0, min(n_sig, i0))
        i1 = max(0, min(n_sig, i1))
        if i1 <= i0 + 3:
            continue

        seg = hp[i0:i1]
        pks, _ = find_peaks(-seg, distance=min_dist)
        if len(pks) == 0:
            continue

        cands = []
        for pk in pks:
            idx = i0 + int(pk)
            t_pk = idx / fs

            # baseline + same-window SD
            bl, bounds = _baseline_and_bounds(t_pk)
            if bl is None:
                continue

            amp = float(bl - hp[idx])
            if amp < min_amp_pA:
                continue

            sd = _local_sd(bl, bounds)
            if sd is None:
                continue

            if amp <= snr_k * sd:
                continue

            cands.append({
                "sweep": itv["sweep"],
                "t": t_pk,
                "amp": amp,
                "snr": amp / sd,
                "rise": _rise_amp(idx),
                "pmax": float(itv.get("max_proba", np.nan)),
            })

        if not cands:
            continue

        # sort by time, apply "big event shadow" suppression
        cands.sort(key=lambda d: d["t"])
        kept = []
        for c in cands:
            if not kept or _keep_after_big(kept[-1], c):
                kept.append(c)

        for k in kept:
            out.append({
                "sweep": k["sweep"],
                "peak_sec": k["t"],
                "amp_pA": k["amp"],
                "snr": k["snr"],
                "parent_max_proba": k["pmax"],
            })

    out.sort(key=lambda d: (d["sweep"], d["peak_sec"]))
    return out



# ============================================================
# PROCESS ONE ABF                                            
# ============================================================
def process_one_abf(abf_path, condition, CFG):
    """
    Run CNN-based detection and refinement for a single ABF file.
    
    This function loads the trained model, iterates through all sweeps, computes a
    probability trace, detects intervals, refines peaks, and writes a JSON output.
    """
    base = Path(CFG["paths"]["base_path"])
    protocol = CFG["protocol"]["mepsc"]
    results_root = base / CFG["results"]["results_subdir"] / protocol

    out_dir = results_root / CFG["results"]["cnn_detection_subdir"] / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"{abf_path.stem}_cnn.json"
    if out_json.exists():
        print(f"[SKIP] {out_json.name}")
        return

    cnn = CFG["cnn"]
    post = CFG["cnn_postprocess"]
    filt = CFG["filter"]

    abf = pyabf.ABF(str(abf_path))
    fs = abf.dataRate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Simple1DCNN(int(cnn["window_sec"] * fs))
    ckpt_path = results_root / CFG["results"]["cnn_model_subdir"] / "mEPSC_cnn.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        saved_input_len = ckpt.get("input_length", None)
        best_threshold = ckpt.get("best_threshold", None)
    else:
        state = ckpt
        saved_input_len = None
        best_threshold = None

    current_input_len = int(cnn["window_sec"] * fs)
    if saved_input_len is not None and int(saved_input_len) != current_input_len:
        raise ValueError(
            f"input_length mismatch: checkpoint={saved_input_len}, current={current_input_len}. "
            f"window_sec or fs differs from training."
        )

    model.load_state_dict(state)
    model.to(device).eval()

    if best_threshold is not None:
        print(f"Model best_threshold (window-level F1) = {best_threshold:.4f}")

    proba_thresh = float(cnn.get("proba_thresh", 0.99))
    print(f"Using proba_thresh for detection = {proba_thresh:.4f}")

    interval_events = []
    refined_events = []

    for sw in tqdm(range(abf.sweepCount), desc=f"{abf_path.name}", leave=False):
        abf.setSweep(sw)
        sig = preprocess_signal(
            abf.sweepY, fs,
            filt["notch_max_harmonic"], filt["notch_Q"],
            filt["lowpass_hz"], filt["lowpass_order"],
        )

        proba_times, proba_values = compute_proba_trace(
            sig, fs, abf.sweepLengthSec, model, device,
            cnn["window_sec"], cnn["step_sec"],
            cnn["valid_start_sec"], cnn["valid_end_sec"],
        )

        intervals = detect_events_in_trace(
            sig, fs, abf.sweepLengthSec, model, device,
            cnn["window_sec"], cnn["step_sec"], proba_thresh,
            post["min_gap_between_events"],
            cnn["valid_start_sec"], cnn["valid_end_sec"],
        )

        itv_dicts = []
        for s, e, p in intervals:
            d = {"sweep": sw, "start_sec": s, "end_sec": e, "max_proba": p}
            interval_events.append(d)
            itv_dicts.append(d)

        refined_events.extend(
            refine_peaks(
                sig, fs, itv_dicts,
                post["min_gap_between_events"],
                post["min_peak_ampl_pA"],
                post["snr_factor"],
                post["local_sd_win_sec"],
                post["highpass_hz"],
                post.get("highpass_order", 2),
                proba_times, proba_values,
                cnn["window_sec"], cnn["step_sec"],
                baseline_sec=1.0,
                proba_thresh=0.90,
                valid_start=cnn["valid_start_sec"],
                valid_end=cnn["valid_end_sec"],
            )
        )

    with open(out_json, "w") as f:
        json.dump({
            "abf_path": str(abf_path),
            "condition": condition,
            "cnn_intervals": interval_events,
            "refined_events": refined_events,
        }, f, indent=2)

    print(f"[OK] {out_json.name}")


# ============================================================
# FILE SELECTION HELPERS                                     
# ============================================================
def select_abf_files_by_dialog(CFG):
    """
    Open a file dialog to select one or more ABF files.
    
    Returns
    -------
    list[pathlib.Path]
        Selected ABF file paths (empty if none selected or tkinter unavailable).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        print(f"tkinter is not available: {e}")
        return []

    base = Path(CFG["paths"]["base_path"])
    proto = CFG["protocol"]["mepsc"]
    data_root = base / CFG["paths"]["data_root"] / proto
    init_dir = str(data_root if data_root.exists() else base)

    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Select ABF files for CNN detection",
        initialdir=init_dir,
        filetypes=[("ABF files", "*.abf"), ("All files", "*.*")],
    )
    root.destroy()

    return [Path(p) for p in file_paths]


def infer_condition_from_path(abf_path: Path, CFG):
    """
    Infer experimental condition from an ABF filename or parent directory.
    
    Returns
    -------
    str
        Condition name if matched against CFG['conditions'], otherwise 'unknown'.
    """
    name_tokens = abf_path.stem.split("_")
    conditions = CFG.get("conditions", [])

    for cond in conditions:
        if cond in name_tokens:
            return cond

    parent_name = abf_path.parent.name
    if parent_name in conditions:
        return parent_name

    return "unknown"


def process_files_from_config(CFG):
    """
    Batch-process ABF files discovered from config.yaml.
    
    When run with --auto, this scans condition subdirectories under the data root
    and processes matching ABF files.
    """
    base = Path(CFG["paths"]["base_path"])
    data_root = base / CFG["paths"]["data_root"] / CFG["protocol"]["mepsc"]
    prefix = CFG["paths"]["prefix"]

    for condition in CFG["conditions"]:
        abf_dir = data_root / condition
        if not abf_dir.exists():
            continue

        files = sorted(abf_dir.glob(f"{prefix}_*_{condition}_*_mEPSC.abf"))
        print(f"\n=== {condition}: {len(files)} files ===")
        for f in files:
            try:
                process_one_abf(f, condition, CFG)
            except Exception as e:
                print(f"[ERROR] {f.name}: {e}")


# ============================================================
# MAIN                                                       
# ============================================================
def main():
    """
    Entry point for interactive (GUI) or automatic (--auto) processing.
    """

    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        process_files_from_config(CFG)
        return

    files = select_abf_files_by_dialog(CFG)
    if not files:
        print("No files selected.")
        return

    print(f"Selected {len(files)} files.")
    for abf_path in files:
        condition = infer_condition_from_path(abf_path, CFG)
        print(f"\n[{condition}] {abf_path.name}")
        try:
            process_one_abf(abf_path, condition, CFG)
        except Exception as e:
            print(f"[ERROR] {abf_path.name}: {e}")


if __name__ == "__main__":
    main()
