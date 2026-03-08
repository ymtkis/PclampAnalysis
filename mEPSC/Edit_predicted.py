"""Edit (curate) CNN-predicted mEPSC detections with a GUI.

This script loads CNN detection JSON files, displays the corresponding ABF sweeps,
and provides an interactive viewer to inspect, adjust, and save edited intervals/peaks.

Main responsibilities
---------------------
- Load project configuration (config.yaml)
- Load and preprocess ABF sweeps (notch + low-pass filtering)
- Load CNN detection JSON (intervals + refined peaks)
- Launch an interactive GUI to edit detections
- Optionally build a command to recompute peaks after editing

Inputs
------
- CNN detection JSON files (e.g., *_cnn.json / *_cnn_edited.json)
- ABF files referenced by the JSON
- config.yaml for paths and filter parameters

Outputs
-------
- Edited JSON saved by the GUI (and optional recomputed peaks via external script)
"""

import json
import yaml
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkfont

import numpy as np
import pyabf
import matplotlib.pyplot as plt
plt.rcParams["keymap.save"] = []

from scipy.signal import iirnotch, filtfilt, butter
from event_gui import PredictionViewer
from config import CFG

base_path = Path(CFG["paths"]["base_path"])
protocol = CFG["protocol"]["mepsc"]
results_root = base_path / CFG["results"]["results_subdir"] / protocol
DETECTION_DIR = results_root / CFG["results"]["cnn_detection_subdir"]

WINDOW_SEC = CFG.get("annotation", {}).get("window_sec", 0.5)
Y_RANGE = CFG.get("annotation", {}).get("y_range", 50)

filter_cfg = CFG["filter"]
NOTCH_MAX_HARMONIC = filter_cfg["notch_max_harmonic"]
NOTCH_Q = filter_cfg["notch_Q"]
LOWPASS_HZ = filter_cfg["lowpass_hz"]
LOWPASS_ORDER = filter_cfg["lowpass_order"]

cnn_cfg = CFG["cnn"]
DEFAULT_VALID_START = cnn_cfg.get("valid_start_sec", 0.4)
DEFAULT_VALID_END = cnn_cfg.get("valid_end_sec", 10.4)


def preprocess_signal(signal, fs):
    """
    Preprocess a raw sweep trace for display and editing.

    Applies:
    1) 50 Hz notch filter and harmonics (up to configured max harmonic)
    2) Butterworth low-pass filter (configured cutoff and order)

    Parameters
    ----------
    signal : np.ndarray
        Raw sweep trace.
    fs : float
        Sampling rate (Hz).

    Returns
    -------
    np.ndarray
        Filtered signal (float32).
    """
    sig = signal.astype(float)

    for k in range(1, NOTCH_MAX_HARMONIC + 1):
        f0 = 50.0 * k
        if f0 >= fs / 2:
            break
        b, a = iirnotch(w0=f0, Q=NOTCH_Q, fs=fs)
        sig = filtfilt(b, a, sig)

    nyq = fs / 2.0
    cutoff = LOWPASS_HZ / nyq
    b, a = butter(LOWPASS_ORDER, cutoff, btype="low")
    sig = filtfilt(b, a, sig)

    return sig.astype(np.float32)


def load_abf_filtered(abf_path: Path):
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

    signals = []
    sweep_lengths = []

    for si in range(abf.sweepCount):
        abf.setSweep(si)
        sig = preprocess_signal(abf.sweepY, fs)
        signals.append(sig)
        sweep_lengths.append(abf.sweepLengthSec)

    return signals, fs, np.asarray(sweep_lengths, float)


def load_detection_json(json_path: Path):
    """
    Load a CNN detection JSON and parse intervals/peaks per sweep.

    Parameters
    ----------
    json_path : Path
        Path to a detection JSON file.

    Returns
    -------
    abf_path : Path
        ABF path referenced by the JSON.
    data : dict
        Full JSON object.
    intervals_by_sweep : dict[int, list[tuple[float, float, float]]]
        Mapping sweep -> [(start_sec, end_sec, max_proba), ...].
    peaks_by_sweep : dict[int, list[dict]]
        Mapping sweep -> list of peak dictionaries.
    valid_start : float
        Default analysis start time (sec).
    valid_end : float
        Default analysis end time (sec).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    abf_path = Path(data["abf_path"])

    intervals_by_sweep = {}
    for ev in data.get("cnn_intervals", []):
        sw = int(ev["sweep"])
        intervals_by_sweep.setdefault(sw, []).append(
            (float(ev["start_sec"]), float(ev["end_sec"]), float(ev.get("max_proba", 0.0)))
        )

    peaks_by_sweep = {}
    for ev in data.get("refined_events", []):
        sw = int(ev["sweep"])
        peaks_by_sweep.setdefault(sw, []).append(
            {
                "peak_sec": float(ev["peak_sec"]),
                "amp_pA": float(ev.get("amp_pA", np.nan)),
                "snr": float(ev.get("snr", np.nan)),
                "parent_max_proba": float(ev.get("parent_max_proba", np.nan)),
            }
        )

    det_params = data.get("detection_params", {})
    valid_start = float(det_params.get("valid_start_sec", DEFAULT_VALID_START))
    valid_end = float(det_params.get("valid_end_sec", DEFAULT_VALID_END))

    return abf_path, data, intervals_by_sweep, peaks_by_sweep, valid_start, valid_end


def init_tk_font(root, family="Arial", size=12, weight="normal"):
    """
    Set a consistent default Tkinter font for dialogs.

    Parameters
    ----------
    root : tk.Tk
        Tk root object.
    family : str
        Font family.
    size : int
        Font size.
    weight : str
        Font weight.
    """
    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(family=family, size=size, weight=weight)
    root.option_add("*Font", default_font)
    

def choose_json_file(initial_dir: Path) -> Path | None:
    """
    Open a file dialog to select a CNN detection JSON.

    Parameters
    ----------
    initial_dir : Path
        Initial directory shown in the dialog.

    Returns
    -------
    Path or None
        Selected JSON path, or None if cancelled.
    """
    root = tk.Tk()
    init_tk_font(root, family="Arial", size=12)
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select CNN detection JSON file",
        initialdir=str(initial_dir),
        filetypes=[
            ("CNN detection JSON", "*_cnn*.json"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    return Path(path) if path else None


def build_recompute_cmd(original_json, edited_json):
    """
    Build a command to recompute refined peaks after editing.

    Notes
    -----
    The GUI can call this command externally to regenerate peaks based on
    edited intervals and a shared config.yaml.
    """
    return [
        sys.executable,
        str(Path(__file__).with_name("recompute_edited_peaks.py")),
        "--auto", str(original_json),
        "--edited", str(edited_json),
        "--tol", "1e-4",
    ]


def launch_viewer(json_path: Path):
    """
    Launch the interactive prediction viewer for a given detection JSON.

    Parameters
    ----------
    json_path : Path
        Path to a CNN detection JSON file.
    """
    abf_path, json_data, intervals_by_sweep, peaks_by_sweep, vstart, vend = load_detection_json(json_path)
    signals, fs, sweep_lengths = load_abf_filtered(abf_path)

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.subplots_adjust(bottom=0.15)

    PredictionViewer(
        fig=fig,
        ax=ax,
        signals=signals,
        fs=fs,
        sweep_lengths=sweep_lengths,
        intervals_by_sweep=intervals_by_sweep,
        peaks_by_sweep=peaks_by_sweep,
        window_sec=WINDOW_SEC,
        y_range=Y_RANGE,
        json_data=json_data,
        json_path=json_path,
        default_valid_start=vstart,
        default_valid_end=vend,
        enable_save=True,
        enable_peak_edit=True,
        recompute_cmd_builder=build_recompute_cmd,
        on_close_callback=lambda: None,
    )

    plt.show()


def main():
    """
    Entry point: repeatedly select a JSON file and open the viewer until cancelled.
    """
    while True:
        json_path = choose_json_file(DETECTION_DIR)
        if json_path is None:
            break
        launch_viewer(json_path)


if __name__ == "__main__":
    main()
