"""
Event annotation GUI for mEPSC analysis.

This script provides an interactive GUI to manually annotate mEPSC event intervals
from ABF electrophysiology recordings.

Main responsibilities
---------------------
- Load ABF files and preprocess signals (notch + low-pass filtering)
- Display filtered and reference traces for each sweep
- Allow manual marking/editing of event intervals
- Save annotations as JSON files for downstream CNN training

Inputs
------
- ABF files selected via file dialog
- config.yaml for paths and filter parameters

Outputs
-------
- JSON files containing annotated event intervals per ABF file
"""

import json
import yaml
from pathlib import Path
import numpy as np
import pyabf
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt, butter
import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkfont

from event_gui import EventAnnotator
from .config import CFG

base_path = Path(CFG["paths"]["base_path"])
protocol = CFG["protocol"]["mepsc"]
results_root = base_path / CFG["results"]["results_subdir"] / protocol


# ============================================================
#              FILE DIALOG FOR ABF SELECTION
# ============================================================

def init_tk_font(root, family="Arial", size=12, weight="normal"):
    """
    Initialize default Tkinter font settings for consistent UI appearance.
    """
    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(family=family, size=size, weight=weight)
    root.option_add("*Font", default_font)


def choose_abf_file(initial_dir: Path) -> Path:
    """
    Open a file dialog to select an ABF file.

    Parameters
    ----------
    initial_dir : Path
        Initial directory shown in the dialog.

    Returns
    -------
    Path
        Selected ABF file path.

    Raises
    ------
    SystemExit
        If no file is selected.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select ABF file",
        initialdir=str(initial_dir),
        filetypes=[("ABF files", "*.abf"), ("All files", "*.*")],
    )
    root.destroy()

    if not path:
        raise SystemExit("No ABF file selected. Exiting.")

    return Path(path)


# ============================================================
#                   LOAD ABF AND PREPARE DATA
# ============================================================

abf_path = choose_abf_file(base_path)
print(f"Using ABF file: {abf_path}")

abf = pyabf.ABF(str(abf_path))
fs = abf.dataRate
n_sweeps = abf.sweepCount

WINDOW_SEC = CFG.get("annotation", {}).get("window_sec", 0.5)
Y_RANGE = CFG.get("annotation", {}).get("y_range", 50)

ANNOTATION_DIR = results_root / CFG["results"]["cnn_annotation_subdir"]
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

output_json = ANNOTATION_DIR / f"events_{abf_path.stem}.json"


# ============================================================
#                   FILTER PARAMETERS
# ============================================================

NOTCH_MAX_HARMONIC = CFG.get("filter", {}).get("notch_max_harmonic", 4)
NOTCH_Q = CFG.get("filter", {}).get("notch_Q", 30)
LOWPASS_HZ = CFG.get("filter", {}).get("lowpass_hz", 500)
LOWPASS_ORDER = CFG.get("filter", {}).get("lowpass_order", 4)


# ============================================================
#                        SIGNAL FILTERING
# ============================================================

def preprocess_signal(signal, fs,
                      notch_harmonics=NOTCH_MAX_HARMONIC,
                      notch_Q=NOTCH_Q,
                      lowpass_hz=LOWPASS_HZ,
                      lowpass_order=LOWPASS_ORDER):
    """
    Apply notch (50 Hz harmonics) and low-pass filtering to a raw signal.

    Parameters
    ----------
    signal : np.ndarray
        Raw electrophysiological signal.
    fs : float
        Sampling rate (Hz).

    Returns
    -------
    np.ndarray
        Filtered signal (float32).
    """
    filtered = signal.astype(float)

    # Remove power-line noise and harmonics
    for k in range(1, notch_harmonics + 1):
        f0 = 50.0 * k
        if f0 >= fs / 2:
            break
        b, a = iirnotch(w0=f0, Q=notch_Q, fs=fs)
        filtered = filtfilt(b, a, filtered)

    # Low-pass filtering for mEPSC analysis
    nyq = fs / 2
    cutoff = lowpass_hz / nyq
    b, a = butter(lowpass_order, cutoff, btype="low")
    filtered = filtfilt(b, a, filtered)

    return filtered.astype(np.float32)


def lowpass_only(signal, fs,
                 lowpass_hz=LOWPASS_HZ,
                 lowpass_order=LOWPASS_ORDER):
    """
    Apply low-pass filtering only (no notch filtering).

    Used for reference trace display.
    """
    x = signal.astype(float)
    nyq = fs / 2
    cutoff = lowpass_hz / nyq
    b, a = butter(lowpass_order, cutoff, btype="low")
    x = filtfilt(b, a, x)
    return x.astype(np.float32)


# ============================================================
#              FILTER ALL SWEEPS
# ============================================================

signals_notch = []
signals_lp = []
sweep_lengths = []

for si in range(n_sweeps):
    abf.setSweep(si)
    raw = abf.sweepY.astype(np.float32)

    # Main trace: notch + low-pass
    sig_notch = preprocess_signal(raw, fs)
    # Reference trace: low-pass only
    sig_lp = lowpass_only(raw, fs)

    signals_notch.append(sig_notch)
    signals_lp.append(sig_lp)
    sweep_lengths.append(abf.sweepLengthSec)

signals_notch = [np.asarray(sig) for sig in signals_notch]
signals_lp = [np.asarray(sig) for sig in signals_lp]
sweep_lengths = np.asarray(sweep_lengths, dtype=float)


# ============================================================
#           LOAD EXISTING JSON (RESUME SUPPORT)
# ============================================================

existing_events_per_sweep = [[] for _ in range(n_sweeps)]

if output_json.exists():
    try:
        with open(output_json, "r", encoding="utf-8") as f:
            prev = json.load(f)

        prev_abf = Path(prev.get("abf_path", ""))

        # Resume only if ABF path matches
        if prev_abf == abf_path:
            for ev in prev.get("events", []):
                sw = int(ev["sweep"])
                s = float(ev["start_sec"])
                e = float(ev["end_sec"])
                if 0 <= sw < n_sweeps:
                    existing_events_per_sweep[sw].append((s, e))

            for sw in range(n_sweeps):
                existing_events_per_sweep[sw].sort(key=lambda x: x[0])

            total_prev = sum(len(v) for v in existing_events_per_sweep)
            print(f"Loaded {total_prev} existing events from {output_json}")
        else:
            print("Existing annotation JSON does not match ABF file. Ignored.")

    except Exception as e:
        print(f"Failed to load existing annotation JSON: {e}")


# ============================================================
#                  RUN GUI AND SAVE JSON
# ============================================================

fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, figsize=(15, 10), sharex=True
)
plt.subplots_adjust(bottom=0.13, hspace=0.05)

annotator = EventAnnotator(
    fig=fig,
    ax=ax_bottom,
    signals=signals_notch,
    fs=fs,
    sweep_lengths=sweep_lengths,
    window_sec=WINDOW_SEC,
    y_range=Y_RANGE,
    initial_events=existing_events_per_sweep,
    lowpass_hz=LOWPASS_HZ,
    ax_ref=ax_top,
    signals_ref=signals_lp,
)

plt.show()

# Collect and save events after GUI is closed
all_events = []
for sweep_idx, events in enumerate(annotator.events_per_sweep):
    for (s, e) in events:
        all_events.append(
            {
                "sweep": int(sweep_idx),
                "start_sec": float(s),
                "end_sec": float(e),
            }
        )

data = {
    "abf_path": str(abf_path),
    "fs": fs,
    "filtered": True,
    "events": all_events,
}

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Saved annotations to {output_json}")
