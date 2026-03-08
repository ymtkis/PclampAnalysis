import json
import yaml
from pathlib import Path

import numpy as np
import pyabf
from tqdm import tqdm
from scipy.signal import iirnotch, filtfilt, butter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.cnn_simple import Simple1DCNN
from .config import CFG


base_path    = Path(CFG["paths"]["base_path"])
protocol     = CFG["protocol"]["mepsc"]
results_root = base_path / CFG["results"]["results_subdir"] / protocol

ANNOTATION_DIR = results_root / CFG["results"]["cnn_annotation_subdir"]
DATASET_DIR    = results_root / CFG["results"]["cnn_dataset_subdir"]
MODEL_DIR      = results_root / CFG["results"]["cnn_model_subdir"]
# Use v1 model (trained without hard negatives)
MODEL_PATH     = MODEL_DIR / "mEPSC_cnn_v1.pt"

# Filter settings (same as training/prediction)
filter_cfg          = CFG["filter"]
NOTCH_MAX_HARMONIC  = filter_cfg["notch_max_harmonic"]
NOTCH_Q             = filter_cfg["notch_Q"]
LOWPASS_HZ          = filter_cfg["lowpass_hz"]
LOWPASS_ORDER       = filter_cfg["lowpass_order"]

# CNN parameters
cnn_cfg         = CFG["cnn"]
WINDOW_SEC      = cnn_cfg["window_sec"]
STEP_SEC        = cnn_cfg["step_sec"]
VALID_START_SEC = cnn_cfg.get("valid_start_sec", 0.4)
VALID_END_SEC   = cnn_cfg.get("valid_end_sec", 10.4)

# Hard negative parameters
HARD_NEG_POS_RATIO = float(cnn_cfg.get("hard_neg_pos_ratio", 1.0))
EDGE_MARGIN_SEC    = float(cnn_cfg.get("edge_margin_sec", 0.02))


def preprocess_signal(signal, fs,
                      notch_harmonics=NOTCH_MAX_HARMONIC,
                      notch_Q=NOTCH_Q,
                      lowpass_hz=LOWPASS_HZ,
                      lowpass_order=LOWPASS_ORDER):
    """Apply 50 Hz notch (with harmonics) and low-pass filter."""
    signal = np.asarray(signal, dtype=float)

    for k in range(1, notch_harmonics + 1):
        f0 = 50.0 * k
        if f0 >= fs / 2:
            break
        b, a = iirnotch(w0=f0, Q=notch_Q, fs=fs)
        signal = filtfilt(b, a, signal)

    nyq = fs / 2.0
    cutoff = lowpass_hz / nyq
    b, a = butter(lowpass_order, cutoff, btype="low")
    signal = filtfilt(b, a, signal)

    return signal.astype(np.float32)



class WindowDataset(Dataset):
    """Z-score normalize each window before feeding to CNN."""
    def __init__(self, segments):
        self.segments = np.asarray(segments, dtype=np.float32)

        mean = self.segments.mean(axis=1, keepdims=True)
        std = self.segments.std(axis=1, keepdims=True) + 1e-6
        self.segments = (self.segments - mean) / std

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x = self.segments[idx]  # (T,)
        x = np.expand_dims(x, axis=0)  # (1, T)
        return torch.from_numpy(x)


def split_by_group(groups, test_ratio=0.2, val_ratio=0.1, seed=42):
    """Same group-wise split as in 3_Training.py."""
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 3:
        raise ValueError(
            f"Too few groups for group-wise split (n_groups={n_groups}). "
            f"Provide at least 3 distinct group IDs."
        )

    rng = np.random.RandomState(seed)
    perm_groups = rng.permutation(unique_groups)

    n_test_groups  = max(1, int(test_ratio * n_groups))
    n_val_groups   = max(1, int(val_ratio * n_groups))
    n_train_groups = n_groups - n_val_groups - n_test_groups
    if n_train_groups <= 0:
        raise ValueError(
            f"Not enough groups for training after split: "
            f"train={n_train_groups}, val={n_val_groups}, test={n_test_groups}"
        )

    test_groups  = perm_groups[:n_test_groups]
    val_groups   = perm_groups[n_test_groups:n_test_groups + n_val_groups]
    train_groups = perm_groups[n_test_groups + n_val_groups:]

    group_to_indices = {}
    for idx, g in enumerate(groups):
        group_to_indices.setdefault(g, []).append(idx)

    train_idx = []
    val_idx   = []
    test_idx  = []

    for g in train_groups:
        train_idx.extend(group_to_indices[g])
    for g in val_groups:
        val_idx.extend(group_to_indices[g])
    for g in test_groups:
        test_idx.extend(group_to_indices[g])

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx   = np.array(val_idx, dtype=np.int64)
    test_idx  = np.array(test_idx, dtype=np.int64)

    return train_idx, val_idx, test_idx


def load_annotation_files(annotation_dir: Path):
    return sorted(annotation_dir.glob("events_*.json"))


def load_abf_all_sweeps(abf_path: Path):
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


def window_time_intervals_for_sweep(total_sec, fs, window_sec, step_sec,
                                    valid_start_sec, valid_end_sec):
    """Generate (t_start, t_end, idx0, idx1) for sliding windows."""
    win_len = int(round(window_sec * fs))
    t0 = max(valid_start_sec, 0.0)
    t1 = min(valid_end_sec, total_sec)

    t = t0
    while t + window_sec <= t1:
        idx0 = int(round(t * fs))
        idx1 = idx0 + win_len
        yield t, t + window_sec, idx0, idx1
        t += step_sec


def interval_overlaps_any(t_start, t_end, events, margin):
    """Check if [t_start, t_end] overlaps any (s,e) with margin."""
    for s, e in events:
        if t_end < s - margin:
            continue
        if t_start > e + margin:
            continue
        return True
    return False


def main():
    groups_path = DATASET_DIR / "groups.npy"
    if not groups_path.exists():
        raise FileNotFoundError(f"{groups_path} not found. Run 2_Create_dataset.py first.")
    groups = np.load(groups_path)
    N = groups.shape[0]
    print(f"Loaded groups.npy with N={N}")

    # Use the same split as training to identify train groups
    train_idx, val_idx, test_idx = split_by_group(
        groups,
        test_ratio=0.2,
        val_ratio=0.1,
        seed=42,
    )
    train_group_ids = np.unique(groups[train_idx])
    print(f"Train groups (for hard negatives): {train_group_ids.tolist()}")

    # Load v1 model (trained without hard negatives)
    model_ckpt = torch.load(str(MODEL_PATH), map_location="cpu")
    if isinstance(model_ckpt, dict) and "model_state_dict" in model_ckpt:
        state_dict     = model_ckpt["model_state_dict"]
        input_length   = int(model_ckpt["input_length"])
        best_threshold = float(model_ckpt.get("best_threshold", 0.5))
    else:
        state_dict     = model_ckpt
        input_length   = int(WINDOW_SEC * 20000)  # fallback
        best_threshold = 0.5

    print(f"Loaded model from {MODEL_PATH}")
    print(f"Using best_threshold for hard negatives: {best_threshold:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Simple1DCNN(input_length=input_length).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    rng = np.random.default_rng(123)

    X_hard = []
    y_hard = []
    groups_hard = []

    ann_files = load_annotation_files(ANNOTATION_DIR)
    print(f"Found {len(ann_files)} annotation files.")

    for file_idx, ann_path in enumerate(tqdm(ann_files, desc="Hard-negative mining")):
        group_id = int(file_idx)
        if group_id not in train_group_ids:
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        abf_path  = Path(ann["abf_path"])
        events_raw = ann.get("events", [])
        if not events_raw:
            continue

        events_by_sweep = {}
        for ev in events_raw:
            sw = int(ev["sweep"])
            s  = float(ev["start_sec"])
            e  = float(ev["end_sec"])
            events_by_sweep.setdefault(sw, []).append((s, e))

        signals, fs, sweep_lengths = load_abf_all_sweeps(abf_path)
        n_sweeps = len(signals)
        win_sec  = WINDOW_SEC
        win_len  = int(round(win_sec * fs))

        for sw, events in events_by_sweep.items():
            if sw < 0 or sw >= n_sweeps:
                continue

            signal    = signals[sw]
            total_sec = float(sweep_lengths[sw])

            use_start = max(VALID_START_SEC, 0.0)
            use_end   = min(VALID_END_SEC, total_sec)
            if use_end - use_start <= win_sec:
                continue

            events_in_range = [
                (s, e) for (s, e) in events
                if (s >= use_start) and (e <= use_end)
            ]
            if not events_in_range:
                continue

            n_pos        = len(events_in_range)
            n_hard_target = int(HARD_NEG_POS_RATIO * n_pos)
            if n_hard_target <= 0:
                continue

            segments = []
            meta     = []  # (t_start, t_end)
            for t_start, t_end, idx0, idx1 in window_time_intervals_for_sweep(
                    total_sec, fs, win_sec, STEP_SEC, use_start, use_end):

                if idx1 > len(signal):
                    continue
                seg = signal[idx0:idx1]
                if len(seg) != win_len:
                    continue
                segments.append(seg)
                meta.append((t_start, t_end))

            if not segments:
                continue

            ds     = WindowDataset(segments)
            loader = DataLoader(ds, batch_size=256, shuffle=False)

            proba_all = []
            with torch.no_grad():
                for xb in loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    proba  = torch.softmax(logits, dim=1)[:, 1]
                    proba_all.extend(proba.cpu().numpy().tolist())
            proba_all = np.array(proba_all, dtype=float)

            candidates = []
            for (t_start, t_end), p in zip(meta, proba_all):
                if p < best_threshold:
                    continue
                if interval_overlaps_any(t_start, t_end, events_in_range, EDGE_MARGIN_SEC):
                    continue
                candidates.append((t_start, p))

            if not candidates:
                continue

            if len(candidates) > n_hard_target:
                idx_sel = rng.choice(len(candidates), size=n_hard_target, replace=False)
                candidates = [candidates[i] for i in idx_sel]

            for t_start, p in candidates:
                idx0 = int(round(t_start * fs))
                idx1 = idx0 + win_len
                if idx1 > len(signal):
                    continue
                seg = signal[idx0:idx1]
                if len(seg) != win_len:
                    continue
                X_hard.append(seg)
                y_hard.append(0)
                groups_hard.append(group_id)

    if not X_hard:
        print("No hard negatives found. "
              "You may lower best_threshold or check your annotations.")
        return

    X_hard      = np.stack(X_hard, axis=0)
    y_hard      = np.array(y_hard, dtype=np.int64)
    groups_hard = np.array(groups_hard, dtype=np.int64)

    np.save(DATASET_DIR / "X_hard.npy", X_hard)
    np.save(DATASET_DIR / "y_hard.npy", y_hard)
    np.save(DATASET_DIR / "groups_hard.npy", groups_hard)

    print(f"Saved hard negatives: X_hard shape = {X_hard.shape}, "
          f"y_hard shape = {y_hard.shape}, "
          f"#unique groups = {len(np.unique(groups_hard))}")


if __name__ == "__main__":
    main()
