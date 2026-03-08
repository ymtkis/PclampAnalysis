import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pyabf
import yaml
from scipy.signal import butter, filtfilt, iirnotch
from config import CFG


def preprocess_signal(signal, fs: float,
                      notch_harmonics: int, notch_Q: float,
                      lowpass_hz: float, lowpass_order: int) -> np.ndarray:
    sig = np.asarray(signal, dtype=float)

    # notch: 50Hz harmonics
    for k in range(1, int(notch_harmonics) + 1):
        f0 = 50.0 * k
        if f0 >= fs / 2:
            break
        b, a = iirnotch(f0, notch_Q, fs=fs)
        sig = filtfilt(b, a, sig)

    # lowpass
    b, a = butter(int(lowpass_order), float(lowpass_hz) / (fs / 2), btype="low")
    sig = filtfilt(b, a, sig)
    return sig.astype(np.float32)


def highpass(sig: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    b, a = butter(int(order), float(cutoff_hz) / (fs / 2), btype="high")
    return filtfilt(b, a, sig)


def is_finite_number(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def intervals_by_sweep_from_json(data: dict) -> Dict[int, List[Tuple[float, float, float]]]:
    intervals_raw = data.get("cnn_intervals", data.get("events", []))
    out = {}  # type: Dict[int, List[Tuple[float, float, float]]]
    for ev in intervals_raw:
        sw = int(ev["sweep"])
        s = float(ev["start_sec"])
        e = float(ev["end_sec"])
        p = float(ev.get("max_proba", np.nan))
        out.setdefault(sw, []).append((s, e, p))
    for sw in out:
        out[sw].sort(key=lambda x: x[0])
    return out


def parent_max_proba(intervals_for_sweep: List[Tuple[float, float, float]], t: float) -> float:
    for s, e, p in intervals_for_sweep:
        if s <= t <= e:
            return float(p)
    return np.nan


def recompute_amp_snr(
    sig_hp: np.ndarray, fs: float, peak_sec: float,
    baseline_before_sec: float, baseline_win_sec: float,
    local_sd_win_sec: float,
    min_amp_pA: float,
) -> Tuple[float, float, float, float]:
    idx = int(round(peak_sec * fs))
    idx = max(0, min(idx, len(sig_hp) - 1))

    bl_end = idx - int(round(baseline_before_sec * fs))
    bl_start = bl_end - int(round(baseline_win_sec * fs))
    if bl_start < 0 or bl_end <= bl_start:
        return (np.nan, np.nan, np.nan, np.nan)

    baseline = float(sig_hp[bl_start:bl_end].mean())
    amp = float(baseline - sig_hp[idx])

    sd_half = int(round(local_sd_win_sec * fs))
    sd0 = max(0, idx - sd_half)
    sd1 = min(len(sig_hp), idx + sd_half)
    if sd1 <= sd0 + 1:
        return (amp, baseline, np.nan, np.nan)

    window = sig_hp[sd0:sd1]

    # prediction本体と同様：イベントっぽい部分を除外してSDを見積もる
    amp_window = baseline - window
    mask = amp_window < float(min_amp_pA)

    if np.any(mask):
        local_sd = float(window[mask].std())
    else:
        local_sd = float(window.std())

    snr = float(amp / local_sd) if (np.isfinite(amp) and np.isfinite(local_sd) and local_sd > 0) else np.nan
    return (amp, baseline, local_sd, snr)



def refined_by_sweep(refined_events: List[dict]) -> Dict[int, List[dict]]:
    out = {}  # type: Dict[int, List[dict]]
    for ev in refined_events:
        sw = int(ev["sweep"])
        out.setdefault(sw, []).append(ev)
    for sw in out:
        out[sw].sort(key=lambda d: float(d["peak_sec"]))
    return out


def nearest_peak(auto_list: List[dict], t: float, tol_sec: float) -> Optional[dict]:
    if not auto_list:
        return None
    best = None
    best_dt = 1e18
    for ev in auto_list:
        dt = abs(float(ev["peak_sec"]) - t)
        if dt < best_dt:
            best_dt = dt
            best = ev
    return best if best_dt <= tol_sec else None


def main():
    ap = argparse.ArgumentParser(description="Recompute amp/snr/parent_max_proba for edited peaks (diff-only fast path).")
    ap.add_argument("--auto", required=True, help="***_cnn.json")
    ap.add_argument("--edited", required=True, help="***_cnn_edited.json (will be overwritten)")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    filt = CFG["filter"]
    post = CFG["cnn_postprocess"]

    auto_path = Path(args.auto)
    edited_path = Path(args.edited)

    auto_data = json.loads(auto_path.read_text(encoding="utf-8"))
    edited_data = json.loads(edited_path.read_text(encoding="utf-8"))

    abf_path = Path(edited_data["abf_path"])
    abf = pyabf.ABF(str(abf_path))
    fs = float(abf.dataRate)

    itv_by_sw = intervals_by_sweep_from_json(edited_data)
    auto_by_sw = refined_by_sweep(auto_data.get("refined_events", []))

    hp_cache = {}  # type: Dict[int, np.ndarray]

    hp_hz = float(post["highpass_hz"])
    hp_order = int(post.get("highpass_order", 2))
    baseline_before_sec = float(post["baseline_before_sec"])
    baseline_win_sec = float(post["baseline_win_sec"])
    local_sd_win_sec = float(post["local_sd_win_sec"])
    min_amp_pA = float(post["min_peak_ampl_pA"])

    updated = []
    n_total = 0
    n_recomputed = 0
    n_reused = 0

    for ev in edited_data.get("refined_events", []):
        n_total += 1
        sw = int(ev["sweep"])
        t = float(ev["peak_sec"])

        need = (not is_finite_number(ev.get("amp_pA"))) or (not is_finite_number(ev.get("snr"))) or (not is_finite_number(ev.get("parent_max_proba")))

        if not need:
            updated.append({
                "sweep": sw,
                "peak_sec": t,
                "amp_pA": float(ev["amp_pA"]),
                "snr": float(ev["snr"]),
                "parent_max_proba": float(ev["parent_max_proba"]),
            })
            continue

        near = nearest_peak(auto_by_sw.get(sw, []), t, float(args.tol))
        if near is not None and is_finite_number(near.get("amp_pA")) and is_finite_number(near.get("snr")) and is_finite_number(near.get("parent_max_proba")):
            updated.append({
                "sweep": sw,
                "peak_sec": t,
                "amp_pA": float(near["amp_pA"]),
                "snr": float(near["snr"]),
                "parent_max_proba": float(near["parent_max_proba"]),
            })
            n_reused += 1
            continue

        if sw not in hp_cache:
            abf.setSweep(sw)
            sig = preprocess_signal(
                abf.sweepY, fs,
                filt["notch_max_harmonic"], filt["notch_Q"],
                filt["lowpass_hz"], filt["lowpass_order"],
            )
            hp_cache[sw] = highpass(sig, fs, hp_hz, hp_order)

        amp, baseline, local_sd, snr = recompute_amp_snr(
            hp_cache[sw], fs, t,
            baseline_before_sec, baseline_win_sec, local_sd_win_sec,
            min_amp_pA
        )
        pmp = parent_max_proba(itv_by_sw.get(sw, []), t)

        updated.append({
            "sweep": sw,
            "peak_sec": t,
            "amp_pA": amp,
            "snr": snr,
            "parent_max_proba": pmp,
        })
        n_recomputed += 1

    updated.sort(key=lambda d: (int(d["sweep"]), float(d["peak_sec"])))
    edited_data["refined_events"] = updated
    edited_path.write_text(json.dumps(edited_data, indent=2), encoding="utf-8")

    print("[OK] wrote:", edited_path)
    print("  total:", n_total, "recomputed:", n_recomputed, "reused:", n_reused)


if __name__ == "__main__":
    main()
