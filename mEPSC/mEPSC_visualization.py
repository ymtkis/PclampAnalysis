import os
import json
import yaml
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyabf
from scipy.ndimage import gaussian_filter1d

from config import CFG
from filters import highpass_filter, lowpass_filter, bandpass_filter, remove_hum_harmonics
from plotting import mEPSC_cdf, plot_average_amp, plot_freq, plot_scaling
from scaling import permutation_test_cells



def mEPSC_event_annotation(
    sweep_data,
    event_times_all,
    fs,
    sweep_duration,
    condition=None,
    colors=None,
    t_start=0.4,
    t_end=10.4,
    sweep_gap_pA=110.0,
    amp_bar_pA=40.0,
    time_bar_s=1.0,
    width_scale=2.25,
):
    n_sweeps, n_samples = sweep_data.shape
    t = np.arange(n_samples) / fs

    i0 = max(0, int(round(t_start * fs)))
    i1 = min(n_samples, int(round(t_end * fs)))
    if i1 <= i0:
        raise ValueError(f"Invalid window: [{t_start}, {t_end}]")

    t_rel = t[i0:i1] - t_start
    x_max = float(t_end - t_start)

    base_w = 10.0
    fig_w = base_w * float(width_scale)
    fig, ax = plt.subplots(figsize=(fig_w, max(4, n_sweeps * 0.55)))

    trace_color = "black"
    trace_ls = "-"
    baseline_color = "0.7"
    marker_color = "red"

    offsets = [(n_sweeps - 1 - i) * float(sweep_gap_pA) for i in range(n_sweeps)]

    for i in range(n_sweeps):
        baseline = gaussian_filter1d(
            bandpass_filter(sweep_data[i], fs, lowcut=1000.0, highcut=2000.0),
            sigma=100,
        )
        segment = remove_hum_harmonics(
            sweep_data[i], fs, base_freq=50.0, max_freq=500.0, Q=30.0
        )
        filtered = highpass_filter(
            lowpass_filter(segment, fs, cutoff=500.0), fs, cutoff=4.0
        )

        baseline_w = baseline[i0:i1]
        filtered_w = filtered[i0:i1]
        offset = offsets[i]

        ax.plot(
            t_rel,
            filtered_w + offset,
            color=trace_color,
            linestyle=trace_ls,
            linewidth=0.15,
        )
        ax.plot(t_rel, baseline_w + offset, color=baseline_color, linewidth=0.35)

    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_ylabel("")

    ax.set_xlim(0.0, x_max)
    xt = np.arange(0.0, x_max + 1e-9, 2.0)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{int(v)}" for v in xt])

    # Fixed y-limits to allow clipping of large events.
    y_low = -0.6 * float(sweep_gap_pA)
    y_high = offsets[0] + 0.6 * float(sweep_gap_pA)
    ax.set_ylim(y_low, y_high)

    # Clip markers to visible range
    y_min, y_max = ax.get_ylim()
    y_rng = y_max - y_min
    y_clip_hi = y_max - 0.01 * y_rng
    y_clip_lo = y_min + 0.01 * y_rng

    for i in range(n_sweeps):
        segment = remove_hum_harmonics(
            sweep_data[i], fs, base_freq=50.0, max_freq=500.0, Q=30.0
        )
        filtered = highpass_filter(
            lowpass_filter(segment, fs, cutoff=500.0), fs, cutoff=4.0
        )
        offset = offsets[i]

        ev_t = [tt for tt in event_times_all[i] if (t_start <= tt <= t_end)]
        if not ev_t:
            continue

        ev_x = [tt - t_start for tt in ev_t]
        ev_idx = [min(max(int(round(tt * fs)), i0), i1 - 1) for tt in ev_t]
        ev_y_raw = [filtered[j] + offset for j in ev_idx]
        ev_y = [min(max(y, y_clip_lo), y_clip_hi) for y in ev_y_raw]

        ax.plot(ev_x, ev_y, "v", color=marker_color, markersize=1.8)

    # Scale bar placement: move down by ~1 sweep row; bring "1 s" text closer to bar.
    x0 = x_max - float(time_bar_s) - 0.35

    y0_base = y_min + 0.05 * y_rng
    y0 = max(y_min + 0.01 * y_rng, y0_base - float(sweep_gap_pA))

    ax.plot([x0, x0 + time_bar_s], [y0, y0], color="black", linewidth=1.5)
    ax.text(
        x0 + time_bar_s / 2,
        y0 - 0.012 * y_rng,
        f"{time_bar_s:g} s",
        ha="center",
        va="top",
        fontsize=8,
    )

    ax.plot([x0, x0], [y0, y0 + amp_bar_pA], color="black", linewidth=1.5)
    ax.text(
        x0 - 0.02 * x_max,
        y0 + amp_bar_pA / 2,
        f"{amp_bar_pA:g} pA",
        ha="right",
        va="center",
        fontsize=8,
    )

    fig.tight_layout()
    return fig


def _safe_sheet_name(name: str) -> str:
    # Excel sheet name limit: 31 chars and cannot contain []:*?/\
    bad = '[]:*?/\\'
    out = "".join("_" if ch in bad else ch for ch in str(name))
    out = out[:31]
    return out if out else "Sheet"


def run_one_set(CFG, conditions, set_name, colors, cell_tables):
    base_path = CFG["paths"]["base_path"]
    protocol = CFG["protocol"]["mepsc"]

    data_root = CFG["paths"]["data_root"]
    experiment_file = CFG["paths"]["experiment_file"]
    results_root_name = CFG["results"]["results_subdir"]
    prefix = CFG["paths"]["prefix"]

    data_path = f"{base_path}/{data_root}/{protocol}"
    results_root = f"{base_path}/{results_root_name}/{protocol}"
    fig_root = f"{results_root}/figure"
    Path(fig_root).mkdir(parents=True, exist_ok=True)

    out_dir = Path(fig_root) / set_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cnn_det_subdir = CFG.get("results", {}).get("cnn_detection_subdir", "cnn_detections")
    detection_dir = f"{results_root}/{cnn_det_subdir}"

    sweep_duration = CFG["event"]["sweep_duration"]
    cnn_mepsc_CFG = CFG.get("cnn_mepsc", {})
    sweep_start = float(cnn_mepsc_CFG.get("valid_start_sec", 0.4))
    valid_end = float(cnn_mepsc_CFG.get("valid_end_sec", sweep_start + sweep_duration))
    min_peak_ampl_pA = float(cnn_mepsc_CFG.get("min_peak_ampl_pA", 4.0))

    scfg = CFG.get("scaling", {})
    scaling_enabled = bool(scfg.get("enabled", True))
    if scaling_enabled:
        n_perm = int(scfg.get("n_perm", 2000))
        seed = int(scfg.get("seed", 0))
        metric = str(scfg.get("metric", "ks"))
        pairs_mode = str(scfg.get("pairs", "vs_sham"))  
        min_events_per_cell = int(scfg.get("min_events_per_cell", 30))

    event_freq_all = {}
    event_amps_all = {}
    avg_amps_all = {}

    for condition in conditions:
        cond_fig_dir = Path(fig_root) / "event_annotation" / condition
        cond_fig_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_excel(f"{base_path}/{experiment_file}", sheet_name=condition)
        mouse_ids = df["id"].tolist()
        slice_nums = df["slice"].tolist()
        cts = df["cell"].tolist()

        freq_summary = {}
        amps_summary = {}
        avg_summary = {}

        per_cell_rows = []

        for mouse_id, slice_num, ct in zip(mouse_ids, slice_nums, cts):
            if ct != "pyramidal":
                continue

            abf_path = f"{data_path}/{condition}/{prefix}_{mouse_id}_{condition}_{slice_num}_{protocol}.abf"
            if not os.path.exists(abf_path):
                continue

            abf = pyabf.ABF(abf_path)
            n_sweeps = abf.sweepCount
            n_samples = abf.sweepPointCount
            fs = abf.dataRate

            sweep_data = np.zeros((n_sweeps, n_samples))
            for i_sweep in range(n_sweeps):
                abf.setSweep(i_sweep)
                sweep_data[i_sweep, :] = abf.sweepY

            det_json = Path(detection_dir) / condition / f"{Path(abf_path).stem}_cnn_edited.json"
            if not det_json.exists():
                print(f"[WARN] Detection JSON not found: {det_json}")
                continue

            with det_json.open("r", encoding="utf-8") as f:
                det_data = json.load(f)

            events = det_data.get("refined_events", [])
            if not events:
                continue

            event_times_all = [[] for _ in range(n_sweeps)]
            event_counts = [0] * n_sweeps
            event_amps = []

            for ev in events:
                sweep_idx = int(ev["sweep"])
                t_peak = float(ev["peak_sec"])
                amp_pA = float(ev["amp_pA"])

                if t_peak < sweep_start or t_peak > valid_end:
                    continue
                if abs(amp_pA) < min_peak_ampl_pA:
                    continue
                if 0 <= sweep_idx < n_sweeps:
                    event_times_all[sweep_idx].append(t_peak)
                    event_counts[sweep_idx] += 1
                    event_amps.append(amp_pA)

            if not event_amps:
                continue

            fig_evt = mEPSC_event_annotation(
                sweep_data,
                event_times_all,
                fs,
                sweep_duration,
                condition=condition,
                colors=colors,
                t_start=sweep_start,
                t_end=valid_end,
                sweep_gap_pA=110.0,
                amp_bar_pA=40.0,
                width_scale=2.25,
            )
            out_evt = cond_fig_dir / f"{prefix}_{mouse_id}_{condition}_{slice_num}_{protocol}_cnn.svg"
            fig_evt.savefig(out_evt, format="svg")
            plt.close(fig_evt)

            total_valid_time = (valid_end - sweep_start) * len(event_counts)
            key = f"{mouse_id}_{slice_num}"

            n_events = int(sum(event_counts))
            freq_hz = float(n_events / total_valid_time)
            mean_amp = float(np.mean(np.abs(event_amps)))

            freq_summary[key] = freq_hz
            amps_summary[key] = event_amps
            avg_summary[key] = mean_amp

            per_cell_rows.append(
                {
                    "set": set_name,
                    "condition": condition,
                    "mouse_id": mouse_id,
                    "slice": slice_num,
                    "cell_key": key,
                    "n_sweeps": n_sweeps,
                    "valid_start_s": sweep_start,
                    "valid_end_s": valid_end,
                    "valid_time_s": total_valid_time,
                    "n_events": n_events,
                    "frequency_hz": freq_hz,
                    "mean_amp_pA": mean_amp,
                }
            )

        event_freq_all[condition] = freq_summary
        event_amps_all[condition] = amps_summary
        avg_amps_all[condition] = avg_summary

        if per_cell_rows:
            df_cells = pd.DataFrame(per_cell_rows)
            # Accumulate across sets; sheet per condition (append rows).
            cell_tables.setdefault(condition, []).append(df_cells)

    cdf_fig, cdf_stats = mEPSC_cdf(event_amps_all, colors=colors)
    avg_fig, avg_stats = plot_average_amp(avg_amps_all, colors=colors)
    freq_fig, freq_stats = plot_freq(event_freq_all, colors=colors)

    cdf_fig.savefig(out_dir / f"mEPSC_amp_distribution_{set_name}.svg", format="svg")
    avg_fig.savefig(out_dir / f"mEPSC_average_amp_{set_name}.svg", format="svg")
    freq_fig.savefig(out_dir / f"mEPSC_frequency_{set_name}.svg", format="svg")

    cdf_stats.to_csv(out_dir / f"mEPSC_amp_distribution_stats_{set_name}.csv", index=False)
    avg_stats.to_csv(out_dir / f"mEPSC_average_amp_stats_{set_name}.csv", index=False)
    freq_stats.to_csv(out_dir / f"mEPSC_frequency_stats_{set_name}.csv", index=False)


    cells_by_cond = {}
    for cond, amps_summary in event_amps_all.items():
        cells = []
        for a in amps_summary.values():
            if a is None:
                continue
            arr = np.asarray(a, dtype=float)
            arr = np.abs(arr[np.isfinite(arr)])
            if arr.size >= min_events_per_cell:
                cells.append(arr)
        cells_by_cond[cond] = cells

    conds_present = [c for c in conditions if c in cells_by_cond and len(cells_by_cond[c]) > 0]

    if len(conds_present) >= 2:

        if pairs_mode == "vs_sham" and "Sham" in conds_present:
            pairs = [("Sham", c) for c in conds_present if c != "Sham"]
        else:
            pairs = list(itertools.combinations(conds_present, 2))

        scaling_rows = []

        for cA, cB in pairs:
            cells_A = cells_by_cond[cA]
            cells_B = cells_by_cond[cB]
            if len(cells_A) < 2 or len(cells_B) < 2:
                print(f"[WARN] scaling skip (too few cells): {cA} n={len(cells_A)}, {cB} n={len(cells_B)}")
                continue

            T_obs, T_null, p = permutation_test_cells(
                cells_A, cells_B,
                n_perm=n_perm,
                seed=seed,
                metric=metric
            )

            fig_s = plot_scaling(
                T_obs, T_null, p,
                title=f"{set_name}: {cA} vs {cB}",
                xlabel=f"Mismatch score ({metric.upper()} after scaling)"
            )

            out_svg = out_dir / f"mEPSC_scaling_{set_name}_{cA}_vs_{cB}.svg"
            fig_s.savefig(out_svg, format="svg")
            plt.close(fig_s)

            scaling_rows.append({
                "set": set_name,
                "A": cA,
                "B": cB,
                "metric": metric,
                "n_perm": n_perm,
                "seed": seed,
                "min_events_per_cell": min_events_per_cell,
                "n_cells_A": len(cells_A),
                "n_cells_B": len(cells_B),
                "T_obs": float(T_obs) if np.isfinite(T_obs) else np.nan,
                "p": float(p) if np.isfinite(p) else np.nan,
            })

        if scaling_rows:
            pd.DataFrame(scaling_rows).to_csv(
                out_dir / f"mEPSC_scaling_stats_{set_name}.csv",
                index=False
            )

    print(f"[OK] Finished: {set_name}")


def main():

    colors = CFG["plot"]["colors"]
    sets = CFG["plot"]["sets"]

    base_path = CFG["paths"]["base_path"]
    protocol = CFG["protocol"]["mepsc"]
    results_root_name = CFG["results"]["results_subdir"]
    results_root = Path(base_path) / results_root_name / protocol
    fig_root = results_root / "figure"
    fig_root.mkdir(parents=True, exist_ok=True)

    cell_tables = {}
    for set_name, conds in sets.items():
        run_one_set(CFG, conds, set_name=set_name, colors=colors, cell_tables=cell_tables)

    # One Excel file, one sheet per condition (rows include set name).
    out_xlsx = fig_root / "cell_metrics_by_condition.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for condition, parts in cell_tables.items():
            df_all = pd.concat(parts, ignore_index=True)
            sheet = _safe_sheet_name(condition)
            df_all.to_excel(writer, sheet_name=sheet, index=False)

    print(f"[OK] wrote: {out_xlsx}")


if __name__ == "__main__":
    main()
