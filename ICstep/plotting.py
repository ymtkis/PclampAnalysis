# ICstep/plotting.py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import sem

from .filters import lowpass_filter, bandpass_filter
from .stats_utils import statistics

try:
    from .config import CFG
    colors = CFG.get("plot", {}).get("colors", {})
except Exception:
    colors = {}



def _linestyle_for_condition(condition: str):
    if condition == "Sham":
        return "-"
    elif "24h" in condition:
        return "--"
    elif "48h)+2h" in condition:
        return "-."
    elif "48h)+24h" in condition:
        return (0, (5, 3, 1, 3, 1, 3))
    else:
        return ":"

def _p_to_marker(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _annotate_significant_pairs(ax, conditions, stats_df, y_step_frac=0.08, alpha=0.05, show_p=False):
    if stats_df is None or len(stats_df) == 0:
        return

    if "p-adjusted" in stats_df.columns:
        pcol = "p-adjusted"
    elif "p-value" in stats_df.columns:
        pcol = "p-value"
    else:
        return

    sig = stats_df[stats_df[pcol] < alpha].copy()
    if len(sig) == 0:
        return

    y_min, y_max = ax.get_ylim()
    y_range = float(y_max - y_min) if y_max > y_min else 1.0

    y_step = max(y_range * float(y_step_frac), y_range * 0.03)

    needed_top = y_max + (len(sig) + 1) * y_step
    ax.set_ylim(y_min, needed_top)
    y_min, y_max = ax.get_ylim()

    for j, (_, row) in enumerate(sig.iterrows()):
        cond1 = row["Condition A"]
        cond2 = row["Condition B"]
        pval = float(row[pcol])

        if cond1 not in conditions or cond2 not in conditions:
            continue

        xi = conditions.index(cond1)
        xj = conditions.index(cond2)

        y_line = y_max - (len(sig) - j) * y_step
        x_center = (xi + xj) / 2.0
        line_len = abs(xj - xi) * 0.8
        x_start = x_center - line_len / 2.0
        x_end = x_center + line_len / 2.0

        ax.plot([x_start, x_end], [y_line, y_line], color="k", linewidth=1.0)
        ax.plot([x_start, x_start], [y_line, y_line - y_range / 30], color="k", linewidth=1.0)
        ax.plot([x_end, x_end], [y_line, y_line - y_range / 30], color="k", linewidth=1.0)

        label = f"p={pval:.3g}" if show_p else _p_to_marker(pval)
        ax.text(x_center, y_line + y_range * 0.01, label, ha="center", va="bottom", fontsize=10)



def ICstep_event_annotation(sweep_data, event_times_all, holding_currents, fs, sweep_start, sweep_duration):
    n_sweeps, n_samples = sweep_data.shape
    window_padding = 0.05

    window_start = int((sweep_start - window_padding) * fs)
    window_end = int((sweep_start + sweep_duration + window_padding) * fs)
    window_len = window_end - window_start

    time = np.linspace(-window_padding, sweep_duration + window_padding, window_len)

    fig, ax = plt.subplots(figsize=(10, max(4, n_sweeps * 0.4)))

    for i in range(n_sweeps):
        segment = sweep_data[i, window_start:window_end]
        filtered = lowpass_filter(segment, fs, cutoff=2000)
        baseline = bandpass_filter(segment, fs, lowcut=2000.0, highcut=4000.0)
        offset = i * 0

        event_times = event_times_all[i]
        event_times_in_window = [
            t for t in event_times
            if (sweep_start - window_padding) <= t <= (sweep_start + sweep_duration + window_padding)
        ]
        event_times_rel = [t - sweep_start + 0.0 for t in event_times_in_window]

        event_indices = [int((t - (sweep_start - window_padding)) * fs) for t in event_times_in_window]
        event_values = [filtered[idx] + offset + 2 for idx in event_indices]

        ax.plot(time, filtered + offset, color="black", linewidth=0.4)
        ax.plot(event_times_rel, event_values, "v", color="red", markersize=2, label="event" if i == 0 else "")

    ax.set_xlabel("Time (s)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)

    fig.tight_layout()
    plt.close(fig)
    return fig


def f_i_plot(
    event_counts_all_conditions,
    currents,
    current_min=-100,
    current_max=200,
    conditions_order=None,
    title=None,
):
    fig, ax = plt.subplots(figsize=(6, 4))

    currents = np.array(currents)
    mask = (currents >= current_min) & (currents <= current_max)
    x_all = currents[mask]

    if conditions_order is None:
        conditions_order = list(event_counts_all_conditions.keys())

    for condition in conditions_order:
        if condition not in event_counts_all_conditions:
            continue

        df = event_counts_all_conditions[condition]
        df_use = df.copy()
        try:
            df_use.columns = [int(c) for c in df_use.columns]
        except Exception:
            pass

        x = [I for I in x_all if I in df_use.columns]
        if len(x) == 0:
            continue

        df_sel = df_use[x]
        y = df_sel.mean(axis=0, skipna=True).values
        yerr = df_sel.apply(sem, axis=0, nan_policy="omit").values

        ls = _linestyle_for_condition(condition)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            capsize=5,
            label=condition,
            color=colors.get(condition, "black"),
            linestyle=ls,
        )

    ax.set_xlabel("Current injected (pA)")
    ax.set_ylabel("Frequency (Hz)")
    if title is not None:
        ax.set_title(title)

    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig

def metrics_plot(spike_metrics_all_conditions):
    conditions = list(spike_metrics_all_conditions.keys())
    metrics_names = [
        "Peak amplitude (mV)",
        "Peak amplitude from onset (mV)",
        "Half duration (ms)",
        "Rise slope (mV/ms)",
        "Fall slope (mV/ms)",
    ]

    n_conditions = len(conditions)
    x_positions = np.arange(n_conditions)
    figs = {}
    stats_dfs_list = []

    ylims = [[40, 160], [0, 120], [0, 4.8], [40, 160], [0, 120]]

    for k, metric in enumerate(metrics_names):
        fig, ax = plt.subplots(figsize=(n_conditions, 4))
        means, errs = [], []
        values_all_conditions = {}
        used_x_positions = []

        for xi, condition in zip(x_positions, conditions):
            df = spike_metrics_all_conditions[condition]

            if metric not in df.index:
                continue

            values = df.loc[metric].values.astype(float)

            if len(values) == 0 or np.all(np.isnan(values)):
                continue

            jitter = np.random.uniform(-0.1, 0.1, size=len(values))
            ax.scatter(
                xi + jitter,
                np.abs(values),
                alpha=0.6,
                color=colors.get(condition, "black"),
            )

            means.append(np.nanmean(np.abs(values)))
            errs.append(sem(values, nan_policy="omit"))
            values_all_conditions[condition] = np.abs(values)
            used_x_positions.append(xi)

        if len(means) > 0:
            ax.errorbar(
                used_x_positions,
                means,
                yerr=errs,
                fmt="_",
                markersize=20,
                capsize=4,
                linestyle="None",
                color="k",
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_xlim([x_positions[0] - 0.5, x_positions[-1] + 0.5])

        ax.set_ylim(ylims[k])
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_yticks(np.arange(y_min, y_max * 1.1, y_range / 3))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if len(values_all_conditions) >= 2:
            stats_df = statistics(values_all_conditions, method="mannwhitneyu", correction="bonferroni")
            stats_df.index.name = metric

            _annotate_significant_pairs(ax, conditions, stats_df, y_step_frac=0.08, alpha=0.05, show_p=False)

            stats_dfs_list.append(stats_df)

        plt.tight_layout()
        figs[metric] = fig

    if len(stats_dfs_list) > 0:
        stats_dfs = pd.concat(stats_dfs_list, keys=metrics_names).droplevel(level=1)
    else:
        stats_dfs = pd.DataFrame()

    return figs, stats_dfs


def rheobase_plot(rheobase_all_conditions):
    conditions = list(rheobase_all_conditions.keys())
    n_conditions = len(conditions)
    x_positions = np.arange(n_conditions)

    fig, ax = plt.subplots(figsize=(n_conditions, 4))
    means, errs = [], []
    values_all_conditions = {}

    y_lim = [0, 330]
    for xi, condition in zip(x_positions, conditions):
        values = [
            val
            for val in rheobase_all_conditions[condition].loc[:, "rheobase_pA"]
            if not (isinstance(val, float) and math.isnan(val))
        ]
        if len(values) == 0:
            continue

        values = np.array(values)
        jitter = np.random.uniform(-0.1, 0.1, size=len(values))
        ax.scatter(xi + jitter, values, alpha=0.6, color=colors.get(condition, "black"))

        means.append(values.mean())
        errs.append(sem(values))
        values_all_conditions[condition] = values

    if len(means) > 0:
        ax.errorbar(
            x_positions[:len(means)],
            means,
            yerr=errs,
            fmt="_",
            markersize=20,
            capsize=4,
            linestyle="None",
            color="k",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylabel("Rheobase (pA)")
    ax.set_xlim([x_positions[0] - 0.5, x_positions[-1] + 0.5])
    ax.set_ylim(y_lim)
    y_min, y_max = ax.get_ylim()
    y_range = 300
    ax.set_yticks(np.arange(0, y_max * 1.1, y_range / 3))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if len(values_all_conditions) >= 2:
        stats_df = statistics(values_all_conditions, method="mannwhitneyu", correction="bonferroni")
        _annotate_significant_pairs(ax, conditions, stats_df, y_step_frac=0.08, alpha=0.05, show_p=False)
    else:
        stats_df = None

    plt.tight_layout()
    plt.close(fig)
    return fig, stats_df


def ri_plot(ri_all_conditions):
    conditions = list(ri_all_conditions.keys())
    n_conditions = len(conditions)
    x_positions = np.arange(n_conditions)

    fig, ax = plt.subplots(figsize=(n_conditions, 4))
    means, errs = [], []
    values_all_conditions = {}

    y_lim = [0, 300]
    for xi, condition in zip(x_positions, conditions):
        values = [
            val for val in ri_all_conditions[condition].loc[:, "ri"]
            if not (isinstance(val, float) and math.isnan(val))
        ]
        jitter = np.random.uniform(-0.1, 0.1, size=len(values))
        ax.scatter(xi + jitter, values, alpha=0.6, color=colors.get(condition, "black"))

        means.append(np.mean(values))
        errs.append(stats.sem(values))
        values_all_conditions[condition] = values

    ax.errorbar(
        x_positions,
        means,
        yerr=errs,
        fmt="_",
        markersize=20,
        capsize=4,
        linestyle="None",
        color="k",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylabel("Input resistance (MΩ)")
    ax.set_xlim([x_positions[0] - 0.5, x_positions[-1] + 0.5])
    ax.set_ylim(y_lim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if len(values_all_conditions) >= 2:
        stats_df = statistics(values_all_conditions, method="mannwhitneyu", correction="bonferroni")
        _annotate_significant_pairs(ax, conditions, stats_df, y_step_frac=0.08, alpha=0.05, show_p=False)
    else:
        stats_df = None
        
    fig.tight_layout()
    plt.close(fig)
    return fig, stats_df
