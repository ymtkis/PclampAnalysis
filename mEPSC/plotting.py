import numpy as np
import matplotlib.pyplot as plt
from stats_utils import statistics


def _style_for_condition(condition, colors):
    # Sham: gray
    if condition == "Sham":
        c = "0.6"
    else:
        # blue-scale for experimental conditions
        c = colors.get(condition, "#1f77b4")

    if condition == "Sham":
        ls = "-"
    elif "24h" in condition:
        ls = "--"
    elif "48h)+2h" in condition:
        ls = "-."
    elif "48h)+24h" in condition:
        ls = (0, (5, 3, 1, 3, 1, 3))
    else:
        ls = ":"

    return c, ls


def mEPSC_cdf(event_amps_all_condition, colors, n_bins=50):
    fig, ax = plt.subplots(figsize=(6, 4))
    values_for_stats = {}

    max_bin_index_95 = None

    for condition, event_amps_summary in event_amps_all_condition.items():
        cell_amps = [
            np.abs(v) for v in event_amps_summary.values()
            if v is not None and len(v) > 0
        ]
        if not cell_amps:
            continue

        all_values = np.concatenate(cell_amps)
        vmin = float(np.min(all_values))
        vmax = float(np.max(all_values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            continue

        bins = np.linspace(vmin, vmax, n_bins + 1)

        cdfs = []
        for amps in cell_amps:
            counts, _ = np.histogram(amps, bins=bins)
            s = np.sum(counts)
            if s == 0:
                continue
            cdfs.append(np.cumsum(counts) / s)

        if not cdfs:
            continue

        cdfs = np.vstack(cdfs)
        med = np.median(cdfs, axis=0)
        q1 = np.percentile(cdfs, 25, axis=0)
        q3 = np.percentile(cdfs, 75, axis=0)

        idx95 = np.where(med >= 0.95)[0]
        if len(idx95) > 0:
            i = idx95[0] + 1
            if max_bin_index_95 is None:
                max_bin_index_95 = i
            else:
                max_bin_index_95 = max(max_bin_index_95, i)

        c, ls = _style_for_condition(condition, colors)
        ax.plot(bins[1:], med, label=condition, color=c, linestyle=ls, linewidth=2)
        ax.fill_between(bins[1:], q1, q3, color=c, alpha=0.25)

        values_for_stats[condition] = [float(np.mean(a)) for a in cell_amps]

    ax.set_xlabel("Amplitude (pA)")
    ax.set_ylabel("Fraction")
    ax.legend()

    if max_bin_index_95 is not None:
        ax.set_xlim(bins[1], bins[1:][max_bin_index_95])

    stats_df = statistics(
        values_for_stats,
        method="mannwhitneyu",
        correction="bonferroni"
    )

    fig.tight_layout()
    plt.close(fig)
    return fig, stats_df

def plot_average_amp(avg_amps_all_conditions, colors):
    conditions = list(avg_amps_all_conditions.keys())
    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=(len(conditions), 4))
    values_all_conditions = {}

    for i, cond in enumerate(conditions):
        vals = np.asarray(list(avg_amps_all_conditions[cond].values()), dtype=float)
        vals = vals[np.isfinite(vals)]
        values_all_conditions[cond] = vals

        c, _ = _style_for_condition(cond, colors)
        jitter = np.random.uniform(-0.1, 0.1, size=len(vals))
        ax.scatter(i + jitter, np.abs(vals), alpha=0.6, color=c)

        if vals.size:
            med = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])
            ax.plot([i, i], [abs(q1), abs(q3)], color=c, linewidth=2)
            ax.plot([i - 0.15, i + 0.15], [abs(med), abs(med)], color=c, linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylabel("Mean amplitude (pA)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    stats_df = statistics(values_all_conditions, method="mannwhitneyu", correction="bonferroni")

    fig.tight_layout()
    plt.close(fig)
    return fig, stats_df

def plot_freq(event_freq_all_conditions, colors):
    conditions = list(event_freq_all_conditions.keys())
    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=(len(conditions), 4))
    values_all_conditions = {}

    for i, cond in enumerate(conditions):
        vals = np.asarray(list(event_freq_all_conditions[cond].values()), dtype=float)
        vals = vals[np.isfinite(vals)]
        values_all_conditions[cond] = vals

        c, _ = _style_for_condition(cond, colors)
        jitter = np.random.uniform(-0.1, 0.1, size=len(vals))
        ax.scatter(i + jitter, vals, alpha=0.6, color=c)

        if vals.size:
            med = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])
            ax.plot([i, i], [q1, q3], color=c, linewidth=2)
            ax.plot([i - 0.15, i + 0.15], [med, med], color=c, linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylabel("mEPSC frequency (Hz)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    stats_df = statistics(values_all_conditions, method="mannwhitneyu", correction="bonferroni")

    fig.tight_layout()
    plt.close(fig)
    return fig, stats_df

def plot_scaling(T_obs, T_null, p, title="", xlabel="Mismatch score (KS D)", bins=40):
    """
    scaling.py の permutation_test_cells() 出力を可視化する
      - T_null: permutationで得た帰無分布 (1D array)
      - T_obs: 観測統計量
      - p: 片側p値（T_null >= T_obs）
    """
    T_null = np.asarray(T_null, dtype=float)
    T_null = T_null[np.isfinite(T_null)]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    if T_null.size:
        ax.hist(T_null, bins=bins, density=True, alpha=0.5)
        # 参考: 95%区間
        lo, hi = np.percentile(T_null, [2.5, 97.5])
        ax.axvline(lo, linewidth=1.5, linestyle=":")
        ax.axvline(hi, linewidth=1.5, linestyle=":")

    if np.isfinite(T_obs):
        ax.axvline(T_obs, linewidth=2.0)

    # タイトル・注釈
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    txt = f"p={p:.3g}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig
