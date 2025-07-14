import glob
import os
import pyabf
import quantities as pq
import neo
import elephant
from elephant.spike_train_generation import peak_detection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats
import itertools



# Filtering
def highpass_filter(data, fs, cutoff, order=2):

    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)

    return filtered_data



def lowpass_filter(data, fs, cutoff, order=2):

    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)

    return filtered_data



def bandpass_filter(data, fs, lowcut, highcut, order=2):

    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band', analog=False)
    filtered_data = filtfilt(b, a, data)

    return filtered_data



# For mEPSP
def mEPSP_detect_events(sweep_data, fs, sweep_start, sweep_duration):

    event_counts = []
    event_times_all = []
    event_amps = []
    filtered_sweep_data = []

    total_sweeps = len(sweep_data)
    exclude_proportion = 0.3
    amp_threshold = 0.1
    rise_time_threshold = 5.0
    drift_threshold = 5.0
    dt_ms = 1000.0 / fs

    sweep_start_idx = int(sweep_start * fs)
    sweep_end_idx = int((sweep_start + sweep_duration) * fs)
    sd_window_idx = int(1.0 * fs)

    excluded = 0
    
    for sweep in sweep_data:

        segment = sweep[sweep_start_idx:sweep_end_idx]
        baseline = gaussian_filter1d(bandpass_filter(segment, fs, lowcut=1000.0, highcut=2000.0), sigma=100)
        filtered = bandpass_filter(segment, fs, lowcut=4.0, highcut=100.0)

        deviation = np.max(np.abs(segment)) - np.min(np.abs(segment))

        if deviation > drift_threshold:
            excluded += 1
        else:      
            filtered_sweep_data.append(segment)

            # peak detection
            peaks, props = find_peaks(filtered)
            dvdt = np.gradient(filtered, dt_ms)       
            valid_events = []
            valid_amps = []

            for idx in peaks:

                if sd_window_idx - idx > 0:
                    sd_start = 0
                    sd_end = 2 * sd_window_idx
                elif sd_window_idx + idx > sweep_end_idx:
                    sd_start = sweep_end_idx - 2 * sd_window_idx
                    sd_end = sweep_end_idx
                else:
                    sd_start = idx - sd_window_idx
                    sd_end = idx + sd_window_idx

                local_sd = np.std(filtered[sd_start:sd_end])       

                if (np.all(dvdt[idx-int(rise_time_threshold/dt_ms):idx] > 0) and 
                filtered[idx] > amp_threshold and
                np.abs(filtered[idx]) > 3 * local_sd):

                    t_event = idx / fs    
                    valid_events.append(t_event)
                    valid_amps.append(filtered[idx])

            event_counts.append(len(valid_events))
            event_times_all.append(valid_events)
            event_amps.append(valid_amps)
    
    # if too many exclusions, skip entire run
    if excluded > exclude_proportion * total_sweeps:
        return None, None, None, None
    filtered_sweep_data = np.vstack(filtered_sweep_data)
    event_amps = [amps for valid_amps in event_amps for amps in valid_amps]

    return filtered_sweep_data, event_counts, event_times_all, event_amps




def mEPSP_event_annotation(sweep_data, event_times_all, fs, sweep_duration):

    n_samples = int(fs * sweep_duration)
    time = np.arange(n_samples) / fs
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(event_times_all) * 0.4)))

    for i in range(len(sweep_data)):
        baseline = gaussian_filter1d(bandpass_filter(sweep_data[i], fs, lowcut=1000.0, highcut=2000.0), sigma=100)
        filtered = bandpass_filter(sweep_data[i], fs, lowcut=4.0, highcut=1000.0)
        offset = i + 0.1

        event_times = event_times_all[i]

        event_indices = [int(t * fs) for t in event_times]
        event_values = [filtered[idx] + offset for idx in event_indices]

        ax.plot(time, filtered + offset, color='black', linewidth=0.1)
        ax.plot(time, baseline + offset, color='gray', linewidth=0.4)
        ax.plot(event_times, event_values, 'v', color='red', markersize=1, label='event' if i == 0 else "")

    ax.set_yticklabels([])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel('')

    fig.tight_layout()
    plt.close(fig)

    return fig



def statistics(values_all_conditions, method='mannwhitneyu', correction='bonferroni'):

    conditions = list(values_all_conditions.keys())

    # t test
    records = []
    for cond1, cond2 in itertools.combinations(conditions, 2):
        vals1 = values_all_conditions[cond1]
        vals2 = values_all_conditions[cond2]

        if method.lower() == "ttest":
            # Student’s t-test
            t_stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=equal_var)
            test_name = "t-test" if equal_var else "Welch t-test"
            stat_value = t_stat

        elif method.lower() == "welch":
            # Welch's t-test
            t_stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
            test_name = "Welch t-test"
            stat_value = t_stat

        elif method.lower() == "mannwhitneyu":
            # Mann–Whitney U test
            u_stat, p_val = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
            test_name = "Mann–Whitney U"
            stat_value = u_stat

        else:
            raise ValueError(f"Unsupported method: {method}")

        records.append({
            "Condition A": cond1,
            "Condition B": cond2,
            "Test": test_name,
            "Statistic": stat_value,
            "p-value": p_val
        })

    stats_df = pd.DataFrame(records)

    # Correction for multiple comparison
    if correction is not None:
        from statsmodels.stats.multitest import multipletests
        pvals = stats_df["p-value"].values
        reject, p_adj, _, _ = multipletests(pvals, method=correction)
        stats_df["p-adjusted"] = p_adj
        stats_df["Reject Null"] = reject
    else:
        stats_df["p-adjusted"] = stats_df["p-value"]
        stats_df["Reject Null"] = stats_df["p-value"] < 0.05

    return stats_df



def mEPSP_cdf(event_amps_all_condition, n_bins=50):

    fig, ax = plt.subplots(figsize=(6, 4))
    values_all_conditions = {}

    for condition, event_amps_summary in event_amps_all_condition.items():
        values = []

        for neuron, event_amps in event_amps_summary.items():
            values.extend(event_amps)

        counts, bin_edges = np.histogram(values, bins=n_bins)
        cum_counts = np.cumsum(counts)
        cum_fraction = cum_counts / cum_counts[-1]

        ax.plot(bin_edges[1:], cum_fraction, drawstyle='steps-post', label=condition)

        values_all_conditions[condition] = values

    ax.set_xlabel('Amplitude(mV)')
    ax.set_ylabel('Fraction')
    ax.legend()

    stats_df = statistics(values_all_conditions, method='mannwhitneyu', correction='bonferroni')

    fig.tight_layout()
    plt.close(fig)

    return fig, stats_df



def plot_average_amp(average_amps_all_conditions):

    conditions = list(average_amps_all_conditions.keys())
    n_conditions = len(conditions)
    x_positions = np.arange(n_conditions)

    fig, ax = plt.subplots(figsize=(n_conditions, 4))
    means, errs = [], []
    values_all_conditions = {}

    for xi, condition in zip(x_positions, conditions):
        values = [val for val in average_amps_all_conditions[condition].values()]
        jitter = np.random.uniform(-0.1, 0.1, size=len(values))
        ax.scatter(xi + jitter, values, alpha=0.5)

        means.append(np.mean(values))
        errs.append(stats.sem(values))
        values_all_conditions[condition] = values

    ax.errorbar(x_positions, means, yerr=errs, fmt='_', markersize=20, capsize=4, linestyle='None', color='k')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel("Mean amplitude (mV)")
    ax.set_xlim([x_positions[0] - 0.5, x_positions[-1] + 0.5])
    ax.set_ylim([0, 0.9])
    ax.set_yticks(np.arange(0, 1, 0.3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    stats_df = statistics(values_all_conditions, method='mannwhitneyu', correction='bonferroni')

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    h_gap = 0.2 

    for i, row in stats_df.iterrows():
        cond1 = row["Condition A"]
        cond2 = row["Condition B"]
        pval  = row["p-adjusted"] 

        xi = conditions.index(cond1)
        xj = conditions.index(cond2)

        y_line = y_max + i * h_gap
        x_center = (xi + xj) / 2.0
        line_len = abs(xj - xi) * 0.8
        x_start = x_center - line_len / 2.0
        x_end   = x_center + line_len / 2.0

        ax.plot([x_start, x_end], [y_line, y_line], color='k', linewidth=1.0)
        ax.plot([x_start, x_start], [y_line, y_line - 0.03], color='k', linewidth=1.0)
        ax.plot([x_end,   x_end],   [y_line, y_line - 0.03], color='k', linewidth=1.0)

        ax.text(x_center, y_line + 0.01, f"p={pval:.3f}", ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    plt.close(fig)

    return fig, stats_df



def plot_freq(event_freq_all_conditions):

    conditions = list(event_freq_all_conditions.keys())
    n_conditions = len(conditions)
    x_positions = np.arange(n_conditions)

    fig, ax = plt.subplots(figsize=(n_conditions, 4))
    means, errs = [], []
    values_all_conditions = {}

    for xi, condition in zip(x_positions, conditions):
        values = [val for val in event_freq_all_conditions[condition].values()]
        jitter = np.random.uniform(-0.1, 0.1, size=len(values))
        ax.scatter(xi + jitter, values, alpha=0.5)

        means.append(np.mean(values))
        errs.append(stats.sem(values))
        values_all_conditions[condition] = values
    
    ax.errorbar(x_positions, means, yerr=errs, fmt='_', markersize=20, capsize=4, linestyle='None', color='k')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel("mEPSP event (Hz)")
    ax.set_xlim([x_positions[0] - 0.5, x_positions[-1] + 0.5])
    ax.set_ylim([0, 1.5])
    ax.set_yticks(np.arange(0, 1.6, 0.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    stats_df = statistics(values_all_conditions, method='mannwhitneyu', correction='bonferroni')

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    h_gap = 0.2 

    for i, row in stats_df.iterrows():
        cond1 = row["Condition A"]
        cond2 = row["Condition B"]
        pval  = row["p-adjusted"] 

        xi = conditions.index(cond1)
        xj = conditions.index(cond2)

        y_line = y_max + i * h_gap
        x_center = (xi + xj) / 2.0
        line_len = abs(xj - xi) * 0.8
        x_start = x_center - line_len / 2.0
        x_end   = x_center + line_len / 2.0

        ax.plot([x_start, x_end], [y_line, y_line], color='k', linewidth=1.0)
        ax.plot([x_start, x_start], [y_line, y_line - 0.05], color='k', linewidth=1.0)
        ax.plot([x_end,   x_end],   [y_line, y_line - 0.05], color='k', linewidth=1.0)

        ax.text(x_center, y_line + 0.01, f"p={pval:.3f}", ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    plt.close(fig)

    return fig, stats_df



base_path = f"/mnt/d/Q_project/PclampAnalysis"
trial_path = f"{base_path}/Data/Trial"
conditions = ["Ctrl", "QIH(48h)"]
protocol = "mEPSP"

event_freq_all_conditions = {}
event_amps_all_conditions = {}
average_amps_all_conditions = {}

for condition in conditions:

    df = pd.read_excel(f'{base_path}/Experiment.xlsx', sheet_name=condition)
    mouse_ids = df['id'].tolist()
    slice_nums =  df['slice'].tolist()

    event_freq_summary = {}
    event_amps_summary = {}
    average_amps_summary = {}

    for mouse_id, slice_num in zip(mouse_ids, slice_nums):

        abf_path = f"{trial_path}/{condition}/{protocol}/Q-OPN4_{mouse_id}_{condition}_{slice_num}_{protocol}.abf"
        
        if not os.path.exists(abf_path):
            continue

        abf = pyabf.ABF(abf_path)
        n_sweeps = abf.sweepCount
        n_samples = abf.sweepPointCount
        fs = abf.dataRate
        unit = abf.adcUnits[0]
        if unit == 'pA':
            scale_factor = 25
        else:
            scale_factor = 1
        sweep_data = np.zeros((n_sweeps, n_samples))

        for i in range(n_sweeps):
            abf.setSweep(i)
            sweep_data[i, :] = abf.sweepY / scale_factor

        sweep_data, event_counts, event_times_all, event_amps = mEPSP_detect_events(sweep_data, fs, sweep_start=0.4, sweep_duration=10.0)

        if sweep_data is not None:

            fig_event_annotation = mEPSP_event_annotation(sweep_data, event_times_all, fs, sweep_duration=10.0)
            save_path = f'{base_path}/Results/{protocol}/Q-OPN4_{mouse_id}_{condition}_{slice_num}_{protocol}.png'
            fig_event_annotation.savefig(save_path, dpi=1000)

            event_freq_summary[f'{mouse_id}_{slice_num}'] = sum(event_counts) / (10 * len(event_counts))
            event_amps_summary[f'{mouse_id}_{slice_num}'] = event_amps
            average_amps_summary[f'{mouse_id}_{slice_num}'] = round(np.mean(event_amps), 3)
            

    event_freq_all_conditions[condition] = event_freq_summary
    event_amps_all_conditions[condition] = event_amps_summary
    average_amps_all_conditions[condition] = average_amps_summary


cdf_fig, cdf_stats = mEPSP_cdf(event_amps_all_conditions)
average_amp_fig, average_amp_stats = plot_average_amp(average_amps_all_conditions)
freq_fig, freq_stats = plot_freq(event_freq_all_conditions)

save_path = f'{base_path}/Results/{protocol}'
cdf_fig.savefig(f'{save_path}/mEPSP_amp_distribution.png', dpi=1000)
average_amp_fig.savefig(f'{save_path}/mEPSP_average_amp.png', dpi=1000)
freq_fig.savefig(f'{save_path}/mEPSP_frequency.png', dpi=1000)

cdf_stats.to_csv(f'{save_path}/mEPSP_amp_distribution_stats.csv', index=False)
average_amp_stats.to_csv(f'{save_path}/mEPSP_average_amp_stats.csv', index=False)
freq_stats.to_csv(f'{save_path}/mEPSP_frequency_stats.csv')
        
    