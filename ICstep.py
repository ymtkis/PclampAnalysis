import glob
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
from scipy.stats import sem
import scipy.stats as stats
import itertools
from collections import defaultdict
import math
import os


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



# Spike metrics
def compute_onset_offset(t, v, peak_idx, dt, dvdt_onset_thr=30.0, window_ms=5.0):

    # compute dV/dt in mV/ms
    window = 4
    diff_vec = np.zeros_like(v)
    # difference between current and 4 samples before
    diff_vec[window:] = (v[window:] - v[:-window]) / (window * dt) / 1000.0
    dvdt = diff_vec
    Vpeak = v[peak_idx]
    v_diff = 10.0

    # Onset: search backward from peak
    onset_idx = None
    for i in range(peak_idx-200, peak_idx, 1):
        if v[i] < Vpeak - v_diff and dvdt[i] >= dvdt_onset_thr:
            onset_idx = i
            break
    # Offset: search forward where v returns to onset voltage
    offset_idx = None
    if onset_idx is not None:
        target_v = v[onset_idx]
        win_samples = int((window_ms/1000.0) / dt)
        max_search = min(len(v), peak_idx + win_samples)
        for j in range(peak_idx+1, max_search):
            if v[j] <= target_v:
                offset_idx = j
                break

    return onset_idx, offset_idx



def compute_ahp(v, offset_idx, next_idx, dt):

    end = next_idx if next_idx is not None else len(v)
    segment = v[offset_idx:end]
    ahp_rel = np.argmin(segment)
    ahp_idx = offset_idx + ahp_rel
    ahp_amp = (v[ahp_idx] - v[offset_idx])
    ahp_time = ahp_rel * dt * 1000.0

    return ahp_amp, ahp_time



# For IC step
def extract_holding_currents(abf_path, step_start=0.18, step_duration=1.0, command_channel=1):

    abf = pyabf.ABF(abf_path)
    fs = abf.dataRate
    n_sweeps = abf.sweepCount
    start_idx = int(step_start * fs)
    end_idx = int((step_start + step_duration) * fs)

    holding_currents = []

    for sweep in range(n_sweeps):
        abf.setSweep(sweep, channel=command_channel)
        command_segment = abf.sweepY[start_idx:end_idx]
        mean_current = np.mean(command_segment)
        rounded_current = round(mean_current / 20) * 20 / 5
        holding_currents.append(rounded_current)

    return holding_currents



def ICstep_detect_events(sweep_data, holding_currents, fs, sweep_start, sweep_duration):

    event_counts = []
    event_times_all = []

    amp_list = []
    amp_onset_list = []
    half_duration_list = []
    rise_slope_list = []
    fall_slope_list = []

    sweep_start_idx = int(sweep_start * fs)
    sweep_end_idx = int((sweep_start+sweep_duration) * fs)
    t = np.arange(sweep_data.shape[1]) / fs
    dt = 1 / fs

    # Vm at 0 pA injection    
    zero_idx = holding_currents.index(0.0)
    rmp = np.mean(sweep_data[zero_idx, sweep_start_idx:sweep_end_idx])

    for sweep in sweep_data:

        segment = sweep[sweep_start_idx:sweep_end_idx]
        filtered = bandpass_filter(segment, fs, lowcut=0.1, highcut=1000.0)
        baseline = bandpass_filter(segment, fs, lowcut=2000.0, highcut=4000.0)
        delta = filtered-baseline
        peaks, _ = find_peaks(delta, height=40)
        spikes = [(sweep_start_idx+i) for i in peaks if (sweep_start_idx+i)/fs>=0.185]
        event_counts.append(len(spikes))
        times = [i/fs for i in spikes]
        event_times_all.append(times)

        # compute per-spike metrics        
        step_start = 0.18
        step_duration = 1.0
        end_idx = int((step_start + step_duration) * fs)
        
        if spikes:
            for i, peak_idx in enumerate(spikes):

                onset_idx, offset_idx = compute_onset_offset(t, sweep, peak_idx, dt)
                if onset_idx is None:
                    continue
                amp = (sweep[peak_idx] - rmp)
                amp_onset = (sweep[peak_idx] - sweep[onset_idx])
                half_dur = (t[peak_idx] - t[onset_idx]) * 1000.0
                rise_slope = (sweep[peak_idx] - sweep[onset_idx]) / ((peak_idx - onset_idx) * dt) / 1000.0
                fall_slope = (sweep[offset_idx] - sweep[peak_idx]) / ((offset_idx - peak_idx) * dt) / 1000.0

                amp_list.append(amp)
                amp_onset_list.append(amp_onset)
                half_duration_list.append(half_dur)
                rise_slope_list.append(rise_slope)
                fall_slope_list.append(fall_slope)

                #next_idx = spikes[i+1] if i+1 < len(spikes) else None
                #ahp_amp, ahp_time = compute_ahp(sweep, offset_idx, next_idx, dt)

    spike_metrics = {'Peak amplitude (mV)':amp_list, 'Peak amplitude from onset (mV)':amp_onset_list, 'Half duration (ms)':half_duration_list, 'Rise slope (mV/ms)':rise_slope_list, 'Fall slope (mV/ms)':fall_slope_list}

    return event_counts, event_times_all, spike_metrics



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
        filtered = bandpass_filter(segment, fs, lowcut=0.1, highcut=1000.0)
        baseline = bandpass_filter(segment, fs, lowcut=2000.0, highcut=4000.0)
        offset = i * 0

        event_times = event_times_all[i]
        event_times_in_window = [t for t in event_times if (sweep_start - window_padding) <= t <= (sweep_start + sweep_duration + window_padding)]
        event_times_rel = [t - sweep_start + 0.0 for t in event_times_in_window]

        event_indices = [int((t - (sweep_start - window_padding)) * fs) for t in event_times_in_window]
        event_values = [filtered[idx] + offset + 2 for idx in event_indices]

        ax.plot(time, filtered + offset, color='black', linewidth=0.4)
        #ax.plot(time, baseline + offset, color='gray', linewidth=0.4)
        ax.plot(event_times_rel, event_values, 'v', color='red', markersize=2, label='event' if i == 0 else "")

        #ax.text(-window_padding - 0.05, offset, f"{holding_currents[i]} pA", va='center', ha='right', fontsize=7)

    #ax.set_yticks([])
    #ax.set_yticklabels([])
    ax.set_xlabel("Time (s)")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)    

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



def f_i_plot(event_counts_all_conditions, holding_currents):

    fig, ax = plt.subplots(figsize=(6, 4))

    x_val = list(holding_currents.values())[0]

    for condition, data in event_counts_all_conditions.items():
    
        y_val = data.mean(axis=0)
        y_err = data.apply(sem, axis=0)
        ax.errorbar(x_val, y_val, yerr=y_err, fmt='-o', capsize=5, label=condition)

    ax.set_xlabel('I_command (pA)')
    ax.set_ylabel('Frequency (Hz)')
    ax.legend()

    #stats_df = statistics(values_all_conditions, method='mannwhitneyu', correction='bonferroni')

    fig.tight_layout()
    plt.close(fig)

    return fig



def metrics_plot(spike_metrics_all_conditions):

    conditions = list(spike_metrics_all_conditions.keys())
    metrics_names = spike_metrics_all_conditions[conditions[0]].index.tolist()
    n_conditions = len(conditions)
    x_positions = np.arange(n_conditions)
    figs = {}
    stats_dfs_list = []

    ylims = [[0, 120], [0, 72], [0, 1.8], [0, 144], [0, 72]]

    for k, metric in enumerate(metrics_names):

        fig, ax = plt.subplots(figsize=(n_conditions, 4))
        means, errs = [], []
        values_all_conditions = {}

        for xi, condition in zip(x_positions, conditions):

            values = np.abs(spike_metrics_all_conditions[condition].loc[metric].values)
            jitter = np.random.uniform(-0.1, 0.1, size=len(values))
            ax.scatter(xi + jitter, values, alpha=0.5)

            means.append(values.mean())
            errs.append(sem(values))
            values_all_conditions[condition] = values

        ax.errorbar(x_positions, means, yerr=errs, fmt='_', markersize=20, capsize=4, linestyle='None', color='k')

        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_xlim([x_positions[0] - 0.5, x_positions[-1] + 0.5])
        ax.set_ylim(ylims[k])
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_yticks(np.arange(y_min, y_max * 1.1, y_range / 3))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)        

        stats_df = statistics(values_all_conditions, method='mannwhitneyu', correction='bonferroni')
        stats_df.index.name = metric

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
            ax.plot([x_start, x_start], [y_line, y_line - y_range / 30], color='k', linewidth=1.0)
            ax.plot([x_end,   x_end],   [y_line, y_line - y_range / 30], color='k', linewidth=1.0)

            ax.text(x_center, y_line + 0.01, f"p={pval:.3f}", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        figs[metric] = fig
        stats_dfs_list.append(stats_df)

    stats_dfs = pd.concat(stats_dfs_list, keys=metrics_names).droplevel(level=1)

    return figs, stats_dfs



def input_resistance(sweep_data_ri_list, fs, baseline_start=2.20, baseline_end=3.00, holding_start=0.20, holding_end=1.00, max_holding_dev=3.0):

    ri_list = []

    baseline_start_idx = int(baseline_start * fs)
    baseline_end_idx   = int(baseline_end * fs)
    holding_start_idx  = int(holding_start * fs)
    holding_end_idx    = int(holding_end * fs)

    for sweep_data_ri in sweep_data_ri_list:

        baseline_segment = sweep_data_ri[baseline_start_idx:baseline_end_idx]
        holding_segment  = sweep_data_ri[holding_start_idx:holding_end_idx]

        holding_dev = holding_segment.max() - holding_segment.min()
        if holding_dev >= max_holding_dev:
            continue

        baseline_mean = np.mean(baseline_segment)
        holding_mean  = np.mean(holding_segment)

        ri_val = (holding_mean - baseline_mean) * 10**3 / (-60.0)
        if ri_val > 70.0:
            ri_list.append(ri_val)

    ri = np.mean(ri_list)

    return ri



def ri_plot(ri_all_conditions):

    conditions = list(ri_all_conditions.keys())
    n_conditions = len(conditions)
    x_positions = np.arange(n_conditions)

    fig, ax = plt.subplots(figsize=(n_conditions, 4))
    means, errs = [], []
    values_all_conditions = {}

    y_lim = [0, 15]
    for xi, condition in zip(x_positions, conditions):

        values = [val for val in ri_all_conditions[condition].loc[:, 'ri'] if not (isinstance(val, float) and math.isnan(val))]
        jitter = np.random.uniform(-0.1, 0.1, size=len(values))
        ax.scatter(xi + jitter, np.abs(values), alpha=0.5)

        means.append(np.mean(values))
        errs.append(stats.sem(values))
        values_all_conditions[condition] = values
    
    ax.errorbar(x_positions, np.abs(means), yerr=errs, fmt='_', markersize=20, capsize=4, linestyle='None', color='k')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_ylabel("Input resistance (MΩ)")
    ax.set_xlim([x_positions[0] - 0.5, x_positions[-1] + 0.5])
    ax.set_ylim([0, 180])
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_yticks(np.arange(0, y_max * 1.1, y_range / 3))
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
        ax.plot([x_start, x_start], [y_line, y_line - y_range / 30], color='k', linewidth=1.0)
        ax.plot([x_end,   x_end],   [y_line, y_line - y_range / 30], color='k', linewidth=1.0)

        ax.text(x_center, y_line + 0.01, f"p={pval:.3f}", ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    plt.close(fig)

    return fig, stats_df


base_path = f"/mnt/d/Q_project/PclampAnalysis"
trial_path = f"{base_path}/Data/Trial"
conditions = ["Ctrl", "QIH(48h)"]
protocol = "ICstep"

event_counts_all_conditions = {}
spike_metrics_all_conditions = {}
ri_all_conditions = {}
columns = {'I_command (pA)': list(range(-100, 201, 20))}

for condition in conditions:

    df = pd.read_excel(f'{base_path}/Experiment.xlsx', sheet_name=condition)
    mouse_ids = df['id'].tolist()
    slice_nums =  df['slice'].tolist()

    event_counts_summary = {}
    spike_metrics_summary = {}
    ri_summary = {}

    for mouse_id, slice_num in zip(mouse_ids, slice_nums):

        replicates = ['1', '2', '3']
        event_counts_list = []
        spike_metrics_list = []
        sweep_data_ri_list = []

        missing_file = False
        for replicate in replicates:
            abf_path = f"{trial_path}/{condition}/{protocol}/Q-OPN4_{mouse_id}_{condition}_{slice_num}_{protocol}_{replicate}.abf"
            if not os.path.exists(abf_path):
                missing_file = True
                break
        if missing_file:
            continue

        for replicate in replicates:

            abf_path = f"{trial_path}/{condition}/{protocol}/Q-OPN4_{mouse_id}_{condition}_{slice_num}_{protocol}_{replicate}.abf"            

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
            sweep_data_ri = np.zeros((1, n_samples))
            holding_currents = extract_holding_currents(abf_path)
            zero_idx = holding_currents.index(0.0)
            
            for i in range(n_sweeps):
                abf.setSweep(i)
                sweep_data[i, :] = abf.sweepY / scale_factor

                if i == 2:
                    sweep_data_ri = abf.sweepY / scale_factor
                    sweep_data_ri_list.append(sweep_data_ri)

            baseline = sweep_data[zero_idx, :].mean()
            sweep_data = sweep_data - baseline

            event_counts, event_times_all, spike_metrics = ICstep_detect_events(sweep_data, holding_currents, fs, sweep_start=0.18, sweep_duration=1.0)
            event_counts_list.append(event_counts)
            spike_metrics_list.append(spike_metrics)

            fig_event_annotation = ICstep_event_annotation(sweep_data, event_times_all, holding_currents, fs, sweep_start=0.18, sweep_duration=1.0)
            fig_event_annotation.savefig(f'{base_path}/Results/{protocol}/event_annotation/Q-OPN4_{mouse_id}_{condition}_{slice_num}_{protocol}_{replicate}.png', dpi=1000)

        average_event_counts = [round(np.mean(vals), 3) for vals in zip(*event_counts_list)]
        event_counts_summary[f'Q-OPN4_{mouse_id}_{slice_num}'] = average_event_counts
        
        spike_metrics_reps = defaultdict(list)
        for rep in spike_metrics_list:
            for metrics, vals in rep.items():
                spike_metrics_reps[metrics].append(vals)

        average_spike_metrics = defaultdict(list)
        for metrics, vals in spike_metrics_reps.items():
            average_spike_metrics[metrics] = list(itertools.chain.from_iterable(spike_metrics_reps[metrics]))
            average_spike_metrics[metrics] = sum(average_spike_metrics[metrics]) / len(average_spike_metrics[metrics])
        
        average_spike_metrics = dict(average_spike_metrics)
        spike_metrics_summary[f'Q-OPN4_{mouse_id}_{slice_num}'] = average_spike_metrics

        ri = input_resistance(sweep_data_ri_list, fs)
        ri_summary[f'Q-OPN4_{mouse_id}_{slice_num}'] = ri

    event_counts_summary_df = pd.DataFrame.from_dict(event_counts_summary, orient='index', columns=columns['I_command (pA)'])
    spike_metrics_summary_df = pd.DataFrame.from_dict(spike_metrics_summary)
    ri_summary_df = pd.DataFrame.from_dict(ri_summary, orient='index', columns=['ri'])

    event_counts_save_path = f"{base_path}/Results/{protocol}/ICstep_{condition}.csv"
    event_counts_summary_df.to_csv(event_counts_save_path)

    spike_metrics_save_path = f"{base_path}/Results/{protocol}/ICstep_spike_metrics_{condition}.csv"
    spike_metrics_summary_df.to_csv(spike_metrics_save_path)

    ri_save_path = f"{base_path}/Results/{protocol}/Input_resistance_{condition}.csv"
    ri_summary_df.to_csv(ri_save_path)

    event_counts_all_conditions[condition] = event_counts_summary_df
    spike_metrics_all_conditions[condition] = spike_metrics_summary_df
    ri_all_conditions[condition] = ri_summary_df

fig_f_i_plot = f_i_plot(event_counts_all_conditions, columns)
figs_spike_metrics, stats_dfs_spike_metrics = metrics_plot(spike_metrics_all_conditions)
file_names = ['Peak_amplitude', 'Peak_amplitude_from_onset', 'Half_duration', 'Rise_slope', 'Fall_slope']
ri_fig, ri_stats = ri_plot(ri_all_conditions)

save_path = f'{base_path}/Results/{protocol}'

fig_f_i_plot.savefig(f'{save_path}/F-I_curve.png', dpi=1000)
for i, fig in enumerate(figs_spike_metrics.values()):
    fig.savefig(f'{save_path}/{file_names[i]}.png', dpi=1000) 
ri_fig.savefig(f'{save_path}/Input_resistance.png', dpi=1000)

stats_dfs_spike_metrics.to_csv(f'{save_path}/spike_metrics_stats.csv')
ri_stats.to_csv(f'{save_path}/Input_resistance_stats.csv')    