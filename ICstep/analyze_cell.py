import os
import numpy as np
import pyabf
from .spike_detection import ICstep_detect_events, extract_holding_currents
from .metrics import compute_rheobase
from .plotting import ICstep_event_annotation

def analyze_cell(
    mouse_id,
    slice_num,
    condition,
    data_path,
    protocol,
    prefix,
    replicates,
    CFG,
):
    cell_id = f"{prefix}_{mouse_id}_{slice_num}"

    low, high = [], []
    low_m, high_m = [], []
    low_r, high_r = [], []
    sweep_ri, I_ri = [], []
    fs = None

    for rep in replicates:
        abf_path = f"{data_path}/{condition}/{prefix}_{mouse_id}_{condition}_{slice_num}_{protocol}_{rep}.abf"
        if not os.path.exists(abf_path):
            continue

        is_low = os.path.getsize(abf_path) / 1024 > 6500
        abf = pyabf.ABF(abf_path)
        fs = fs or abf.dataRate
        scales = [20, 5] if abf.adcUnits[0] == "pA" else [1, 1]

        holding = extract_holding_currents(abf_path)
        sweeps = np.zeros((abf.sweepCount, abf.sweepPointCount), dtype=float)
        for i in range(abf.sweepCount):
            abf.setSweep(i)
            sweeps[i, :] = abf.sweepY / scales[0]

        for s, I in zip(sweeps, holding):
            sweep_ri.append(s)
            I_ri.append(I/scales[1])

        ec, et, sm = ICstep_detect_events(sweeps, holding, fs, 0.18, 1.0)
        rheo = compute_rheobase(ec, holding)

        if is_low:
            low.append(ec); low_m.append(sm); low_r.append(rheo)
        else:
            high.append(ec); high_m.append(sm); high_r.append(rheo)

    return {
        "cell_id": cell_id,
        "event_counts_low": low,
        "event_counts_high": high,
        "spike_metrics_low": low_m,
        "spike_metrics_high": high_m,
        "rheobase_low": low_r,
        "rheobase_high": high_r,
        "sweep_data_ri": sweep_ri,
        "holding_currents_ri": I_ri,
        "fs": fs,
    }
