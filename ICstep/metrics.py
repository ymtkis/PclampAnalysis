import numpy as np

def compute_rheobase(event_counts, holding_currents):
    for n, I in zip(event_counts, holding_currents):
        if n > 0:
            return I
    return np.nan


def input_resistance(
    sweep_data_list,
    fs,
    holding_currents,
    baseline_start=0.02,
    baseline_end=0.17,
    holding_start=0.50,
    holding_end=1.00,
    negative_only=True,
):
    b0, b1 = int(baseline_start * fs), int(baseline_end * fs)
    h0, h1 = int(holding_start * fs), int(holding_end * fs)

    dV, dI = [], []
    print(holding_currents)
    for sweep, I in zip(sweep_data_list, holding_currents):
        if negative_only and I >= 0:
            continue
        dV.append(np.mean(sweep[h0:h1]) - np.mean(sweep[b0:b1]))
        dI.append(I)

    if len(dI) < 2:
        return np.nan

    slope, _ = np.polyfit(dI, dV, 1)
    return slope * 1e3
