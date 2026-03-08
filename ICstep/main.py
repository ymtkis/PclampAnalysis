# ICstep/main.py
import os
import pandas as pd
import numpy as np
from collections import defaultdict

from .config import CFG
from .analyze_cell import analyze_cell
from .metrics import input_resistance
from .plotting import f_i_plot, metrics_plot, rheobase_plot, ri_plot


def main():
    base_path = CFG["paths"]["base_path"]
    protocol = CFG["protocol"]["ic_step"]
    data_path = f'{base_path}/{CFG["paths"]["data_root"]}/{protocol}'

    conditions = CFG["conditions"]
    prefix = CFG["paths"]["prefix"]

    event_counts_all_conditions = {}
    spike_metrics_all_conditions = {}
    ri_all_conditions = {}
    rheobase_all_conditions = {}

    i_start = CFG["analysis"]["i_command_start"]
    i_end = CFG["analysis"]["i_command_end"]
    i_step = CFG["analysis"]["i_command_step"]
    currents = list(range(i_start, i_end + i_step, i_step))

    per_condition_outdir = f"{base_path}/Results/{protocol}"
    os.makedirs(per_condition_outdir, exist_ok=True)

    for condition in conditions:
        df = pd.read_excel(
            f'{base_path}/{CFG["paths"]["experiment_file"]}',
            sheet_name=condition,
        )

        ecs, sms, ris, rhs = {}, {}, {}, {}

        for mid, sl, ct in zip(df["id"], df["slice"], df["cell"]):
            if ct != "pyramidal":
                continue

            res = analyze_cell(
                mid,
                sl,
                condition,
                data_path,
                protocol,
                prefix,
                CFG["analysis"]["replicates"],
                CFG,
            )
            cid = res["cell_id"]

            total_low = sum(sum(c) for c in res["event_counts_low"])
            if (total_low > 0) and res["spike_metrics_low"]:
                metrics_reps = res["spike_metrics_low"]
                rheo_values = res["rheobase_low"]
                metrics_source = "low"
            elif res["spike_metrics_high"]:
                metrics_reps = res["spike_metrics_high"]
                rheo_values = res["rheobase_high"]
                metrics_source = "high"
            else:
                metrics_reps = []
                rheo_values = []
                metrics_source = "none"

            sm = defaultdict(list)
            for rep in metrics_reps:
                for k, v in rep.items():
                    sm[k].extend(v)

            sms[cid] = {k: (np.mean(v) if len(v) else np.nan) for k, v in sm.items()}
            sms[cid]["_source"] = metrics_source

            rhs[cid] = float(np.nanmean(rheo_values)) if rheo_values else np.nan

            ris[cid] = input_resistance(
                res["sweep_data_ri"],
                res["fs"],
                res["holding_currents_ri"],
            )

            if res["event_counts_low"]:
                ecs[cid] = [round(np.mean(v), 3) for v in zip(*res["event_counts_low"])]

        event_counts_df = pd.DataFrame.from_dict(ecs, orient="index", columns=currents)
        spike_metrics_df = pd.DataFrame.from_dict(sms)
        ri_df = pd.DataFrame.from_dict(ris, orient="index", columns=["ri"])
        rheobase_df = pd.DataFrame.from_dict(rhs, orient="index", columns=["rheobase_pA"])

        event_counts_df.to_csv(f"{per_condition_outdir}/ICstep_{condition}.csv")
        spike_metrics_df.to_csv(f"{per_condition_outdir}/ICstep_spike_metrics_{condition}.csv")
        ri_df.to_csv(f"{per_condition_outdir}/Input_resistance_{condition}.csv")
        rheobase_df.to_csv(f"{per_condition_outdir}/rheobase_{condition}.csv")

        event_counts_all_conditions[condition] = event_counts_df
        spike_metrics_all_conditions[condition] = spike_metrics_df
        ri_all_conditions[condition] = ri_df
        rheobase_all_conditions[condition] = rheobase_df

    results_root = f'{base_path}/{CFG["results"]["results_subdir"]}/{protocol}'
    os.makedirs(results_root, exist_ok=True)

    def _subset(d, keys):
        return {k: d[k] for k in keys if k in d}


    metric_file_names = CFG["plot"]["metrics"]
    sets = CFG["plot"]["sets"]

    for set_name, order in sets.items():

        group_dir = f"{results_root}/{set_name}"
        os.makedirs(group_dir, exist_ok=True)

        ec_sub = _subset(event_counts_all_conditions, order)
        sm_sub = _subset(spike_metrics_all_conditions, order)
        rheo_sub = _subset(rheobase_all_conditions, order)
        ri_sub = _subset(ri_all_conditions, order)

        # --- F-I ---
        fig_f_i = f_i_plot(ec_sub, currents, conditions_order=order, title=set_name)
        fig_f_i.savefig(f"{group_dir}/F-I_curve.svg", format="svg")

        # --- spike metrics ---
        figs_metrics, stats_metrics = metrics_plot(sm_sub)
        for fname, fig in zip(metric_file_names, figs_metrics.values()):
            fig.savefig(f"{group_dir}/{fname}.svg", format="svg")
        stats_metrics.to_csv(f"{group_dir}/spike_metrics_stats.csv")

        # --- rheobase ---
        fig_rheo, stats_rheo = rheobase_plot(rheo_sub)
        fig_rheo.savefig(f"{group_dir}/Rheobase.svg", format="svg")
        if stats_rheo is not None:
            stats_rheo.to_csv(f"{group_dir}/Rheobase_stats.csv")

        # --- input resistance ---
        fig_ri, stats_ri = ri_plot(ri_sub)
        fig_ri.savefig(f"{group_dir}/Input_resistance.svg", format="svg")
        if stats_ri is not None:
            stats_ri.to_csv(f"{group_dir}/Input_resistance_stats.csv")



    return {
        "event_counts_all_conditions": event_counts_all_conditions,
        "spike_metrics_all_conditions": spike_metrics_all_conditions,
        "ri_all_conditions": ri_all_conditions,
        "rheobase_all_conditions": rheobase_all_conditions,
        "currents": currents,
        "results_root": results_root,
        "per_condition_outdir": per_condition_outdir,
    }


if __name__ == "__main__":
    main()
