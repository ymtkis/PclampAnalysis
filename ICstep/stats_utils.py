import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats


def statistics(values_all_conditions, method="mannwhitneyu", correction="bonferroni", equal_var=True):
    conditions = list(values_all_conditions.keys())

    records = []
    for cond1, cond2 in itertools.combinations(conditions, 2):
        vals1 = np.asarray(values_all_conditions[cond1], dtype=float)
        vals2 = np.asarray(values_all_conditions[cond2], dtype=float)

        vals1 = vals1[~np.isnan(vals1)]
        vals2 = vals2[~np.isnan(vals2)]

        if method.lower() == "ttest":
            t_stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=equal_var)
            test_name = "t-test" if equal_var else "Welch t-test"
            stat_value = t_stat

        elif method.lower() == "welch":
            t_stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
            test_name = "Welch t-test"
            stat_value = t_stat

        elif method.lower() == "mannwhitneyu":
            u_stat, p_val = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
            test_name = "Mann–Whitney U"
            stat_value = u_stat

        else:
            raise ValueError(f"Unsupported method: {method}")

        records.append(
            {
                "Condition A": cond1,
                "Condition B": cond2,
                "Test": test_name,
                "Statistic": stat_value,
                "p-value": p_val,
            }
        )

    stats_df = pd.DataFrame(records)

    if len(stats_df) == 0:
        return stats_df

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
