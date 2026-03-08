import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats


def statistics(values_all_conditions, method="mannwhitneyu", correction="bonferroni"):
    records = []
    conditions = list(values_all_conditions.keys())

    for c1, c2 in itertools.combinations(conditions, 2):
        v1 = np.asarray(values_all_conditions[c1], dtype=float)
        v2 = np.asarray(values_all_conditions[c2], dtype=float)

        if v1.size == 0 or v2.size == 0:
            records.append(
                {
                    "Condition A": c1,
                    "Condition B": c2,
                    "Test": method,
                    "Statistic": np.nan,
                    "p-value": np.nan,
                    "Note": "skipped (empty group)",
                }
            )
            continue

        if method == "mannwhitneyu":
            stat, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            name = "Mann–Whitney U"
        elif method == "ttest":
            stat, p = stats.ttest_ind(v1, v2, equal_var=True)
            name = "t-test"
        elif method == "welch":
            stat, p = stats.ttest_ind(v1, v2, equal_var=False)
            name = "Welch t-test"
        else:
            raise ValueError(method)

        records.append(
            {
                "Condition A": c1,
                "Condition B": c2,
                "Test": name,
                "Statistic": stat,
                "p-value": p,
            }
        )

    df = pd.DataFrame(records)

    if df.empty or df["p-value"].dropna().empty:
        df["p-adjusted"] = np.nan
        df["Reject Null"] = False
        return df

    if correction is not None:
        from statsmodels.stats.multitest import multipletests

        pvals = df["p-value"].astype(float).values
        valid = np.isfinite(pvals)
        p_adj = np.full_like(pvals, np.nan, dtype=float)
        reject = np.full_like(pvals, False, dtype=bool)

        r, pa, _, _ = multipletests(pvals[valid], method=correction)
        p_adj[valid] = pa
        reject[valid] = r

        df["p-adjusted"] = p_adj
        df["Reject Null"] = reject
    else:
        df["p-adjusted"] = df["p-value"]
        df["Reject Null"] = df["p-value"] < 0.05

    return df