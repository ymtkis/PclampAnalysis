import numpy as np

def ks_d_stat(x, y, grid_n=2000):
    """
    KS統計量 D = max |F_x(t) - F_y(t)|
    x, y: 1D arrays
    """
    x = np.asarray(x); y = np.asarray(y)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 5 or len(y) < 5:
        return np.nan

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan

    grid = np.linspace(lo, hi, grid_n)
    Fx = np.searchsorted(np.sort(x), grid, side="right") / len(x)
    Fy = np.searchsorted(np.sort(y), grid, side="right") / len(y)
    return float(np.max(np.abs(Fx - Fy)))


def scaling_factor_median(ref, target, eps=1e-12):
    """
    target * alpha ≈ ref となる alpha を median 比で決める
    """
    ref = np.asarray(ref); target = np.asarray(target)
    ref = ref[np.isfinite(ref)]
    target = target[np.isfinite(target)]
    if len(ref) < 5 or len(target) < 5:
        return np.nan
    m_ref = np.median(ref)
    m_tgt = np.median(target)
    if np.abs(m_tgt) < eps:
        return np.nan
    return float(m_ref / m_tgt)


def pairwise_scaled_residual(ref_cell, tgt_cell, metric="ks"):
    """
    ref_cell: reference events
    tgt_cell: target events to be scaled
    """
    alpha = scaling_factor_median(ref_cell, tgt_cell)
    if not np.isfinite(alpha):
        return np.nan
    scaled = np.asarray(tgt_cell) * alpha

    if metric == "ks":
        return ks_d_stat(ref_cell, scaled)
    else:
        raise ValueError("metric not supported")


def condition_mismatch_score(cells_A, cells_B, metric="ks"):
    """
    A各cellをreferenceにして、B全cellとのscaled residualを平均 → それをAで平均
    """
    scores = []
    for a in cells_A:
        ds = []
        for b in cells_B:
            d = pairwise_scaled_residual(a, b, metric=metric)
            if np.isfinite(d):
                ds.append(d)
        if len(ds) > 0:
            scores.append(np.mean(ds))
    if len(scores) == 0:
        return np.nan
    return float(np.mean(scores))


def permutation_test_cells(cells_A, cells_B, n_perm=2000, seed=0, metric="ks"):
    """
    cellラベルをシャッフルして帰無分布を作る
    """
    rng = np.random.default_rng(seed)

    # 観測統計量
    T_obs = condition_mismatch_score(cells_A, cells_B, metric=metric)

    # 全cellをまとめてラベルperm
    all_cells = list(cells_A) + list(cells_B)
    nA = len(cells_A)

    T_null = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        idx = rng.permutation(len(all_cells))
        A_perm = [all_cells[i] for i in idx[:nA]]
        B_perm = [all_cells[i] for i in idx[nA:]]
        T_null[k] = condition_mismatch_score(A_perm, B_perm, metric=metric)

    # 片側（「観測のほうがズレが大きい」）p値
    valid = np.isfinite(T_null)
    Tn = T_null[valid]
    if len(Tn) == 0 or not np.isfinite(T_obs):
        return T_obs, T_null, np.nan

    p = (np.sum(Tn >= T_obs) + 1) / (len(Tn) + 1)  # +1補正
    return T_obs, T_null, float(p)
