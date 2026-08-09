"""qPHD: quantile-trimmed MST-based intrinsic dimension across scales.

Port of PHDimScale from PHDQ/scripts/phd_scale.py without CUDA deps.

Point cloud -> for each subsample size n: MST -> edge lengths -> trimmed
sums S(q); fit log S vs log n; slope s => d = alpha / (1 - s).

Three trimming modes (notation from the task):
  q_small    -- drop the q-fraction of the *shortest* edges
  q_large    -- drop the q-fraction of the *longest* edges
  q0.5_range -- drop q shortest, keep edges in quantile band [q, q+0.5]
"""
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

MODES = ["q_small", "q_large", "q0.5_range"]


def mst_edge_lengths(points):
    """points: (n, d) -> sorted (ascending) MST edge lengths, shape (n-1,)."""
    adj = squareform(pdist(points))
    mst = minimum_spanning_tree(adj)
    return np.sort(mst.data)


def trimmed_sum(lens_sorted, q, mode, alpha=1.0, p_range=0.5, discrete=False):
    """Sum of alpha-powers of MST edge lengths after quantile trimming.

    lens_sorted must be ascending.

    discrete=True reproduces the paper's floor()-based edge counting. That
    makes the effective trimmed fraction depend on m = n-1: at the smallest
    subsample floor() rounds q down by up to 0.5/m, so less is trimmed there,
    S is inflated at small n, and the log-log slope tilts. On a q grid finer
    than 0.1 this shows up as a saw-tooth in d_hat(q). The default
    (discrete=False) gives the boundary edge a fractional weight instead, so
    the trimmed fraction is exactly q at every n.
    """
    m = lens_sorted.size
    if m == 0:
        return 0.0

    if discrete:
        if mode == "q_small":
            kept = lens_sorted[int(np.floor(q * m)) :]
        elif mode == "q_large":
            k = int(np.floor(q * m))
            kept = lens_sorted[: m - k] if k > 0 else lens_sorted
        elif mode == "q0.5_range":
            k_min = int(np.floor(q * m))
            k_max = int(np.ceil(min(1.0, q + p_range) * m))
            kept = lens_sorted[k_min : min(m, k_max)]
        else:
            raise ValueError(f"unknown mode {mode}")
        return float((kept**alpha).sum())

    w = np.ones(m)

    def cut_low(frac):
        """Zero out the lowest `frac` fraction of edges, fractional boundary."""
        t = frac * m
        k = int(np.floor(t))
        w[:k] = 0.0
        if k < m:
            w[k] *= 1.0 - (t - k)

    def cut_high(frac):
        """Zero out the highest `frac` fraction of edges, fractional boundary."""
        t = frac * m
        k = int(np.floor(t))
        if k > 0:
            w[m - k :] = 0.0
        if m - k - 1 >= 0:
            w[m - k - 1] *= 1.0 - (t - k)

    if mode == "q_small":
        cut_low(q)
    elif mode == "q_large":
        cut_high(q)
    elif mode == "q0.5_range":
        cut_low(q)
        cut_high(max(0.0, 1.0 - min(1.0, q + p_range)))
    else:
        raise ValueError(f"unknown mode {mode}")

    return float(((lens_sorted**alpha) * w).sum())


def fit_loglog(xs, ys):
    """LS fit y = a + b*x on log-transformed inputs.

    Returns a dict with the fit and its goodness-of-fit diagnostics. The
    residual measures say how well the assumed power law S ~ n^b actually
    holds for this particular measurement: a clean power law means the
    d_hat it implies is meaningful, while large residuals mean the single
    slope is a bad summary and d_hat is not to be trusted.

    resid_rmse : root-mean-square residual in log space (scale-free)
    resid_max  : largest absolute residual in log space
    slope_se   : standard error of the slope (OLS)
    """
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    nan = {
        "slope": np.nan, "intercept": np.nan, "r2": np.nan,
        "resid_rmse": np.nan, "resid_max": np.nan, "slope_se": np.nan,
        "n_points": int(x.size),
    }
    if x.size < 2:
        return nan

    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    dof = x.size - 2
    sxx = float(np.sum((x - x.mean()) ** 2))
    slope_se = (
        float(np.sqrt(ss_res / dof / sxx)) if dof > 0 and sxx > 0 else np.nan
    )

    return {
        "slope": float(b),
        "intercept": float(a),
        "r2": np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot,
        "resid_rmse": float(np.sqrt(ss_res / x.size)),
        "resid_max": float(np.max(np.abs(resid))),
        "slope_se": slope_se,
        "n_points": int(x.size),
    }


def d_from_energy_slope(alpha, slope):
    """log S(n) ~ const + slope*log n, slope ~ 1 - alpha/d  =>  d = alpha/(1-slope)."""
    if not np.isfinite(slope) or abs(1.0 - slope) < 1e-12:
        return np.nan
    return float(alpha) / (1.0 - slope)


Q_MAX_BY_MODE = {"q_small": 0.9, "q_large": 0.9, "q0.5_range": 0.5}


def qphd(
    points,
    q_list=tuple(np.round(np.arange(0, 0.901, 0.05), 2)),
    modes=MODES,
    n_fraction_list=(0.2, 0.4, 0.6, 0.8, 1.0),
    alpha=1.0,
    p_range=0.5,
    replicates=10,
    replace=True,
    discrete=False,
    pool=None,
    rng=None,
):
    """Estimate d_hat(q) for each trimming mode.

    points: (N, dim) point cloud; N sets the subsample sizes n = f * N.
    pool: if given, the subsamples are drawn from this cloud instead of
        `points` (e.g. all tokens of the text, while n is still pegged to a
        fixed L). `points` is then used only to fix the n grid.
    q is capped per mode by Q_MAX_BY_MODE: the range mode only makes sense
    while the band [q, q+p_range] fits inside [0, 1].

    Returns tidy DataFrame: mode, q, slope, r2, d_hat, n_min, n_max.
    """
    rng = rng or np.random.default_rng()
    N = points.shape[0]
    src = points if pool is None else pool
    records = []
    n_values = sorted({int(f * N) for f in n_fraction_list if int(f * N) > 1})
    if not replace:
        n_values = [n for n in n_values if n <= src.shape[0]]

    for n in n_values:
        for _ in range(replicates):
            idx = rng.choice(src.shape[0], size=n, replace=replace)
            lens = mst_edge_lengths(src[idx])
            for mode in modes:
                for q in q_list:
                    if q > Q_MAX_BY_MODE.get(mode, 1.0):
                        continue
                    records.append(
                        {
                            "mode": mode,
                            "q": q,
                            "n": n,
                            "S": trimmed_sum(
                                lens, q, mode, alpha, p_range, discrete=discrete
                            ),
                        }
                    )
    df = pd.DataFrame(records)
    agg = df.groupby(["mode", "q", "n"], as_index=False)["S"].mean()

    rows = []
    for (mode, q), sub in agg.groupby(["mode", "q"]):
        sub = sub.sort_values("n")
        fit = fit_loglog(np.log(sub["n"]), np.log(sub["S"]))
        d_hat = d_from_energy_slope(alpha, fit["slope"])
        # delta method: d = alpha/(1-b)  =>  sd(d) = sd(b) * alpha/(1-b)^2
        denom = 1.0 - fit["slope"]
        d_se = (
            fit["slope_se"] * alpha / denom**2
            if np.isfinite(fit["slope_se"]) and abs(denom) > 1e-12
            else np.nan
        )
        rows.append(
            {
                "mode": mode,
                "q": float(q),
                "d_hat": d_hat,
                "d_se": d_se,
                **fit,
                "n_min": int(sub["n"].min()),
                "n_max": int(sub["n"].max()),
            }
        )
    return pd.DataFrame(rows)
