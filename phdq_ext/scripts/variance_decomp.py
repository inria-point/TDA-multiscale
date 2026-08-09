"""Split the spread of d_hat into estimator noise and real text-to-text signal.

d_hat is measured, not observed: for a fixed text it still depends on which
tokens and which subsamples were drawn. Re-running the same text under
different seeds gives repeated measurements of one underlying value, so a
one-way random-effects decomposition applies:

    d_hat[i][j] = mu + a[i] + e[i][j]      i = text, j = seed

    a[i]    ~ (0, var_between)   real differences between texts
    e[i][j] ~ (0, var_within)    measurement noise of the estimator

var_within is estimated by pooling the per-text variance across seeds.
The variance of the per-text *means* is inflated by the noise that survives
averaging, so it is debiased:

    var_between = var(mean_j d_hat[i][j]) - var_within / n_seeds

The reported ratio sqrt(var_between)/sqrt(var_within) is the signal-to-noise
of a single measurement: below 1 the estimator cannot tell texts apart.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data import GENRES, load
from embedder import Embedder
from qphd import qphd

BASE = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--n-texts", type=int, default=10, help="texts per genre")
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--replicates", type=int, default=8)
    ap.add_argument("--source", default="human")
    args = ap.parse_args()

    q_list = tuple(np.round(np.arange(0, 0.901, 0.1), 2))
    emb = Embedder()
    rows = []
    for genre in GENRES:
        got = 0
        for i, (tid, text) in enumerate(load(genre, args.source)):
            if got >= args.n_texts:
                break
            e = emb.embed_cached(text, cache_key=f"{args.source}_{genre}_{i}")
            if e.shape[0] < args.L:
                continue
            for seed in range(args.n_seeds):
                rng = np.random.default_rng(10007 * seed + i)
                idx = rng.choice(e.shape[0], args.L, replace=False)
                df = qphd(e[idx], q_list=q_list, replicates=args.replicates, rng=rng)
                df["genre"] = genre
                df["text"] = f"{genre}_{i}"
                df["seed"] = seed
                rows.append(df)
            got += 1
        print(f"{genre}: {got} texts x {args.n_seeds} seeds", flush=True)

    d = pd.concat(rows)
    d = d[d["d_hat"] > 0]

    out = []
    for (mode, q), s in d.groupby(["mode", "q"]):
        per_text = s.groupby("text")["d_hat"]
        n_seeds = s.groupby("text").size().min()
        var_within = float(per_text.var().mean())
        var_mean = float(per_text.mean().var())
        var_between = max(0.0, var_mean - var_within / n_seeds)
        sd_w, sd_b = np.sqrt(var_within), np.sqrt(var_between)
        out.append(
            {
                "mode": mode,
                "q": q,
                "mean_d": s["d_hat"].mean(),
                "sd_within": sd_w,
                "sd_between": sd_b,
                "snr": sd_b / sd_w if sd_w > 0 else np.nan,
                "cv_between": sd_b / s["d_hat"].mean(),
                "resid_rmse": s["resid_rmse"].mean(),
                "d_se": s["d_se"].mean(),
                "n_texts": s["text"].nunique(),
            }
        )
    res = pd.DataFrame(out)
    path = os.path.join(BASE, "results", f"variance_decomp_{args.source}_L{args.L}.csv")
    res.to_csv(path, index=False)

    pd.set_option("display.width", 200)
    for mode in ["q_small", "q_large", "q0.5_range"]:
        sub = res[res["mode"] == mode]
        if sub.empty:
            continue
        print(f"\n=== {mode}")
        print(sub.drop(columns=["mode"]).round(3).to_string(index=False))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
