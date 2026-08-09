"""Compare qPHD estimator variants by measurement quality.

Each variant is scored on the same texts with the same seeds, so the
comparison is paired. Reported per (variant, mode, q):

    cv_within  = sd_within / mean_d   relative noise of one measurement
    rel_resid  = resid_rmse / (|b| * log(n_max/n_min))
    mean_d                            the value itself, to catch shifts

Variants cover the three axes asked about: where the subsample is drawn
from (fixed L-token subset vs the whole text), how many replicates are
averaged, and how the subsample sizes n are spaced.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from data import GENRES, load
from embedder import Embedder
from qphd import qphd

BASE = os.path.join(os.path.dirname(__file__), "..")

UNIFORM5 = (0.2, 0.4, 0.6, 0.8, 1.0)
LOG5 = tuple(np.round(np.logspace(np.log10(0.2), 0.0, 5), 3))
MORE_SMALL = (0.2, 0.25, 0.32, 0.45, 1.0)
MORE_LARGE = (0.2, 0.55, 0.75, 0.9, 1.0)
LOG8 = tuple(np.round(np.logspace(np.log10(0.2), 0.0, 8), 3))
WIDE = (0.1, 0.2, 0.35, 0.6, 1.0)

VARIANTS = {
    # name:            (fractions, replicates, replace, pool_is_full_text)
    "baseline":        (UNIFORM5,   8,  True,  False),
    "no_replace":      (UNIFORM5,   8,  False, False),
    "pool_full":       (UNIFORM5,   8,  False, True),
    "rep32":           (UNIFORM5,  32,  True,  False),
    "rep64":           (UNIFORM5,  64,  True,  False),
    "grid_log":        (LOG5,       8,  True,  False),
    "grid_more_small": (MORE_SMALL, 8,  True,  False),
    "grid_more_large": (MORE_LARGE, 8,  True,  False),
    "grid_log8":       (LOG8,       8,  True,  False),
    "grid_wide":       (WIDE,       8,  True,  False),
    "combo":           (LOG8,      32,  False, True),
}


def decompose(df, n_seeds):
    """One-way random-effects split of d_hat, see variance_decomp.py."""
    out = []
    for (mode, q), s in df.groupby(["mode", "q"]):
        per_text = s.groupby("text")["d_hat"]
        var_w = float(per_text.var().mean())
        var_b = max(0.0, float(per_text.mean().var()) - var_w / n_seeds)
        mean_d = s["d_hat"].mean()
        span = np.log(s["n_max"] / s["n_min"])
        rel_resid = (s["resid_rmse"] / (s["slope"].abs() * span)).mean()
        out.append(
            {
                "mode": mode,
                "q": q,
                "mean_d": mean_d,
                "cv_within": np.sqrt(var_w) / mean_d,
                "cv_between": np.sqrt(var_b) / mean_d,
                "rel_resid": rel_resid,
                "rel_dse": (s["d_se"] / s["d_hat"]).mean(),
            }
        )
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--n-texts", type=int, default=5, help="per genre")
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--source", default="human")
    args = ap.parse_args()

    q_list = (0.0, 0.3, 0.5, 0.6, 0.9)
    emb = Embedder()

    # collect the token clouds once
    clouds = []
    for genre in GENRES:
        got = 0
        for i, (tid, text) in enumerate(load(genre, args.source)):
            if got >= args.n_texts:
                break
            e = emb.embed_cached(text, cache_key=f"{args.source}_{genre}_{i}")
            if e.shape[0] < args.L:
                continue
            clouds.append((f"{genre}_{i}", genre, e))
            got += 1
    print(f"{len(clouds)} texts, {args.n_seeds} seeds, {len(VARIANTS)} variants",
          flush=True)

    all_res = []
    for vname, (fracs, reps, replace, pool_full) in VARIANTS.items():
        rows = []
        for name, genre, e in clouds:
            for seed in range(args.n_seeds):
                rng = np.random.default_rng(10007 * seed + hash(name) % 9973)
                idx = rng.choice(e.shape[0], args.L, replace=False)
                subset = e[idx]
                df = qphd(
                    subset,
                    q_list=q_list,
                    n_fraction_list=fracs,
                    replicates=reps,
                    replace=replace,
                    pool=e if pool_full else None,
                    rng=rng,
                )
                df["text"] = name
                rows.append(df)
        res = decompose(pd.concat(rows), args.n_seeds)
        res["variant"] = vname
        all_res.append(res)
        m = res[(res["mode"] == "q_small") & (res["q"] == 0.3)].iloc[0]
        print(f"  {vname:16s} q_small@0.3: d={m['mean_d']:6.2f} "
              f"cv_within={m['cv_within']:.3f} rel_resid={m['rel_resid']:.4f}",
              flush=True)

    out = pd.concat(all_res)
    path = os.path.join(BASE, "results", f"variant_sweep_L{args.L}.csv")
    out.to_csv(path, index=False)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
