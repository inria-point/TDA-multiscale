"""Experiment 2: should L depend on q?

Trimming a fraction q leaves only (1-q)*(n-1) edges in the sum, so the high-q
end of the curve is estimated from far fewer edges than the low-q end. The fix
is to scale L so that the number of *retained* edges is the same at every q:

    q_small / q_large : kept fraction = 1-q     -> L(q) = N / (1-q)
    q0.5_range        : kept fraction = 0.5     -> L(q) = 2N   (already flat)

Confound to keep in mind: d_hat itself depends on L (d ~ L^-0.2, see
length_sweep.py), so scaling L with q moves the curve for two different
reasons at once. Both the fixed-L and the scaled-L curve are therefore
reported, along with a fixed-L curve at the *mean* scaled L, which separates
"equal edge counts" from "bigger L overall".
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import config as cfg
from data import GENRES, load
from embedder import Embedder
from qphd import Q_MAX_BY_MODE, qphd

BASE = os.path.join(os.path.dirname(__file__), "..")


def kept_fraction(mode, q, p_range=0.5):
    if mode == "q0.5_range":
        return min(1.0, q + p_range) - q
    return 1.0 - q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100, help="target retained edges")
    ap.add_argument("--q-max", type=float, default=0.8)
    ap.add_argument("--n-texts", type=int, default=8, help="per genre")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--q-step", type=float, default=0.2)
    ap.add_argument("--replicates", type=int, default=12)
    args = ap.parse_args()

    q_list = tuple(np.round(np.arange(0, args.q_max + 0.001, args.q_step), 2))
    L_max = int(np.ceil(args.N / (1.0 - args.q_max)))
    print(f"N={args.N}, q<={args.q_max} -> L ranges {args.N}..{L_max}", flush=True)

    emb = Embedder()
    clouds = []
    for genre in GENRES:
        got = 0
        for i, (tid, text) in enumerate(load(genre, "human")):
            if got >= args.n_texts:
                break
            e = emb.embed_cached(text, cache_key=f"human_{genre}_{i}")
            if e.shape[0] < L_max:
                continue
            clouds.append((f"{genre}_{i}", genre, e))
            got += 1
    print(f"{len(clouds)} texts with >={L_max} tokens", flush=True)
    print(pd.Series([g for _, g, _ in clouds]).value_counts().to_string(), flush=True)

    # mean L of the scaled policy, so the third arm matches it on average
    scaled_Ls = [args.N / kept_fraction("q_small", q) for q in q_list]
    L_mean = int(np.mean(scaled_Ls))
    print(f"scaled L per q: {[int(x) for x in scaled_Ls]}; mean {L_mean}", flush=True)

    rows = []
    for name, genre, e in clouds:
        ntok = e.shape[0]
        for seed in range(args.seeds):
            for arm, L_of_q in [
                ("fixed_L_small", lambda m, q: args.N),
                ("fixed_L_mean", lambda m, q: L_mean),
                ("scaled_L", lambda m, q: int(np.ceil(args.N / kept_fraction(m, q)))),
            ]:
                for mode in ["q_small", "q_large", "q0.5_range"]:
                    for q in q_list:
                        if q > Q_MAX_BY_MODE[mode]:
                            continue
                        L = min(L_of_q(mode, q), ntok)
                        rng = np.random.default_rng(
                            10007 * seed + hash(name) % 9973 + int(q * 100)
                        )
                        pts = e[rng.choice(ntok, L, replace=False)]
                        df = qphd(pts, q_list=(q,), modes=[mode], rng=rng,
                                  **cfg.qphd_kwargs(pool=e,
                                                    replicates=args.replicates))
                        df["arm"] = arm
                        df["genre"] = genre
                        df["text"] = name
                        df["L_used"] = L
                        df["seed"] = seed
                        rows.append(df)
        print(f"  {name} done", flush=True)

    d = pd.concat(rows)
    d = d[d["d_hat"] > 0]
    path = os.path.join(BASE, "results", "exp_L_per_q.csv")
    d.to_csv(path, index=False)

    out = []
    for (arm, mode, q), s in d.groupby(["arm", "mode", "q"]):
        per_text = s.groupby("text")["d_hat"]
        n_seeds = s.groupby("text").size().min()
        var_w = float(per_text.var().mean())
        mean_d = s["d_hat"].mean()
        out.append(
            {
                "arm": arm, "mode": mode, "q": q, "mean_d": mean_d,
                "cv_within": np.sqrt(var_w) / mean_d,
                "rel_resid": (s["resid_rmse"] / (s["slope"].abs()
                              * np.log(s["n_max"] / s["n_min"]))).mean(),
                "L": s["L_used"].mean(),
            }
        )
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(BASE, "results", "exp_L_per_q_summary.csv"), index=False)

    pd.set_option("display.width", 220)
    for mode in ["q_small", "q_large"]:
        for col in ["mean_d", "cv_within"]:
            print(f"\n=== {mode}: {col}")
            print(res[res["mode"] == mode].pivot(index="arm", columns="q",
                  values=col).round(3).to_string())
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
