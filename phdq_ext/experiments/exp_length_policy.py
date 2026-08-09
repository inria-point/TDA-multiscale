"""Experiment 1: at fixed L, does the original text length still matter?
And is L better set as a fraction of the text than as an absolute count?

d_hat should be a property of the text, not of how long it happens to be.
So for each policy we regress log d_hat on log(n_tokens) *within genre*
(genre is confounded with length, so it is partialled out) and report the
slope. Slope 0 = the policy has removed the length dependence.

Policies
  abs_L        L tokens, subsamples drawn from the whole text (the fixed config)
  abs_L_pool2L L tokens, pool restricted to a 2L subsample -> pool size is
               the same for every text, isolating the effect of pool size
  frac_50      L = 50% of the text
  frac_100     L = the whole text
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
from qphd import qphd

BASE = os.path.join(os.path.dirname(__file__), "..")
Q_LIST = (0.0, 0.3, 0.6, 0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--n-texts", type=int, default=40, help="per genre")
    ap.add_argument("--min-tokens", type=int, default=320)
    ap.add_argument("--max-tokens", type=int, default=900,
                    help="cap; frac_100 runs the MST on the whole text")
    ap.add_argument("--replicates", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    emb = Embedder()
    clouds = []
    for genre in GENRES:
        got = 0
        for i, (tid, text) in enumerate(load(genre, "human")):
            if got >= args.n_texts:
                break
            e = emb.embed_cached(text, cache_key=f"human_{genre}_{i}")
            if not (args.min_tokens <= e.shape[0] <= args.max_tokens):
                continue
            clouds.append((f"{genre}_{i}", genre, e))
            got += 1
    lens = np.array([e.shape[0] for _, _, e in clouds])
    print(f"{len(clouds)} texts, tokens {lens.min()}-{lens.max()} "
          f"(median {np.median(lens):.0f})", flush=True)

    rows = []
    for policy in ["abs_L", "abs_L_pool2L", "frac_50", "frac_100"]:
        for name, genre, e in clouds:
            ntok = e.shape[0]
            for seed in range(args.seeds):
                rng = np.random.default_rng(10007 * seed + hash(name) % 9973)
                if policy == "abs_L":
                    L, pool = args.L, e
                elif policy == "abs_L_pool2L":
                    L = args.L
                    pool = e[rng.choice(ntok, min(2 * args.L, ntok), replace=False)]
                elif policy == "frac_50":
                    L, pool = max(64, ntok // 2), e
                else:
                    L, pool = ntok, e
                pts = e[rng.choice(ntok, L, replace=False)]
                df = qphd(pts, q_list=Q_LIST, rng=rng,
                          **cfg.qphd_kwargs(pool=pool, replicates=args.replicates))
                df["policy"] = policy
                df["genre"] = genre
                df["text"] = name
                df["ntok"] = ntok
                df["L_used"] = L
                rows.append(df)
        print(f"  {policy} done", flush=True)

    d = pd.concat(rows)
    d = d[d["d_hat"] > 0]
    path = os.path.join(BASE, "results", "exp_length_policy.csv")
    d.to_csv(path, index=False)

    # within-genre regression of log d on log n_tokens
    out = []
    for (policy, mode, q), s in d.groupby(["policy", "mode", "q"]):
        slopes, resids = [], []
        for genre, g in s.groupby("genre"):
            per = g.groupby("text").agg(d=("d_hat", "mean"), n=("ntok", "first"))
            if len(per) < 5 or per["n"].nunique() < 5:
                continue
            b, a = np.polyfit(np.log(per["n"]), np.log(per["d"]), 1)
            slopes.append(b)
            resids.append(np.std(np.log(per["d"]) - (a + b * np.log(per["n"]))))
        out.append(
            {
                "policy": policy,
                "mode": mode,
                "q": q,
                "slope_vs_len": np.mean(slopes),
                "resid_sd": np.mean(resids),
                "cv_texts": (s.groupby("text")["d_hat"].mean().std()
                             / s["d_hat"].mean()),
            }
        )
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(BASE, "results", "exp_length_policy_summary.csv"),
               index=False)

    pd.set_option("display.width", 220)
    for mode in ["q_small", "q_large"]:
        print(f"\n=== {mode}: slope of log d_hat vs log n_tokens (0 = length-free)")
        print(res[res["mode"] == mode].pivot(index="policy", columns="q",
              values="slope_vs_len").round(3).to_string())
        print(f"\n=== {mode}: spread of d_hat across texts (cv)")
        print(res[res["mode"] == mode].pivot(index="policy", columns="q",
              values="cv_texts").round(3).to_string())
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
