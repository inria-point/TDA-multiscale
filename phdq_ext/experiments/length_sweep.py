"""How much does d_hat still depend on L, once the estimator is fixed?

The same texts are measured at several L. Only the subsample-size window
changes (n runs over [0.2L, L]); with pool = the whole text the point cloud
itself is identical across L, so the comparison isolates the window.

Run under both the paper's settings and the improved ones to see how much
of the L-dependence was bootstrap artefact and how much is intrinsic
multi-scale behaviour.
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
LOG8 = tuple(np.round(np.logspace(np.log10(0.2), 0.0, 8), 3))
UNIFORM5 = (0.2, 0.4, 0.6, 0.8, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-tokens", type=int, default=768)
    ap.add_argument("--n-texts", type=int, default=6, help="per genre")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--L-list", type=int, nargs="+",
                    default=[128, 192, 256, 384, 512, 768])
    args = ap.parse_args()

    emb = Embedder()
    clouds = []
    for genre in GENRES:
        got = 0
        for i, (tid, text) in enumerate(load(genre, "human")):
            if got >= args.n_texts:
                break
            e = emb.embed_cached(text, cache_key=f"human_{genre}_{i}")
            if e.shape[0] < args.min_tokens:
                continue
            clouds.append((f"{genre}_{i}", genre, e))
            got += 1
    print(f"{len(clouds)} texts with >={args.min_tokens} tokens", flush=True)
    print(pd.Series([g for _, g, _ in clouds]).value_counts().to_string(), flush=True)

    rows = []
    for config in ["legacy", "v2"]:
        for L in args.L_list:
            for name, genre, e in clouds:
                for seed in range(args.n_seeds):
                    rng = np.random.default_rng(10007 * seed + hash(name) % 9973)
                    sub = e[rng.choice(e.shape[0], L, replace=False)]
                    df = qphd(
                        sub,
                        q_list=(0.0, 0.3, 0.6, 0.9),
                        n_fraction_list=UNIFORM5 if config == "legacy" else LOG8,
                        replicates=8 if config == "legacy" else 32,
                        replace=(config == "legacy"),
                        pool=None if config == "legacy" else e,
                        rng=rng,
                    )
                    df["L"] = L
                    df["config"] = config
                    df["text"] = name
                    rows.append(df)
            print(f"  {config} L={L} done", flush=True)

    d = pd.concat(rows)
    d = d[d["d_hat"] > 0]
    path = os.path.join(BASE, "results", "length_sweep.csv")
    d.to_csv(path, index=False)

    pd.set_option("display.width", 200)
    for config in ["legacy", "v2"]:
        s = d[d["config"] == config]
        print(f"\n=== {config}: mean d_hat by L")
        print(s.pivot_table(index=["mode", "q"], columns="L",
                            values="d_hat").round(2).to_string())
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
