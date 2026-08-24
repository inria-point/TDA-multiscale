"""Pull the texts that a hypothesis is about, so it can be read and not only scored.

For a feature that separates the extremes of d at some q, this shows the texts
sitting at both ends of *both* axes at once: high d and high on the feature
against low d and low on the feature. Those are the texts a perturbation would
have to turn into one another.
"""
import argparse
import os
import sys
import textwrap

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features", nargs="+", help="feature names to illustrate")
    ap.add_argument("--mode", default="q_small")
    ap.add_argument("--q", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=2, help="texts per end")
    ap.add_argument("--chars", type=int, default=560)
    args = ap.parse_args()

    feat = pd.read_csv(os.path.join(BASE, "results", "coling_features.csv"))
    d = pd.read_csv(os.path.join(BASE, "results", "coling_qphd_L201.csv.gz"))
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    d = d[(d["mode"] == args.mode) & (np.isclose(d["q"], args.q)) & (d["d_hat"] > 0)]

    m = feat.merge(d[["id", "d_hat"]], on="id").merge(
        pool[["id", "text"]], on="id")
    # rank within domain, so the illustration is not just a domain contrast
    for c in args.features + ["d_hat"]:
        m[c + "_z"] = m.groupby("sub_source")[c].transform(
            lambda x: (x - x.mean()) / (x.std() or 1))

    for f in args.features:
        print("=" * 100)
        print(f"ГИПОТЕЗА: {f}   ({args.mode}, q={args.q})")
        print("=" * 100)
        for label, sign in [("ВЫСОКАЯ d, высокий признак", +1),
                            ("НИЗКАЯ d, низкий признак", -1)]:
            score = sign * (m[f + "_z"] + m["d_hat_z"])
            sel = m.loc[score.nlargest(args.k).index]
            print(f"\n--- {label}")
            for _, r in sel.iterrows():
                print(f"  [{r['model']} / {r['sub_source']}]  "
                      f"d={r['d_hat']:.1f}  {f}={r[f]:.3f}")
                body = " ".join(str(r["text"]).split())[: args.chars]
                print(textwrap.fill(body, 96, initial_indent="    ",
                                    subsequent_indent="    "))
                print()


if __name__ == "__main__":
    main()
