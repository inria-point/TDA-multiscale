"""Translate the three bands into the edge types they are actually made of.

edge_taxonomy.py gives the composition of every 5% slice of the edge-length
distribution. A band is a range of q, and q is a trimming fraction, so a band is
literally a set of those slices: q_small at q keeps the edges above the q-th
quantile, q_large keeps those below the (1-q)-th, and q0.5_range keeps a window
of width 0.5 starting at q. Averaging the kept set over the band's q range gives
a weight for every slice, and the weights turn the composition table into the
one sentence the report needs: this band looks at these tokens.

Two weightings, because d is not read off the edges themselves but off
S(q) = sum of the retained lengths:

  по числу рёбер   what share of the retained edges is of each type
  по длине         what share of the retained *sum* each type contributes,
                   which is the quantity the log-log fit is run on

The q grid has step 0.05 and the slices are 5% wide, so the two line up exactly
and no interpolation is needed.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from edge_taxonomy import CELLS, N_BINS
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
QMAX = {"q_small": 0.9, "q_large": 0.9, "q0.5_range": 0.5}


def weights(mode, qmin, qmax, n=N_BINS):
    """Mean retention of each 5% slice over the band's q range."""
    w = np.zeros(n)
    qs = np.round(np.arange(qmin, qmax + 1e-9, 0.05), 2)
    for q in qs:
        k = int(round(q * n))
        if mode == "q_small":
            w[k:] += 1
        elif mode == "q_large":
            w[:n - k] += 1
        elif mode == "q0.5_range":
            w[k:min(n, k + n // 2)] += 1
    return w / w.sum()


def main():
    res = os.path.join(BASE, "results")
    comp = pd.read_csv(os.path.join(res, "edge_taxonomy_bins.csv"),
                       index_col=0)[CELLS]
    geom = pd.read_csv(os.path.join(res, "edge_taxonomy_geometry.csv"),
                       index_col=0)
    pd.set_option("display.width", 250)

    by_count, by_len, spans = {}, {}, {}
    for name, (mode, qmin) in BANDS.items():
        w = weights(mode, qmin, QMAX[mode])
        by_count[name] = w @ comp.values
        wl = w * geom["rel_len"].values
        wl = wl / wl.sum()
        by_len[name] = wl @ comp.values
        used = np.where(w > 0)[0]
        spans[name] = (f"{used[0] * 5}-{(used[-1] + 1) * 5}% перцентиля, "
                       f"центр тяжести {(w * (np.arange(N_BINS) + 0.5) * 5).sum():.0f}%")

    for title, tbl in [("по числу рёбер", by_count), ("по длине (то, из чего "
                                                      "складывается S)", by_len)]:
        t = pd.DataFrame(tbl, index=CELLS).T  # comp is already in per cent
        print(f"\nсостав полосы, {title}, %\n")
        print(t.round(1).to_string())
        tag = "count" if "числу" in title else "len"
        t.round(3).to_csv(os.path.join(res, f"band_composition_{tag}.csv"))

    print("\nкакие перцентили длины полоса вообще видит\n")
    for k, v in spans.items():
        print(f"  {k:9s} {v}")

    # the three summary numbers the report leans on
    t = pd.DataFrame(by_count, index=CELLS).T
    same = [c for c in CELLS if c.startswith("=")]
    print("\nсводка\n")
    s = pd.DataFrame({
        "один и тот же токен, %": t[same].sum(axis=1),
        "служебное или пунктуация, %": t[[c for c in CELLS
                                          if "пунктуац" in c or "служеб" in c]
                                         ].sum(axis=1),
        "два разных смысловых, %": t["≠ смысловые"],
        "куски одного слова, %": t["куски одного слова"],
    })
    print(s.round(1).to_string())
    s.round(3).to_csv(os.path.join(BASE, "results", "band_composition_summary.csv"))


if __name__ == "__main__":
    main()
