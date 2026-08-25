"""A two-number notation for a qPHD profile, and the plane it lives in.

A profile is nineteen numbers per trimming mode, but it is not nineteen-
dimensional: over the perturbations measured here two components carry 95% of
the variance, and they line up with two directly readable coordinates.

    C  (coarse)  mean effect in q_small over q = 0.1..0.3
    F  (fine)    mean effect in q_large over q = 0.7..0.9

C tracks the first component at |r| = 0.99, F the second at 0.91. So a
manipulation, a style or a generator can be written as a pair (C, F) in per
cent, and the quadrants of that plane are the profile types: both up, both
down, and the mixed corner where the coarse scale falls while the fine scale
rises -- which is where the degenerate decoders sit.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_property_map import DEFECTS, paired, profiles

BASE = os.path.join(os.path.dirname(__file__), "..")
COARSE = (0.1, 0.3)
FINE = (0.7, 0.9)


def cf(prof_small, prof_large, index=None):
    """The pair (C, F) for every row."""
    def band(m, lo, hi):
        cols = [c for c in m.columns if lo - 1e-9 <= float(c) <= hi + 1e-9]
        return m[cols].mean(axis=1)

    c = band(prof_small, *COARSE)
    f = band(prof_large, *FINE)
    out = pd.DataFrame({"C": c, "F": f})
    return out.reindex(index) if index is not None else out


def generator_cf():
    q = pd.read_csv(os.path.join(BASE, "results", "coling_qphd_L201.csv.gz"))
    d = q[q["d_hat"] > 0].copy()
    human = (d[d["is_human"]].groupby(["sub_source", "mode", "q"])["d_hat"]
             .mean().rename("h"))
    d = d.join(human, on=["sub_source", "mode", "q"]).dropna(subset=["h"])
    d["rel"] = (d["d_hat"] - d["h"]) / d["h"] * 100
    n = d[(d["mode"] == "q_small") & (d["q"] == 0)].groupby("model").size()
    keep = n[n >= 25].index.difference(["human"])
    s = d[(d["mode"] == "q_small") & d["model"].isin(keep)]
    l = d[(d["mode"] == "q_large") & d["model"].isin(keep)]
    out = cf(s.pivot_table(index="model", columns="q", values="rel"),
             l.pivot_table(index="model", columns="q", values="rel"))
    out["n"] = n.reindex(out.index)
    return out


def main():
    pert = pd.read_csv(os.path.join(BASE, "results", "perturb_L201_coling.csv.gz"))
    p = paired(pert[pert["d_hat"] > 0])
    P = cf(profiles(p, "q_small"), profiles(p, "q_large"))
    G = generator_cf()

    P.round(1).to_csv(os.path.join(BASE, "results", "profile_plane_perturbations.csv"))
    G.round(1).to_csv(os.path.join(BASE, "results", "profile_plane_generators.csv"))

    pd.set_option("display.width", 200)
    print("=== пертурбации, (C, F) в %")
    print(P.round(1).sort_values("C").to_string())
    print("\n=== генераторы, (C, F) в % относительно человека")
    print(G.round(1).sort_values("C").to_string())
    print(f"\nкорреляция C и F: пертурбации {P['C'].corr(P['F']):+.2f}, "
          f"генераторы {G['C'].corr(G['F']):+.2f}")

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.axhline(0, c="k", lw=1)
    ax.axvline(0, c="k", lw=1)
    for name, r in P.iterrows():
        col = "#b7791f" if name in DEFECTS else (
            "#2b6cb0" if name.startswith("style_") else "#4a5568")
        ax.scatter(r["C"], r["F"], s=42, c=col, marker="o", zorder=3)
        ax.annotate(name, (r["C"], r["F"]), fontsize=6.6, color=col,
                    xytext=(4, 3), textcoords="offset points")
    for name, r in G.iterrows():
        ax.scatter(r["C"], r["F"], s=54, c="#c53030", marker="^", zorder=4)
        ax.annotate(name, (r["C"], r["F"]), fontsize=6.6, color="#c53030",
                    xytext=(4, -8), textcoords="offset points")
    ax.set_xlabel("C — крупный масштаб: q_small, q = 0.1…0.3, % к базе")
    ax.set_ylabel("F — мелкий масштаб: q_large, q = 0.7…0.9, % к базе")
    ax.set_title("Плоскость профилей: пертурбации (круги) и генераторы "
                 "(треугольники)\nжёлтые — дефекты, синие — стили, "
                 "серые — механические, красные — модели")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(BASE, "figures", "profile_plane.png")
    fig.savefig(path, dpi=150)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
