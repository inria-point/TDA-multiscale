"""Models and perturbations in three interpretable bands instead of 49 numbers.

The principal components are a fitted basis; these three are read straight off
the loading curves, where each component's weight is concentrated:

  крупный   q_small,     q >= 0.5   the long edges kept
  мелкий    q_large,     q >= 0.6   the short edges kept
  средний   q0.5_range,  q >= 0.3   the central band

Each is the mean relative change of d over its range, in per cent, against the
same human baseline the map uses. Unlike the components these do not rotate
when the set changes, so a position means the same thing in every run.

The question the bands are for: of the eight sign combinations, which are
occupied by generators, which by perturbations, and which by neither.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import taxonomy as T
from pc_space import paired, profiles

BASE = os.path.join(HERE, "..")
# Two cuts, kept side by side rather than one replacing the other.
#
# "base" is where the loading curves put their weight. "hi" pushes both
# one-sided bands out to 0.7: the central band is a window of width 0.5, so at
# q = 0.5 it already spans the upper half of the edges and overlaps the coarse
# band by construction, and the two cuts differ in how much of that overlap
# survives. Select with the BANDS environment variable; every output file
# carries the variant in its name.
BAND_SETS = {
    "base": {"крупный": ("q_small", 0.5), "мелкий": ("q_large", 0.6),
             "средний": ("q0.5_range", 0.3)},
    "hi": {"крупный": ("q_small", 0.7), "мелкий": ("q_large", 0.7),
           "средний": ("q0.5_range", 0.3)},
}
VARIANT = os.environ.get("BANDS", "base")
BANDS = BAND_SETS[VARIANT]
SUF = "" if VARIANT == "base" else f"_{VARIANT}"
FILES = ["perturb_L201_coling_all.csv.gz", "perturb_L201_hapax.csv.gz",
         "perturb_L201_pc2neg.csv.gz", "perturb_L201_pc2probes.csv.gz",
         "perturb_L201_collapse.csv.gz", "perturb_L201_composed.csv.gz",
         "perturb_L201_rarify4.csv.gz", "perturb_L201_rarify3.csv.gz",
         "perturb_L201_downmore.csv.gz", "perturb_L201_syn.csv.gz",
         "perturb_L201_scriptessay.csv.gz"]


def bands_from(pr_by_mode):
    out = {}
    for name, (mode, qmin) in BANDS.items():
        m = pr_by_mode[mode]
        cols = [c for c in m.columns if c >= qmin]
        out[name] = m[cols].mean(axis=1)
    return pd.DataFrame(out)


def perturbations():
    frames = []
    for fn in FILES:
        path = os.path.join(BASE, "results", fn)
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        p = paired(d[d["d_hat"] > 0])
        frames.append(bands_from({m: profiles(p, m) for m in
                                  ["q_small", "q_large", "q0.5_range"]}))
    P = pd.concat(frames)
    return P[~P.index.duplicated(keep="last")]


def generators(min_texts=20):
    d = pd.read_csv(os.path.join(BASE, "results", "coling_qphd_L201.csv.gz"))
    d = d[d["d_hat"] > 0].copy()
    h = (d[d["is_human"]].groupby(["sub_source", "mode", "q"])["d_hat"]
         .mean().rename("h"))
    d = d.join(h, on=["sub_source", "mode", "q"]).dropna(subset=["h"])
    d["rel"] = (d["d_hat"] - d["h"]) / d["h"] * 100
    n = d[(d["mode"] == "q_small") & (d["q"] == 0)].groupby("model").size()
    keep = n[n >= min_texts].index.difference(["human"])
    d = d[d["model"].isin(keep)]
    piv = {m: d[d["mode"] == m].pivot_table(index="model", columns="q",
                                            values="rel")
           for m in ["q_small", "q_large", "q0.5_range"]}
    return bands_from(piv)


def octant(row, dead=3.0):
    """Sign pattern, with a dead zone so noise is not called a direction."""
    return "".join("0" if abs(v) < dead else ("+" if v > 0 else "−")
                   for v in row)


def main():
    G, P = generators(), perturbations()
    P.index = [T.KEEP.get(i, i) for i in P.index]
    G["octant"] = [octant(r) for _, r in G[list(BANDS)].iterrows()]
    P["octant"] = [octant(r) for _, r in P[list(BANDS)].iterrows()]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 120)

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    for name, D in [("МОДЕЛИ", G), ("ПЕРТУРБАЦИИ", P)]:
        X = D[list(BANDS)].values
        Z = linkage(pdist(X), method="ward")
        D["кластер"] = fcluster(Z, 4, criterion="maxclust")
        print(f"\n{'=' * 70}\n{name}: {len(D)} точек, три полосы в %\n")
        print(D.sort_values(["кластер", "крупный"]).round(1).to_string())
        print(f"\nсредние по кластерам:")
        print(D.groupby("кластер")[list(BANDS)].agg(["mean", "size"])
              .round(1).to_string())

    print(f"\n{'=' * 70}\nПОКРЫТИЕ: знаки (крупный, мелкий, средний), "
          f"мёртвая зона |x| < 3%\n")
    rows = []
    for oct_ in sorted(set(G["octant"]) | set(P["octant"])):
        g = G[G["octant"] == oct_]
        p = P[P["octant"] == oct_]
        rows.append({"знаки": oct_, "моделей": len(g), "пертурбаций": len(p),
                     "модели": ", ".join(list(g.index)[:4]),
                     "пертурбации": ", ".join(list(p.index)[:4])})
    print(pd.DataFrame(rows).to_string(index=False))

    miss = [r for r in rows if r["моделей"] and not r["пертурбаций"]]
    print("\nзаняты моделями, но не покрыты пертурбациями:",
          ", ".join(r["знаки"] for r in miss) or "нет")
    G.round(2).to_csv(os.path.join(BASE, "results", f"bands_models{SUF}.csv"))
    P.round(2).to_csv(os.path.join(BASE, "results", f"bands_perts{SUF}.csv"))

    print(f"\n{'=' * 70}\nПЕРТУРБАЦИИ ПО ПОЛОСАМ\n")
    P["n_полос"] = [sum(c != "0" for c in o) for o in P["octant"]]
    for k in [1, 2, 3]:
        sub = P[P["n_полос"] == k]
        if sub.empty:
            continue
        title = {1: "ОДНОПОЛОСНЫЕ — двигают ровно одну полосу",
                 2: "ДВУХПОЛОСНЫЕ", 3: "ТРЁХПОЛОСНЫЕ"}[k]
        print(f"--- {title}: {len(sub)}\n")
        for oct_ in sorted(sub["octant"].unique()):
            g = sub[sub["octant"] == oct_]
            g = g.reindex(g[list(BANDS)].abs().max(axis=1)
                          .sort_values(ascending=False).index)
            which = [n for n, c in zip(BANDS, oct_) if c != "0"]
            sign = [c for c in oct_ if c != "0"]
            label = ", ".join(f"{w} {s_}" for w, s_ in zip(which, sign))
            print(f"  [{oct_}]  {label}")
            print(g[list(BANDS)].round(1).to_string(header=False))
            print()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    pairs = [("крупный", "мелкий"), ("крупный", "средний"),
             ("мелкий", "средний")]
    for ax, (a, b) in zip(axes, pairs):
        ax.axhline(0, c="k", lw=1)
        ax.axvline(0, c="k", lw=1)
        ax.scatter(P[a], P[b], s=34, c="#4a5568", label="пертурбации",
                   alpha=0.75)
        ax.scatter(G[a], G[b], s=90, c="#c53030", marker="^",
                   label="генераторы", zorder=4)
        ax.scatter(0, 0, s=420, marker="*", c="#1a7f37", zorder=5,
                   edgecolors="black", linewidths=0.8)
        ax.set_xlabel(f"{a}, %")
        ax.set_ylabel(f"{b}, %")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=9)
    fig.suptitle("Три полосы: генераторы и пертурбации; звезда — человек")
    fig.tight_layout()
    path = os.path.join(BASE, "figures", f"three_bands{SUF}.png")
    fig.savefig(path, dpi=150)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
