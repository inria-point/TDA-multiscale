"""Models and perturbations in one principal-component basis.

Both are expressed the same way — per cent change of d relative to a human
baseline at every (mode, q) — so they are commensurable and can share a basis.
The basis is fitted on the union, and both groups are projected into it, which
is what makes the question "does this generator look like that perturbation"
answerable geometrically rather than by eye.

The full profile is used: all three trimming modes, 49 coordinates. Fitting on
a subset of modes shifts the components materially — dropping q0.5_range moved
PC1 from 96% to 93% and inflated PC3 sixfold.
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
ALL_MODES = ["q_small", "q_large", "q0.5_range"]


def full_profile(pivot_by_mode):
    parts = []
    for mode in ALL_MODES:
        m = pivot_by_mode[mode].copy()
        m.columns = [f"{mode}@{c:g}" for c in m.columns]
        parts.append(m)
    return pd.concat(parts, axis=1)


def perturbation_profiles(path):
    d = pd.read_csv(path)
    p = paired(d[d["d_hat"] > 0])
    return full_profile({m: profiles(p, m) for m in ALL_MODES})


def generator_profiles(path, min_texts=25):
    d = pd.read_csv(path)
    d = d[d["d_hat"] > 0].copy()
    h = (d[d["is_human"]].groupby(["sub_source", "mode", "q"])["d_hat"]
         .mean().rename("h"))
    d = d.join(h, on=["sub_source", "mode", "q"]).dropna(subset=["h"])
    d["rel"] = (d["d_hat"] - d["h"]) / d["h"] * 100
    n = d[(d["mode"] == "q_small") & (d["q"] == 0)].groupby("model").size()
    keep = n[n >= min_texts].index.difference(["human"])
    piv = {m: d[(d["mode"] == m) & d["model"].isin(keep)]
           .pivot_table(index="model", columns="q", values="rel")
           for m in ALL_MODES}
    return full_profile(piv), n


def main():
    pert = perturbation_profiles(os.path.join(BASE, "results",
                                              "perturb_L201_coling.csv.gz"))
    gen, n = generator_profiles(os.path.join(BASE, "results",
                                             "coling_qphd_L201.csv.gz"))
    cols = [c for c in pert.columns if c in gen.columns]
    X = pd.concat([pert[cols], gen[cols]]).fillna(0)
    kind = np.array(["pert"] * len(pert) + ["gen"] * len(gen))

    mu = X.values.mean(0)
    u, s, vt = np.linalg.svd(X.values - mu, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    P = pd.DataFrame((X.values - mu) @ vt[:3].T, index=X.index,
                     columns=["PC1", "PC2", "PC3"])
    P["kind"] = kind
    P.round(1).to_csv(os.path.join(BASE, "results", "pc_space.csv"))
    print(f"базис на объединении: {len(pert)} пертурбаций + {len(gen)} моделей, "
          f"{len(cols)} координат")
    print("доли дисперсии:", np.round(var[:4], 3))

    for a, b in [("PC1", "PC2"), ("PC2", "PC3")]:
        fig, ax = plt.subplots(figsize=(10.5, 8.5))
        ax.axhline(0, c="k", lw=1)
        ax.axvline(0, c="k", lw=1)
        for name, r in P.iterrows():
            is_gen = r["kind"] == "gen"
            col = ("#c53030" if is_gen else
                   "#b7791f" if name in DEFECTS else
                   "#2b6cb0" if str(name).startswith("style_") else "#4a5568")
            ax.scatter(r[a], r[b], s=58 if is_gen else 40, c=col,
                       marker="^" if is_gen else "o", zorder=3)
            ax.annotate(name, (r[a], r[b]), fontsize=6.4, color=col,
                        xytext=(4, 3), textcoords="offset points")
        i, j = int(a[-1]) - 1, int(b[-1]) - 1
        ax.set_xlabel(f"{a} — {var[i]:.1%} дисперсии")
        ax.set_ylabel(f"{b} — {var[j]:.1%} дисперсии")
        ax.set_title("Модели (треугольники) и пертурбации (круги) "
                     "в общем базисе компонент\n"
                     "жёлтые — дефекты, синие — стили, серые — механические")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = os.path.join(BASE, "figures", f"pc_space_{a}_{b}.png")
        fig.savefig(path, dpi=150)
        print("saved", path)

    pd.set_option("display.width", 200)
    print("\n=== модели")
    print(P[P.kind == "gen"].drop(columns="kind").round(1)
          .sort_values("PC1").to_string())
    print("\n=== пертурбации")
    print(P[P.kind == "pert"].drop(columns="kind").round(1)
          .sort_values("PC1").to_string())


if __name__ == "__main__":
    main()
