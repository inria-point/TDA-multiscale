"""Models and perturbations in one principal-component basis.

Both are expressed the same way -- per cent change of d against a human
baseline at every (mode, q) -- so they are commensurable and share a basis,
which makes "does this generator look like that perturbation" a geometric
question rather than something read off a chart by eye.

Two choices matter and were both arrived at the hard way.

The full profile is used: all three trimming modes, 49 coordinates. Fitting on
a subset shifts the components materially -- dropping q0.5_range moved PC1 from
96% to 93% and inflated PC3 sixfold.

There is no centring. Human text is the baseline every profile is measured
against, so the zero vector already means "unchanged" -- a meaningful origin
that ordinary PCA discards by re-centring on the mean of whatever happens to be
in the set. With centring, adding the loop series moved the origin by a third
of the cloud radius and flipped the sign of every weak perturbation while the
axes themselves turned by only 5 degrees; without it the same points move by
about one unit.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

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


def colour(name, is_gen):
    if is_gen:
        return "#c53030"
    if name in DEFECTS or "loop" in name:
        return "#b7791f"
    if str(name).startswith("style_"):
        return "#2b6cb0"
    if "ngram" in name:
        return "#6b46c1"
    return "#4a5568"


def main():
    pert = perturbation_profiles(os.path.join(BASE, "results",
                                              "perturb_L201_coling_all.csv.gz"))
    gen, n = generator_profiles(os.path.join(BASE, "results",
                                             "coling_qphd_L201.csv.gz"))
    cols = [c for c in pert.columns if c in gen.columns]
    X = pd.concat([pert[cols], gen[cols]]).fillna(0)
    kind = np.array(["pert"] * len(pert) + ["gen"] * len(gen))

    u, s, vt = np.linalg.svd(X.values, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    P = pd.DataFrame(X.values @ vt[:3].T, index=X.index,
                     columns=["PC1", "PC2", "PC3"])
    P["kind"] = kind
    P.round(1).to_csv(os.path.join(BASE, "results", "pc_space.csv"))
    print(f"базис: {len(pert)} пертурбаций + {len(gen)} моделей, "
          f"{len(cols)} координат")
    print("доли дисперсии:", np.round(var[:4], 3))

    for a, b in [("PC1", "PC2"), ("PC2", "PC3")]:
        i, j = int(a[-1]) - 1, int(b[-1]) - 1
        fig, ax = plt.subplots(figsize=(13, 10))
        ax.axhline(0, c="k", lw=1)
        ax.axvline(0, c="k", lw=1)
        ax.scatter(0, 0, s=700, marker="*", c="#1a7f37",
                   edgecolors="black", linewidths=0.9, zorder=6)
        texts = [ax.text(0, 0, "человек", fontsize=10, color="#1a7f37",
                         fontweight="bold")]
        for name, r in P.iterrows():
            is_gen = r["kind"] == "gen"
            c = colour(name, is_gen)
            ax.scatter(r[a], r[b], s=62 if is_gen else 42, c=c,
                       marker="^" if is_gen else "o", zorder=3)
            texts.append(ax.text(r[a], r[b], name, fontsize=6.8, color=c))
        # without repulsion the centre of the cloud is unreadable, and that is
        # where most of the perturbations sit
        adjust_text(texts, ax=ax, expand=(1.2, 1.4), force_text=(0.4, 0.6),
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4))
        ax.set_xlabel(f"{a} — {var[i]:.1%} дисперсии")
        ax.set_ylabel(f"{b} — {var[j]:.1%} дисперсии")
        ax.set_title("Модели (треугольники) и пертурбации (круги); "
                     "звезда — неизменённый человеческий текст\n"
                     "жёлтые — зацикливания и дефекты, синие — стили, "
                     "фиолетовые — н-граммы, серые — прочие механические")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = os.path.join(BASE, "figures", f"pc_space_{a}_{b}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("saved", path)

    pd.set_option("display.width", 200)
    print("\n=== модели")
    print(P[P.kind == "gen"].drop(columns="kind").round(1)
          .sort_values("PC1").to_string())


if __name__ == "__main__":
    main()
