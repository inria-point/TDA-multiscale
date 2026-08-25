"""What each principal component means in q-space.

A component is a weighting of the 49 profile coordinates (three trimming modes
x q). Plotting that weighting against q says what shape of dimension response
the axis measures -- flat, edge-only, or a sign change across scales -- and
that is what a text property has to be matched against. Interpreting positions
on the map without this is reading a chart with unlabelled axes.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pc_space import ALL_MODES, generator_profiles, perturbation_profiles

BASE = os.path.join(os.path.dirname(__file__), "..")


def main():
    pert = perturbation_profiles(os.path.join(BASE, "results",
                                              "perturb_L201_coling_all.csv.gz"))
    gen, _ = generator_profiles(os.path.join(BASE, "results",
                                             "coling_qphd_L201.csv.gz"))
    cols = [c for c in pert.columns if c in gen.columns]
    X = pd.concat([pert[cols], gen[cols]]).fillna(0)
    _, s, vt = np.linalg.svd(X.values, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()

    L = pd.DataFrame(vt[:3].T, index=cols, columns=["PC1", "PC2", "PC3"])
    L["mode"] = [c.split("@")[0] for c in L.index]
    L["q"] = [float(c.split("@")[1]) for c in L.index]
    L.round(4).to_csv(os.path.join(BASE, "results", "pc_loadings.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, pc in zip(axes, ["PC1", "PC2", "PC3"]):
        ax.axhline(0, c="k", lw=1)
        for mode, c in zip(ALL_MODES, ["#2b6cb0", "#c53030", "#2f855a"]):
            m = L[L["mode"] == mode].sort_values("q")
            ax.plot(m["q"], m[pc], "o-", c=c, ms=4, label=mode)
        ax.set_title(f"{pc} — {var[int(pc[-1]) - 1]:.1%} дисперсии")
        ax.set_xlabel("q")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("вес координаты в компоненте")
    axes[0].legend(fontsize=9)
    fig.suptitle("Из чего складываются компоненты: вес каждого (режим, q) "
                 "в профиле")
    fig.tight_layout()
    path = os.path.join(BASE, "figures", "pc_loadings.png")
    fig.savefig(path, dpi=150)
    print("saved", path)

    pd.set_option("display.width", 200)
    for pc in ["PC1", "PC2", "PC3"]:
        piv = L.pivot_table(index="q", columns="mode", values=pc)
        print(f"\n=== {pc}")
        print(piv.loc[[0.0, 0.2, 0.4, 0.6, 0.8, 0.9]].round(3).to_string())


if __name__ == "__main__":
    main()
