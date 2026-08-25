"""The full q-profile of the composition, against its two parts.

The composition was judged at a single q. This draws the whole grid in all
three trimming modes, with the additive prediction -- the sum of the two
parts' displacements -- as a dashed line, so it is visible where addition
holds and where it does not.

text-davinci-003 is drawn for reference: it is the generator the composition
was aimed at, and the only way to see whether the match at q=0.8 is a match of
shape or a crossing of two different curves.
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
from pc_space import ALL_MODES, paired, profiles

BASE = os.path.join(HERE, "..")
TITLES = {"q_small": "q_small — оставлены длинные рёбра (крупный масштаб)",
          "q_large": "q_large — оставлены короткие рёбра (мелкий масштаб)",
          "q0.5_range": "q0.5_range — центральная полоса"}


def pert(path, names):
    d = pd.read_csv(path)
    p = paired(d[d["d_hat"] > 0])
    return {m: profiles(p, m).reindex(names) for m in ALL_MODES}


def generator(name):
    d = pd.read_csv(os.path.join(BASE, "results", "coling_qphd_L201.csv.gz"))
    d = d[d["d_hat"] > 0].copy()
    h = (d[d["is_human"]].groupby(["sub_source", "mode", "q"])["d_hat"]
         .mean().rename("h"))
    d = d.join(h, on=["sub_source", "mode", "q"]).dropna(subset=["h"])
    d["rel"] = (d["d_hat"] - d["h"]) / d["h"] * 100
    d = d[d["model"] == name]
    return {m: d[d["mode"] == m].groupby("q")["rel"].mean()
            for m in ALL_MODES}


def main():
    lit = pert(os.path.join(BASE, "results", "perturb_L201_collapse.csv.gz"),
               ["style_literary_apiyi"])
    ech = pert(os.path.join(BASE, "results", "perturb_L201_pc2neg.csv.gz"),
               ["echo_p25", "echo_p50"])
    com = pert(os.path.join(BASE, "results", "perturb_L201_composed.csv.gz"),
               ["literary_echo25", "literary_echo50"])
    gen = generator("text-davinci-003")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex="col")
    for row, (dose, ename, cname) in enumerate(
            [(25, "echo_p25", "literary_echo25"),
             (50, "echo_p50", "literary_echo50")]):
        for col, mode in enumerate(ALL_MODES):
            ax = axes[row, col]
            q = lit[mode].columns.values
            a = lit[mode].loc["style_literary_apiyi"].values
            b = ech[mode].loc[ename].values
            c = com[mode].loc[cname].values
            ax.axhline(0, c="k", lw=1)
            ax.plot(q, a, "o-", ms=3, c="#2b6cb0", label="художественный стиль")
            ax.plot(q, b, "o-", ms=3, c="#b7791f", label=f"эхо {dose}%")
            ax.plot(q, a + b, "--", lw=2, c="#718096",
                    label="сумма (предсказание)")
            ax.plot(q, c, "o-", ms=4, lw=2.2, c="#c53030",
                    label="композиция (факт)")
            g = gen[mode]
            ax.plot(g.index.values, g.values, ":", lw=2, c="#1a7f37",
                    label="text-davinci-003")
            ax.set_title(TITLES[mode] if row == 0 else "", fontsize=10)
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel(f"эхо {dose}%\nизменение d, %")
            if row == 1:
                ax.set_xlabel("q")
    axes[0, 0].legend(fontsize=8.5, loc="best")
    fig.suptitle("Профиль композиции «художественный стиль + эхо» против "
                 "суммы её частей")
    fig.tight_layout()
    path = os.path.join(BASE, "figures", "compose_profile.png")
    fig.savefig(path, dpi=150)
    print("saved", path)

    pd.set_option("display.width", 200)
    for mode in ALL_MODES:
        a = lit[mode].loc["style_literary_apiyi"]
        b = ech[mode].loc["echo_p25"]
        c = com[mode].loc["literary_echo25"]
        d = pd.DataFrame({"сумма": a + b, "факт": c, "невязка": c - (a + b)})
        print(f"\n=== {mode}, эхо 25%")
        print(d.loc[[q for q in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
                     if q in d.index]].round(1).to_string())


if __name__ == "__main__":
    main()
