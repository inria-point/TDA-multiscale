"""What the per-text annotation establishes, in three summaries.

First, how much of each band the two families of property explain, pooled and
then with each perturbation given its own intercept. The second figure is the
one that matters: absorbing the perturbation into the model removes everything
that distinguishes the operations from each other and leaves only what varies
between texts given the same operation.

Second, the same decomposition property by property.

Third, the texts themselves in the band space, coloured by a judged property,
so that a claim about the space can be looked at rather than only tabulated.
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
from judge import PROPS
from mech_props import NAMES as MECH
from three_bands import BANDS, SUF

BASE = os.path.join(HERE, "..")


def r2(X, y):
    X = np.c_[np.ones(len(X)), X]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()


def demean(df, cols, by):
    """Subtract each perturbation's own mean: what is left varies text to text."""
    out = df.copy()
    for c in cols:
        out[c] = df[c] - df.groupby(by)[c].transform("mean")
    return out


def main():
    D = pd.read_csv(os.path.join(BASE, "results", f"annotated{SUF}.csv"))
    # judged properties enter as paired shifts too: scoring the 104 sources
    # made the same correction available that pairing gave the counted measures
    S = pd.read_csv(os.path.join(BASE, "results", f"judge_sources{SUF}.csv"))
    for f in (D, S):
        f["text_id"] = f["text_id"].astype(str).str.replace("^coling::", "",
                                                            regex=True)
    D = D.merge(S, on="text_id", how="left")
    for k in PROPS:
        if k in D and f"base_{k}" in D:
            D[f"d_{k}"] = D[k] - D[f"base_{k}"]
    for m in MECH:
        if f"src_{m}" in D:
            D[f"d_{m}"] = D[m] - D[f"src_{m}"]
    mech = [f"d_{m}" for m in MECH if f"d_{m}" in D]
    judge = [f"d_{k}" for k in PROPS if f"d_{k}" in D]

    rows = []
    for band in BANDS:
        sub = D[[band, "perturbation"] + mech + judge].dropna()
        y = sub[band].values
        pooled = {
            "механика": r2(sub[mech].values, y),
            "судья": r2(sub[judge].values, y),
            "вместе": r2(sub[mech + judge].values, y),
        }
        dm = demean(sub, [band] + mech + judge, "perturbation")
        yd = dm[band].values
        within = {
            "механика": r2(dm[mech].values, yd),
            "судья": r2(dm[judge].values, yd),
            "вместе": r2(dm[mech + judge].values, yd),
        }
        rows.append({"полоса": band, "n": len(sub),
                     **{f"вся выборка: {k}": v for k, v in pooled.items()},
                     **{f"внутри пертурбаций: {k}": v
                        for k, v in within.items()}})
    S = pd.DataFrame(rows).set_index("полоса")
    S.round(3).to_csv(os.path.join(BASE, "results", f"summary_r2{SUF}.csv"))
    pd.set_option("display.width", 210)
    print("доля объяснённой дисперсии (R²)\n")
    print(S.round(2).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    pairs = [("крупный", "мелкий"), ("крупный", "средний"),
             ("мелкий", "средний")]
    colour = "integration"
    ok = D[[*BANDS, colour]].dropna()
    for ax, (a, b) in zip(axes, pairs):
        ax.axhline(0, c="k", lw=1)
        ax.axvline(0, c="k", lw=1)
        s = ax.scatter(ok[a], ok[b], c=ok[colour], s=13, cmap="RdYlGn",
                       vmin=1, vmax=10, alpha=0.75, linewidths=0)
        ax.scatter(0, 0, s=340, marker="*", c="#1a7f37", zorder=5,
                   edgecolors="black", linewidths=0.8)
        ax.set_xlabel(f"{a}, %")
        ax.set_ylabel(f"{b}, %")
        ax.grid(alpha=0.25)
    fig.colorbar(s, ax=axes, label=PROPS[colour]["ru"] + " (судья, 1-10)",
                 fraction=0.02)
    fig.suptitle(f"{len(ok)} отдельных текстов в пространстве полос; "
                 "звезда — неизменённый текст")
    path = os.path.join(BASE, "figures", f"per_text_bands{SUF}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("\nsaved", path)


if __name__ == "__main__":
    main()
