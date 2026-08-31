"""Every property against every band, twice over.

First an overview: the dumbbell shows the pooled correlation as an open circle
and the within-perturbation one as a filled circle, so the arrow length is
exactly the part of the relation that came from comparing different operations
rather than different texts.

Then one panel per property, per band, with each perturbation's own regression
line drawn in its own colour. The overview says how much survives; the panels
say whether what survives is one consistent relation or a cancellation of
slopes pointing different ways.
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
MIN_N = 15
COL = {"судья": "#b7791f", "механика": "#2b6cb0"}


def overview(W):
    W = W.copy()
    W["worst"] = W[[f"{b} общая" for b in BANDS]].abs().max(axis=1)
    W = W.sort_values("worst").reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 10), sharey=True)
    for ax, band in zip(axes, BANDS):
        a, b = W[f"{band} общая"], W[f"{band} внутри"]
        ax.axvline(0, c="k", lw=1)
        for i, (x0, x1, kind) in enumerate(zip(a, b, W["тип"])):
            ax.plot([x0, x1], [i, i], c=COL[kind], lw=1.6, alpha=0.55, zorder=1)
            ax.scatter(x0, i, s=52, facecolors="white", edgecolors=COL[kind],
                       linewidths=1.6, zorder=3)
            ax.scatter(x1, i, s=52, c=COL[kind], zorder=4)
        ax.set_yticks(range(len(W)))
        ax.set_yticklabels(W["признак"], fontsize=8.5)
        ax.set_xlabel("корреляция Спирмена")
        ax.set_title(f"{band} масштаб")
        ax.set_xlim(-0.8, 0.8)
        ax.grid(axis="x", alpha=0.25)

    h = [plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                    markeredgecolor="#555", label="по всем текстам"),
         plt.Line2D([], [], marker="o", ls="", color="#555",
                    label="внутри пертурбаций"),
         plt.Line2D([], [], color=COL["механика"], lw=3, label="механика"),
         plt.Line2D([], [], color=COL["судья"], lw=3, label="судья")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Все признаки × 3 полосы: сколько связи переживает "
                 "поглощение пертурбации", fontsize=12)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    path = os.path.join(BASE, "figures", f"all_props{SUF}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path, f"({len(W)} строк)")


def panels(D, W, band, props):
    """One scatter per property; colour = perturbation, line = its own slope."""
    look = W.set_index("признак")
    big = D["perturbation"].value_counts()
    keep = big[big >= MIN_N].index
    cmap = plt.get_cmap("tab20")
    cols = {p: cmap(i % 20) for i, p in enumerate(sorted(keep))}
    sub = D[D["perturbation"].isin(keep)]

    fig, axes = plt.subplots(4, 5, figsize=(22, 16))
    for ax, (col, ru) in zip(axes.ravel(), props):
        g0 = sub[[band, col, "perturbation"]].dropna()
        for p, g in g0.groupby("perturbation"):
            c = cols[p]
            ax.scatter(g[col], g[band], s=11, color=c, alpha=0.65, linewidths=0)
            if len(g) >= MIN_N and g[col].nunique() > 3:
                b1, b0 = np.polyfit(g[col], g[band], 1)
                xs = np.array([g[col].min(), g[col].max()])
                ax.plot(xs, b0 + b1 * xs, color=c, lw=1.3, alpha=0.9)
        ax.axhline(0, c="k", lw=0.9)
        r = look.loc[ru] if ru in look.index else None
        tag = (f"общая {r[f'{band} общая']:+.2f}   внутри {r[f'{band} внутри']:+.2f}"
               if r is not None else "")
        ax.set_title(f"{ru}\n{tag}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(props):]:
        ax.axis("off")
    fig.suptitle(f"{band.upper()} масштаб (ось Y, %) против 20 признаков. "
                 "Точка — текст, цвет — пертурбация, линия — наклон внутри неё",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = os.path.join(BASE, "figures", f"props_{band}{SUF}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("saved", path)


def main():
    W = pd.read_csv(os.path.join(BASE, "results", f"within_between{SUF}.csv"))
    overview(W)

    D = pd.read_csv(os.path.join(BASE, "results", f"annotated{SUF}.csv"))
    for m in MECH:
        if f"src_{m}" in D:
            D[f"d_{m}"] = D[m] - D[f"src_{m}"]
    # judged properties as scored; mechanical ones as paired shifts, which is
    # the form that survived the within test
    props = [(k, PROPS[k]["ru"]) for k in PROPS if k in D]
    props += [(f"d_{m}", f"{MECH[m]} (сдвиг)") for m in MECH
              if f"d_{m}" in D]
    print(f"{len(props)} признаков на панель")
    for band in BANDS:
        panels(D, W, band, props)


if __name__ == "__main__":
    main()
