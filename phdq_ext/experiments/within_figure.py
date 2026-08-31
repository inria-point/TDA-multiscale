"""Why the middle band survives the within-perturbation test and the coarse one does not.

Each panel plots individual texts, one colour per perturbation, with that
perturbation's own regression line. A relation that exists only between
operations shows as a staircase of flat clouds; one that also exists inside an
operation shows as clouds that are themselves tilted, all in the same
direction.
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
from mech_props import NAMES as MECH
from three_bands import SUF

BASE = os.path.join(HERE, "..")
MIN_N = 15
PANELS = [
    ("средний", "d_word_entropy", "сдвиг энтропии словоупотребления",
     "средняя полоса: внутри 0.36"),
    ("мелкий", "d_trigram_repeat", "сдвиг повтора триграмм",
     "мелкая полоса: внутри -0.21"),
    ("крупный", "d_ttr", "сдвиг разнообразия словаря",
     "крупная полоса: внутри 0.15"),
    ("крупный", "integration", "уместность слов (судья, 1-10)",
     "крупная полоса: внутри 0.02"),
]


def main():
    D = pd.read_csv(os.path.join(BASE, "results", f"annotated{SUF}.csv"))
    for m in MECH:
        if f"src_{m}" in D:
            D[f"d_{m}"] = D[m] - D[f"src_{m}"]

    big = D["perturbation"].value_counts()
    keep = big[big >= MIN_N].index
    cmap = plt.get_cmap("tab20")
    cols = {p: cmap(i % 20) for i, p in enumerate(sorted(keep))}

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11))
    for ax, (band, prop, xlab, title) in zip(axes.ravel(), PANELS):
        sub = D[D["perturbation"].isin(keep)][[band, prop, "perturbation"]].dropna()
        for p, g in sub.groupby("perturbation"):
            c = cols[p]
            ax.scatter(g[prop], g[band], s=15, color=c, alpha=0.7, linewidths=0)
            if len(g) >= MIN_N and g[prop].nunique() > 3:
                b, a = np.polyfit(g[prop], g[band], 1)
                xs = np.array([g[prop].min(), g[prop].max()])
                ax.plot(xs, a + b * xs, color=c, lw=1.6, alpha=0.9)
        ax.axhline(0, c="k", lw=1)
        ax.set_xlabel(xlab)
        ax.set_ylabel(f"{band} масштаб, %")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle("Каждая точка — один текст, каждый цвет — одна пертурбация, "
                 "линия — наклон внутри неё.\nНаклонные и согласованные линии = "
                 "связь держится на уровне текста; плоские = держалась только "
                 "на различии между операциями", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(BASE, "figures", f"within_perturbation{SUF}.png")
    fig.savefig(path, dpi=150)
    print("saved", path, f"({len(keep)} пертурбаций)")


if __name__ == "__main__":
    main()
