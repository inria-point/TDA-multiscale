"""Damage a reader sees against displacement the geometry records.

The two need not agree, and where they disagree is where each measure is
saying something the other cannot. Quiet operations sit bottom-right: a text
that still reads as human while its dimension has moved twenty per cent.
Loud ones sit top-left: a reader calls the text destroyed and the dimension
does not move at all.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from taxonomy import group_of
from three_bands import BANDS as _B, SUF

BANDS = list(_B)
BASE = os.path.join(HERE, "..")
WELL_FORMED = ["coherence", "literacy", "naturalness", "integration"]


def main():
    D = pd.read_csv(os.path.join(BASE, "results", f"annotated{SUF}.csv"))
    g = D.groupby("perturbation")
    P = pd.DataFrame({"n": g.size(), "оценки": g[WELL_FORMED].mean().mean(axis=1),
                      **{b: g[b].mean() for b in BANDS}})
    P["сдвиг"] = P[BANDS].abs().max(axis=1)

    fig, ax = plt.subplots(figsize=(13.5, 9))
    ax.axvline(6.0, c="#888", ls="--", lw=1)
    ax.axhline(10, c="#888", ls="--", lw=1)
    ax.scatter(P["оценки"], P["сдвиг"], s=54, c="#2b6cb0", zorder=3)
    for name, r in P.iterrows():
        ax.annotate(name, (r["оценки"], r["сдвиг"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points", color="#333")
    ax.text(8.6, 33, "тихие:\nчитается как человек,\nгеометрия ушла",
            fontsize=10, color="#1a7f37", ha="right", fontweight="bold")
    ax.text(1.6, 33, "громкие и заметные", fontsize=10, color="#666")
    ax.text(1.6, 1.5, "громкие:\nсудья видит поломку,\nгеометрия на месте",
            fontsize=10, color="#c53030", fontweight="bold")
    ax.set_xlabel("оценка судьи: связность, грамотность, естественность, "
                  "уместность (среднее, 1-10)")
    ax.set_ylabel("максимальный сдвиг по трём полосам, |%|")
    ax.set_title("Что видит читатель против того, что видит геометрия\n"
                 "40 пертурбаций, по 30 текстов каждая")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(BASE, "figures", f"quiet_loud{SUF}.png")
    fig.savefig(path, dpi=150)
    print("saved", path)
    print(P.sort_values("сдвиг", ascending=False).round(1).to_string())


if __name__ == "__main__":
    main()
