"""Which judged shifts land the bands reliably on one side of zero.

A correlation asks whether the two move together across the whole range. That
is not the question here. The question is whether some particular value of a
shift -- "the judge marked this text down two points for word aptness" -- picks
out a set of texts whose band displacement is one-signed, and by how much.

So each (property, shift value) is treated as a cell and described by the
median displacement, the interquartile range, and the share of texts on the
majority side. A cell counts as characteristic when it holds enough texts, the
median is displaced by more than a few per cent, and the majority side holds
at least three texts in four -- all three, since a large median with an even
split is one long tail, and a lopsided split at a median of one per cent is a
reliable statement about nothing.
"""
import os
import re
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
from judge_shift import load
from three_bands import BANDS as _B, SUF

BANDS = list(_B)
BASE = os.path.join(HERE, "..")
MIN_N = 25          # cells smaller than this are not described
MIN_MED = 6.0       # per cent; below this the displacement is not worth naming
MIN_SIDE = 0.72     # share of texts on the majority side


def cells(D):
    rows = []
    for k, meta in PROPS.items():
        col = f"d_{k}"
        if col not in D:
            continue
        for val, g in D.groupby(col):
            if len(g) < MIN_N:
                continue
            for band in BANDS:
                v = g[band].dropna()
                if len(v) < MIN_N:
                    continue
                med = v.median()
                side = max((v > 0).mean(), (v < 0).mean())
                rows.append({
                    "признак": meta["ru"], "ключ": k, "сдвиг": int(val),
                    "полоса": band, "n": len(v),
                    "медиана": med, "среднее": v.mean(),
                    "кв25": v.quantile(0.25), "кв75": v.quantile(0.75),
                    "одна сторона": side,
                    "перевес": "вверх" if med > 0 else "вниз",
                })
    return pd.DataFrame(rows)


def main():
    D = load()
    C = cells(D)
    C.round(2).to_csv(os.path.join(BASE, "results", f"shift_cells{SUF}.csv"),
                      index=False)
    strong = C[(C["медиана"].abs() >= MIN_MED) &
               (C["одна сторона"] >= MIN_SIDE)].copy()
    strong = strong.reindex(strong["медиана"].abs()
                            .sort_values(ascending=False).index)
    pd.set_option("display.width", 210)
    pd.set_option("display.max_rows", 200)
    print(f"ячеек всего: {len(C)} (n >= {MIN_N}); "
          f"характерных: {len(strong)}\n")
    print(strong[["признак", "сдвиг", "полоса", "n", "медиана", "кв25", "кв75",
                  "одна сторона", "перевес"]].round(1).to_string(index=False))

    print("\n\nпризнаки, у которых есть характерные ячейки:")
    print(strong.groupby(["признак", "полоса"])
          .agg(ячеек=("сдвиг", "size"),
               сдвиги=("сдвиг", lambda s: ", ".join(map(str, sorted(s)))))
          .to_string())

    figure(D, C, strong)


def figure(D, C, strong):
    """Rows are the properties that produced characteristic cells."""
    props = [k for k in PROPS
             if PROPS[k]["ru"] in set(strong["признак"])]
    if not props:
        print("нечего рисовать")
        return
    fig, axes = plt.subplots(len(props), 3, sharex="row",
                             figsize=(16.5, 3.3 * len(props)), squeeze=False)
    mark = {(r["ключ"], r["сдвиг"], r["полоса"]) for _, r in strong.iterrows()}
    for i, k in enumerate(props):
        col = f"d_{k}"
        vals = sorted(v for v, g in D.groupby(col) if len(g) >= MIN_N)
        for j, band in enumerate(BANDS):
            ax = axes[i][j]
            ax.axhline(0, c="k", lw=1)
            data = [D[D[col] == v][band].dropna().values for v in vals]
            bp = ax.boxplot(data, positions=range(len(vals)), widths=0.62,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color="black", lw=1.6))
            for v, box in zip(vals, bp["boxes"]):
                hit = (k, int(v), band) in mark
                box.set_facecolor("#dd8452" if hit else "#c8d3e0")
                box.set_edgecolor("#8a4b1f" if hit else "#7b8794")
                box.set_linewidth(1.5 if hit else 0.9)
            for x, v in enumerate(vals):
                n = len(D[D[col] == v][band].dropna())
                ax.annotate(f"n={n}", (x, 0), xytext=(0, -3),
                            textcoords="offset points", ha="center",
                            va="top", fontsize=6.5, color="#555")
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels([f"{v:+.0f}" for v in vals], fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            if i == 0:
                ax.set_title(f"{band} масштаб", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{PROPS[k]['ru']}\n{band} масштаб, %",
                              fontsize=9)
            else:
                ax.set_ylabel(f"{band} масштаб, %", fontsize=8)
            if i == len(props) - 1:
                ax.set_xlabel("оценка минус оценка исходника")
    fig.suptitle("Распределение сдвига по полосам внутри каждого значения "
                 "судейского сдвига\nоранжевым — ячейки, где медиана дальше "
                 f"{MIN_MED:.0f}% от нуля и не меньше {MIN_SIDE:.0%} текстов "
                 "лежит на одной стороне", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = os.path.join(BASE, "figures", f"shift_cells{SUF}.png")
    fig.savefig(path, dpi=140)
    print("\nsaved", os.path.relpath(path, BASE))


if __name__ == "__main__":
    main()
