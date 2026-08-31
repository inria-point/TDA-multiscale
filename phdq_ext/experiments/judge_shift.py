"""Judged properties as paired shifts, one figure per property.

Pairing was decisive for the mechanical measures -- vocabulary spread went from
0.13 to 0.38 within perturbations once each text was compared against itself
rather than against the corpus. The judge had no such correction available
because only perturbed texts had been scored. Now that the 104 sources are
scored too, every row carries `score - own source's score`, and the same
question can be asked of the judge.

No regression lines here: the point is to see where the mass of texts sits,
and a fitted slope invites reading a trend into a cloud that may not have one.
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
from three_bands import BANDS as _B, SUF

BANDS = list(_B)
BASE = os.path.join(HERE, "..")
MIN_N = 15
OUT = os.path.join(BASE, "figures", f"judge_shift{SUF}")


def load():
    D = pd.read_csv(os.path.join(BASE, "results", f"annotated{SUF}.csv"))
    S = pd.read_csv(os.path.join(BASE, "results", f"judge_sources{SUF}.csv"))
    for f in (D, S):
        f["text_id"] = f["text_id"].astype(str).str.replace("^coling::", "",
                                                            regex=True)
    D = D.merge(S, on="text_id", how="left")
    for k in PROPS:
        if k in D and f"base_{k}" in D:
            D[f"d_{k}"] = D[k] - D[f"base_{k}"]
    return D


def weighted_within(D, col, band):
    """Mean of per-perturbation correlations, weighted by texts used."""
    num = den = 0.0
    for _, g in D.groupby("perturbation"):
        g = g[[col, band]].dropna()
        if len(g) < MIN_N or g[col].nunique() < 3:
            continue
        r = g[col].corr(g[band], method="spearman")
        if pd.notna(r):
            num += r * len(g)
            den += len(g)
    return num / den if den else np.nan


def table(D):
    rows = []
    for k in PROPS:
        for col, tag in ((k, "оценка"), (f"d_{k}", "сдвиг")):
            if col not in D:
                continue
            r = {"признак": PROPS[k]["ru"], "вид": tag}
            for band in BANDS:
                g = D[[col, band]].dropna()
                r[f"{band} общая"] = g[col].corr(g[band], method="spearman")
                r[f"{band} внутри"] = weighted_within(D, col, band)
            rows.append(r)
    T = pd.DataFrame(rows)
    T.round(3).to_csv(os.path.join(BASE, "results",
                                   f"judge_shift{SUF}.csv"), index=False)
    return T


def figures(D, T):
    os.makedirs(OUT, exist_ok=True)
    big = D["perturbation"].value_counts()
    keep = big[big >= MIN_N].index
    cmap = plt.get_cmap("tab20")
    cols = {p: cmap(i % 20) for i, p in enumerate(sorted(keep))}
    sub = D[D["perturbation"].isin(keep)]
    look = T.set_index(["признак", "вид"])

    for k, meta in PROPS.items():
        col = f"d_{k}"
        if col not in sub:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
        for ax, band in zip(axes, BANDS):
            g0 = sub[[band, col, "perturbation"]].dropna()
            for p, g in g0.groupby("perturbation"):
                ax.scatter(g[col], g[band], s=16, color=cols[p], alpha=0.7,
                           linewidths=0)
            ax.axhline(0, c="k", lw=1)
            ax.axvline(0, c="k", lw=1)
            rr = look.loc[(meta["ru"], "сдвиг")]
            ra = look.loc[(meta["ru"], "оценка")]
            ax.set_title(f"{band} масштаб\n"
                         f"сдвиг: общая {rr[f'{band} общая']:+.2f}, "
                         f"внутри {rr[f'{band} внутри']:+.2f}   |   "
                         f"абсолют: {ra[f'{band} общая']:+.2f}", fontsize=9)
            ax.set_xlabel(f"{meta['ru']}: оценка минус оценка исходника")
            ax.set_ylabel(f"{band} масштаб, %")
            ax.grid(alpha=0.25)
        fig.suptitle(f"{meta['ru']} — сдвиг относительно неизменённого текста"
                     f"\nточка — один текст, цвет — пертурбация; "
                     f"перекрестие — исходник", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        name = re.sub(r"[^\w]+", "_", meta["ru"]).strip("_")
        path = os.path.join(OUT, f"{k}_{name}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print("saved", os.path.relpath(path, BASE))


def main():
    D = load()
    T = table(D)
    pd.set_option("display.width", 230)
    print("абсолютная оценка против сдвига относительно исходника\n")
    print(T.round(2).to_string(index=False))
    figures(D, T)


if __name__ == "__main__":
    main()
