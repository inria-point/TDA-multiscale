"""Where on the q axis does "reads as human" separate from "reads as noise"?

The judge scale was built with 5 as the floor: below it a text is closer to
noise than to writing. That makes a natural split, and the question is whether
the two halves part company at particular trimming levels rather than
everywhere at once -- a separation confined to one end of q would say the
distinction lives at one scale of the geometry.

Two figures. The first is the split against the three bands, as a scatter, so
the overlap is visible rather than summarised. The second is the whole q axis:
mean displacement per group at every q of every trimming mode, with a standard
error band, since the interesting thing is where the two curves come apart and
whether the gap is wider than the noise.
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
from judge_shift import load
from three_bands import BANDS as _B, FILES, SUF

BANDS = list(_B)
BASE = os.path.join(HERE, "..")
MODES = ["q_small", "q_large", "q0.5_range"]
MODE_RU = {"q_small": "q_small — остаются длинные рёбра (крупный масштаб)",
           "q_large": "q_large — остаются короткие рёбра (мелкий масштаб)",
           "q0.5_range": "q0.5_range — окно шириной 0.5 (средний масштаб)"}
CUT = 5
LO, HI = "#c53030", "#2b6cb0"          # validated pair, see dataviz validator
LAB = {0: f"судья < {CUT} — ближе к шуму", 1: f"судья ≥ {CUT} — читается как текст"}
MIN_N = 25


def profiles():
    """Relative d at every (mode, q) for every perturbed text."""
    frames = []
    for fn in FILES:
        p = os.path.join(BASE, "results", fn)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d["d_hat"] > 0]
        base = (d[d["perturbation"] == "identity"]
                .set_index(["text_id", "mode", "q"])["d_hat"].rename("d0"))
        if base.empty:
            continue
        s = d[d["perturbation"] != "identity"].join(
            base, on=["text_id", "mode", "q"]).dropna(subset=["d0"])
        s["rel"] = (s["d_hat"] - s["d0"]) / s["d0"] * 100
        frames.append(s[["perturbation", "text_id", "mode", "q", "rel"]])
    P = pd.concat(frames)
    P["text_id"] = P["text_id"].astype(str).str.replace("^coling::", "",
                                                        regex=True)
    return P.drop_duplicates(["perturbation", "text_id", "mode", "q"],
                             keep="last")


def scatter(D, keys):
    fig, axes = plt.subplots(len(BANDS), len(keys), sharey="row",
                             figsize=(3.05 * len(keys), 3.4 * len(BANDS)))
    for i, band in enumerate(BANDS):
        for j, k in enumerate(keys):
            ax = axes[i][j]
            g = D[[k, band]].dropna()
            lo = g[g[k] < CUT]
            hi = g[g[k] >= CUT]
            jit = lambda s: s + np.random.default_rng(0).normal(0, .13, len(s))
            ax.axhline(0, c="#999", lw=.9)
            ax.scatter(jit(lo[k]), lo[band], s=7, c=LO, alpha=.45,
                       linewidths=0)
            ax.scatter(jit(hi[k]), hi[band], s=7, c=HI, alpha=.45,
                       linewidths=0)
            for sub, c in ((lo, LO), (hi, HI)):
                if len(sub) >= MIN_N:
                    # a 2px surface ring keeps the median marker legible where
                    # the two clouds overlap
                    ax.plot([sub[k].min() - .3, sub[k].max() + .3],
                            [sub[band].median()] * 2, c=c, lw=2.6,
                            path_effects=None, solid_capstyle="round",
                            zorder=5)
            d = (hi[band].median() - lo[band].median()) if len(lo) >= MIN_N \
                else np.nan
            ax.set_title(f"{PROPS[k]['ru']}\nразрыв медиан "
                         + (f"{d:+.1f}%" if d == d else "—"), fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xticks(range(1, 11, 2))
            ax.grid(alpha=.2)
            if j == 0:
                ax.set_ylabel(f"{band} масштаб, %", fontsize=9)
            if i == len(BANDS) - 1:
                ax.set_xlabel("оценка судьи", fontsize=8)
    h = [plt.Line2D([], [], marker="o", ls="", color=LO, label=LAB[0]),
         plt.Line2D([], [], marker="o", ls="", color=HI, label=LAB[1])]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False,
               fontsize=10)
    fig.suptitle("Человеческое против шума: три полосы против каждой "
                 "судейской шкалы\nтолстая черта — медиана группы",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.028, 1, 0.945))
    path = os.path.join(BASE, "figures", f"human_split_bands{SUF}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", os.path.relpath(path, BASE))


def curves(P, D, keys):
    J = D.set_index(["perturbation", "text_id"])
    M = P.join(J[keys], on=["perturbation", "text_id"]).dropna(subset=keys,
                                                               how="all")
    fig, axes = plt.subplots(len(keys), len(MODES), sharex="col",
                             figsize=(19, 4.1 * len(keys)))
    for i, k in enumerate(keys):
        for j, mode in enumerate(MODES):
            ax = axes[i][j]
            s = M[(M["mode"] == mode)].dropna(subset=[k])
            s = s.assign(grp=(s[k] >= CUT).astype(int))
            ax.axhline(0, c="#999", lw=1)
            for g, c in ((0, LO), (1, HI)):
                t = s[s["grp"] == g].groupby("q")["rel"]
                m, se, n = t.mean(), t.sem(), t.size()
                m, se = m[n >= MIN_N], se[n >= MIN_N]
                if m.empty:
                    continue
                # the shaded band is +-1 standard error of the mean: where the
                # two shadows stop touching, the split is real at that q
                ax.fill_between(m.index, m - se, m + se, color=c, alpha=.22,
                                linewidth=0)
                ax.plot(m.index, m, color=c, lw=2.2,
                        label=f"{LAB[g]} (n={int(n[n >= MIN_N].mean())})")
            ax.grid(alpha=.22)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(MODE_RU[mode], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{PROPS[k]['ru']}\nсдвиг d, %", fontsize=9)
            if i == len(keys) - 1:
                ax.set_xlabel("q — доля отброшенных рёбер", fontsize=9)
            ax.legend(fontsize=7.5, frameon=False, loc="best")
    fig.suptitle("Где по оси q расходятся «читается как текст» и «ближе к "
                 "шуму»\nлиния — среднее по группе, тень — ±1 стандартная "
                 "ошибка среднего", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.055 / len(keys) * 3))
    path = os.path.join(BASE, "figures", f"human_split_q{SUF}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("saved", os.path.relpath(path, BASE))


def main():
    D = load()
    keys = [k for k in PROPS if k in D and
            (D[k] < CUT).sum() >= MIN_N and (D[k] >= CUT).sum() >= MIN_N]
    print(f"{len(D)} текстов; шкал с обеими группами: {len(keys)}")
    for k in keys:
        print(f"  {PROPS[k]['ru']:26s} < {CUT}: {(D[k] < CUT).sum():4d}   "
              f">= {CUT}: {(D[k] >= CUT).sum():4d}")
    scatter(D, keys)
    curves(profiles(), D, keys)


if __name__ == "__main__":
    main()
