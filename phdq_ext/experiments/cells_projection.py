"""The cells, on real data: a few content-word types projected into the plane.

frame_schematic.py draws the arrangement with the measured proportions but
invented positions. This does the opposite -- real vectors from a real text,
nothing invented -- at the cost of a projection.

The projection is metric MDS on the actual distances, not t-SNE. t-SNE optimises
neighbourhoods and is free to move things apart, which is exactly the property
that would fake the claim being made here: that cells sit apart from each other
with clear space between. MDS minimises the distortion of the distances
themselves, and the Spearman correlation between drawn and true distances is
printed on the figure so the reader can see how much to trust it.

Shown: every content-word type of the text with at least three occurrences, each
in its own colour, its centroid as a cross; plus a sample of once-used content
words in grey, which are cells of one point and have no centroid to mark.
"""
import os
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import MDS

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from edge_taxonomy import COS_CUT, NORM_CUT, SKIP, corpus_ranks, token_class
from embedder import Embedder
from sink_cluster import sink_direction

BASE = os.path.join(HERE, "..")
CONTENT = {"смысл-част", "смысл-редк"}
MIN_K = int(os.environ.get("MIN_K", 3))
N_TYPES = int(os.environ.get("N_TYPES", 8))
N_SING = int(os.environ.get("N_SING", 45))
SING_COL = "#b0b0b0"


def collect(text, emb, ranks, u_sink, rng):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    v, t = e[keep], [toks[i] for i in keep]
    nrm = np.linalg.norm(v, axis=1)
    sink = ((v @ u_sink) / nrm > COS_CUT) & (nrm < NORM_CUT)
    ok = [j for j in range(len(t))
          if not sink[j] and token_class(t[j], ranks) in CONTENT]
    v, t = v[ok], [t[j] for j in ok]
    where = defaultdict(list)
    for j, x in enumerate(t):
        where[x].append(j)
    multi = sorted([(len(ix), x) for x, ix in where.items() if len(ix) >= MIN_K],
                   reverse=True)[:N_TYPES]
    singles = [ix[0] for x, ix in where.items() if len(ix) == 1]
    if len(multi) < 4 or len(singles) < 10:
        return None
    rng.shuffle(singles)
    singles = singles[:N_SING]
    idx, lab = [], []
    for _, x in multi:
        for j in where[x]:
            idx.append(j)
            lab.append(x)
    for j in singles:
        idx.append(j)
        lab.append(None)
    return v[idx], lab, [x for _, x in multi]


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    ranks = corpus_ranks(hum, emb)
    u_sink, _ = sink_direction(hum[300:340], emb)
    rng = np.random.default_rng(0)

    best = None
    for i in [50, 86, 96, 119, 30, 78, 0, 12]:
        r = collect(hum[i], emb, ranks, u_sink, rng)
        if r and (best is None or len(r[2]) > len(best[1][2])):
            best = (i, r)
    if best is None:
        raise SystemExit("не нашлось текста с достаточным числом повторов")
    ti, (V, lab, types) = best
    print(f"текст {ti}: {len(types)} типов с k>={MIN_K}, "
          f"{sum(x is None for x in lab)} одиночек, {len(V)} точек")

    D = squareform(pdist(V))
    XY = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
             normalized_stress=False, n_init=8).fit_transform(D)
    iu = np.triu_indices(len(V), 1)
    faith = spearmanr(squareform(pdist(XY))[iu], D[iu]).statistic
    print(f"верность проекции: Спирмен {faith:.3f}")

    fig, ax = plt.subplots(figsize=(11, 8.6))
    m = np.array([x is None for x in lab])
    ax.scatter(XY[m, 0], XY[m, 1], s=42, c=SING_COL, zorder=2, linewidths=0.5,
               edgecolors="white", label="однократное слово")
    # tab10 index 7 is grey and would collide with the singletons, which are
    # the one colour that must stay unambiguous; index 8 is an olive that is
    # hard to read on white
    palette = [c for q, c in enumerate(plt.get_cmap("tab10").colors)
               if q not in (7, 8)]
    for q, x in enumerate(types):
        sel = np.array([y == x for y in lab])
        col = palette[q % len(palette)]
        ax.scatter(XY[sel, 0], XY[sel, 1], s=64, color=col, zorder=3,
                   linewidths=0.6, edgecolors="white")
        c = XY[sel].mean(0)
        ax.scatter(*c, marker="X", s=190, color=col, zorder=5, linewidths=1.4,
                   edgecolors="white")
        ax.annotate(x.lstrip("Ġ▁"), c, (0, 22), textcoords="offset points",
                    fontsize=12.5, color=col, ha="center", weight="bold",
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.75))
    ax.scatter([], [], marker="X", s=140, color="#444", label="центр ячейки")
    ax.legend(fontsize=10.5, loc="best", frameon=False)
    ax.set_title(f"Ячейки смысловых слов, текст {ti}: каждый тип своим цветом, "
                 "центр — крестом", fontsize=13, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.5, 0.045,
             "Метрический MDS по настоящим расстояниям, не t-SNE: t-SNE волен "
             "растаскивать группы, а здесь проверяется именно то, что они и так "
             f"порознь. Верность проекции: Спирмен {faith:.2f} между "
             "нарисованным и настоящим расстоянием.",
             ha="center", fontsize=9, color="#666", wrap=True)
    fig.text(0.5, 0.012,
             "Однократные слова — это ячейки из одной точки: своего центра у "
             "них нет, ближайший сосед всегда чужой.",
             ha="center", fontsize=9, color="#666")
    fig.subplots_adjust(bottom=0.12, top=0.94)
    out = os.path.join(BASE, "figures", "cells_projection.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"рисунок: {out}")


if __name__ == "__main__":
    main()
