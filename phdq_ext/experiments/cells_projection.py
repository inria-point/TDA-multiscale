"""The cells, on real data: a few content-word types projected into the plane.

frame_schematic.py draws the arrangement with the measured proportions but
invented positions. This does the opposite -- real vectors from a real text,
nothing invented -- at the cost of a projection.

The projection is metric MDS on the actual distances, not t-SNE. t-SNE optimises
neighbourhoods and is free to move things apart, which is exactly the property
that would fake the claim being made here: that cells sit apart from each other
with clear space between. MDS minimises the distortion of the distances
themselves, and three measures of faithfulness are printed on the figure.

One of them is a warning. The correlation between a cell's true radius and its
drawn radius swings from +0.86 to -0.25 across small changes in how many types
are shown -- so which cell looks loosest on the page is not a fact about the
data, and no claim about relative cell size may be read off this picture. What
does survive is the grouping: in the real space a point's nearest neighbour is
its own type 99% of the time, and the projection keeps that at about 90%.

k >= 2 was tried and does not work: with 26 cells the radius correlation falls
to zero and the nearest-neighbour agreement to 74%, because two-point cells have
no internal structure for MDS to preserve and it scatters them across the page.

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
N_TYPES = int(os.environ.get("N_TYPES", 10))
N_SING = int(os.environ.get("N_SING", 30))
N_LABEL = int(os.environ.get("N_LABEL", 8))
SING_COL = "#b0b0b0"
ACCENT_WARN = "#a63603"


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
    # three separate questions, and they do not have the same answer
    scale = pdist(V).mean()
    ra, rb = [], []
    for x in types:
        sel = np.array([y == x for y in lab])
        ra.append(np.linalg.norm(V[sel] - V[sel].mean(0), axis=1).mean() / scale)
        rb.append(np.linalg.norm(XY[sel] - XY[sel].mean(0), axis=1).mean()
                  / pdist(XY).mean())
    rad_r = np.corrcoef(ra, rb)[0, 1]

    def own_nn(M):
        DD = squareform(pdist(M))
        np.fill_diagonal(DD, np.inf)
        L = np.array([str(x) for x in lab])
        m = L != "None"
        return (L[DD.argmin(1)][m] == L[m]).mean() * 100

    nn_true, nn_drawn = own_nn(V), own_nn(XY)
    print(f"верность: расстояния {faith:.3f}, радиус ячейки {rad_r:+.3f}, "
          f"сосед своего типа {nn_true:.0f}% -> {nn_drawn:.0f}%")

    fig, ax = plt.subplots(figsize=(11, 8.6))
    m = np.array([x is None for x in lab])
    ax.scatter(XY[m, 0], XY[m, 1], s=42, c=SING_COL, zorder=2, linewidths=0.5,
               edgecolors="white", label="однократное слово")
    # tab10 index 7 is grey and would collide with the singletons, which are
    # the one colour that must stay unambiguous; index 8 is an olive that is
    # hard to read on white. With many cells the palette repeats, so each cell
    # also gets spokes from its centroid: the grouping then reads off the lines
    # and does not depend on colours being unique.
    base = [c for q, c in enumerate(plt.get_cmap("tab10").colors)
            if q not in (7, 8)]
    if len(types) > len(base):
        base += [c for q, c in enumerate(plt.get_cmap("tab20").colors)
                 if q not in (14, 15, 16, 17)]
    for q, x in enumerate(types):
        sel = np.array([y == x for y in lab])
        col = base[q % len(base)]
        c = XY[sel].mean(0)
        for pt in XY[sel]:
            ax.plot([c[0], pt[0]], [c[1], pt[1]], color=col, lw=1.1,
                    alpha=0.75, zorder=1)
        ax.scatter(XY[sel, 0], XY[sel, 1], s=52, color=col, zorder=3,
                   linewidths=0.6, edgecolors="white")
        ax.scatter(*c, marker="X", s=150, color=col, zorder=5, linewidths=1.3,
                   edgecolors="white")
        if q < N_LABEL:
            ax.annotate(x.lstrip("Ġ▁"), c, (0, 20), textcoords="offset points",
                        fontsize=11.5, color=col, ha="center", weight="bold",
                        zorder=6,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec="none", alpha=0.75))
    ax.scatter([], [], marker="X", s=140, color="#444", label="центр ячейки")
    ax.legend(fontsize=10.5, loc="best", frameon=False)
    ax.set_title(f"Ячейки смысловых слов (k ≥ {MIN_K}), текст {ti}: "
                 f"{len(types)} типов, центр — крестом, спицы — принадлежность "
                 "ячейке", fontsize=13, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.5, 0.062,
             "Метрический MDS по настоящим расстояниям, не t-SNE: t-SNE волен "
             "растаскивать группы, а здесь проверяется именно то, что они и так "
             "порознь.  Верность: расстояния "
             f"{faith:.2f} (Спирмен); ближайший сосед — свой тип в "
             f"{nn_true:.0f}% случаев в настоящем пространстве и "
             f"{nn_drawn:.0f}% на рисунке.",
             ha="center", fontsize=9, color="#666")
    fig.text(0.5, 0.032,
             "Размер ячейки проекция НЕ передаёт: корреляция настоящего радиуса "
             f"с нарисованным здесь {rad_r:+.2f}, но она скачет от +0.86 до "
             "−0.25 при смене числа показанных типов. Какая ячейка на вид "
             "рыхлее — не факт о данных.",
             ha="center", fontsize=8.8, color=ACCENT_WARN)
    fig.text(0.5, 0.006,
             "Однократные слова — это ячейки из одной точки: своего центра у "
             "них нет, ближайший сосед всегда чужой.",
             ha="center", fontsize=9, color="#666")
    fig.subplots_adjust(bottom=0.15, top=0.94)
    # the filename carries the setting: re-running with a different k must not
    # silently overwrite the picture made with the previous one
    tag = "" if (MIN_K, N_TYPES) == (3, 10) else f"_k{MIN_K}_n{N_TYPES}"
    out = os.path.join(BASE, "figures", f"cells_projection{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"рисунок: {out}")


if __name__ == "__main__":
    main()
