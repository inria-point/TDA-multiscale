"""A schematic of the cells-and-frame picture, drawn in the plane.

The report states the arrangement in numbers -- cell radius 0.232 of the mean
pairwise distance, centres 0.864 apart, nearest foreign centre 1.64 sums-of-radii
away -- and those numbers are hard to see. This draws an arrangement with the
same proportions in two dimensions, so the shape of the claim is visible at a
glance.

What is honest here and what is not. The proportions are the measured ones: the
cell radius, the spacing of centres and the gap between neighbouring cells are
set from results/GEOMETRY.md, and the ratio of singletons to cells (94 : 31) is
the measured one too. The positions are invented, the counts are scaled down for
legibility, and the real cloud is about 24-dimensional -- in the plane the same
proportions look far more crowded than they are, because volume grows with
dimension. This is a diagram, not a projection of anything.

Three panels: the arrangement, what the MST makes of it, and what word shuffling
does to it (cells swell by 17%, the frame does not move, the gap falls
1.64 -> 1.34).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

HERE = os.path.dirname(__file__)
BASE = os.path.join(HERE, "..")

# measured, from GEOMETRY.md section 5 (content words)
RADIUS = 0.232          # mean distance of a cell's points to their centroid
NEAREST_CENTRE = 0.789  # to the nearest foreign centre
GAP = NEAREST_CENTRE / (2 * RADIUS)      # = 1.70, reported as 1.64 on average
INFLATE = 1.168         # what shuffling does to the radius
# counts kept in the measured ratio 31 cells : 94 singletons, scaled down
N_CELL, N_SING = 14, 42
KS = [2, 2, 2, 3, 3, 4, 5]

CELL = "#8fb8d6"
SING = "#d7191c"
CENTRE = "#08306b"
EDGE_IN = "#6baed6"
EDGE_STALK = "#d7191c"
EDGE_FRAME = "#999999"


def layout(rng, r=RADIUS):
    """Centres at least NEAREST_CENTRE apart, then points scattered in each."""
    want = N_CELL + N_SING
    cen, tries = [], 0
    while len(cen) < want and tries < 200000:
        tries += 1
        p = rng.uniform(0, 6, 2)
        if all(np.hypot(*(p - q)) >= NEAREST_CENTRE for q in cen):
            cen.append(p)
    cen = np.array(cen)
    rng.shuffle(cen)
    multi, single = cen[:N_CELL], cen[N_CELL:]
    pts, owner = [], []
    for j, c in enumerate(multi):
        k = KS[j % len(KS)]
        # sigma set so the mean distance to the centroid is r
        d = rng.normal(scale=r / 0.89, size=(k, 2))
        d -= d.mean(0)
        pts.append(c + d)
        owner += [j] * k
    pts.append(single)
    owner += [-1] * len(single)
    return np.vstack(pts), np.array(owner), multi, single


# 1.5 radii: two such circles span 3 r, and the nearest centre is 3.4 r away,
# so they visibly do not touch -- which is the claim the picture is making. At
# 2.2 r they would overlap and contradict it.
CIRCLE = 1.5


def draw_cells(ax, multi, r, alpha=0.15):
    for c in multi:
        ax.add_patch(plt.Circle(c, r * CIRCLE, fc=CENTRE, ec="none",
                                alpha=alpha, zorder=0))


def panel(ax, X, owner, multi, single, r, title, mst=False, circles=True):
    if circles:
        draw_cells(ax, multi, r)
    if mst:
        m = minimum_spanning_tree(squareform(pdist(X))).tocoo()
        for a, b, w in zip(m.row, m.col, m.data):
            same = owner[a] == owner[b] and owner[a] >= 0
            stalk = owner[a] == -1 or owner[b] == -1
            col, lw = ((EDGE_IN, 2.4) if same else
                       (EDGE_STALK, 1.6) if stalk else (EDGE_FRAME, 1.0))
            ax.plot(*zip(X[a], X[b]), color=col, lw=lw, zorder=1,
                    solid_capstyle="round")
    ax.scatter(X[owner >= 0, 0], X[owner >= 0, 1], s=34, c=CELL, zorder=3,
               linewidths=0.6, edgecolors="white")
    ax.scatter(single[:, 0], single[:, 1], s=40, c=SING, zorder=4,
               linewidths=0.6, edgecolors="white")
    ax.scatter(multi[:, 0], multi[:, 1], s=58, facecolors="none",
               edgecolors=CENTRE, linewidths=1.6, zorder=5)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    rng = np.random.default_rng(7)
    X, owner, multi, single = layout(rng)
    Xi = X.copy()
    for j in range(N_CELL):
        m = owner == j
        Xi[m] = multi[j] + (X[m] - multi[j]) * INFLATE

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.2))
    panel(ax[0], X, owner, multi, single, RADIUS,
          "Устройство: ячейки и каркас")
    panel(ax[1], X, owner, multi, single, RADIUS,
          "Что из этого делает MST", mst=True, circles=False)
    panel(ax[2], Xi, owner, multi, single, RADIUS * INFLATE,
          "После перемешивания слов", mst=True)

    lim = [(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5),
           (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)]
    for a in ax:
        a.set_xlim(*lim[0])
        a.set_ylim(*lim[1])

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=CELL, ms=9,
                   label="вхождение повторяющегося токена"),
        plt.Line2D([], [], marker="o", ls="", color=SING, ms=9,
                   label="однократный токен (ячейка из одной точки)"),
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec=CENTRE, mew=1.6,
                   ms=10, label="центр ячейки"),
        plt.Line2D([], [], color=EDGE_IN, lw=3,
                   label="ребро внутри ячейки — мелкая полоса"),
        plt.Line2D([], [], color=EDGE_STALK, lw=2.2,
                   label="черешок одиночки — крупная полоса"),
        plt.Line2D([], [], color=EDGE_FRAME, lw=1.6,
                   label="ребро между ячейками"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.5,
               frameon=False, bbox_to_anchor=(0.5, 0.105))
    fig.text(0.5, 0.055,
             f"Пропорции измеренные: радиус ячейки {RADIUS}, до ближайшего "
             f"чужого центра {NEAREST_CENTRE}, зазор {GAP:.2f} радиусов "
             f"(круг нарисован в {CIRCLE} радиуса); "
             f"одиночек к ячейкам 3:1. Справа ячейки раздуты на "
             f"{(INFLATE - 1) * 100:.0f}%, центры не сдвинуты.",
             ha="center", fontsize=9.5, color="#555")
    fig.text(0.5, 0.028,
             "Расположение выдумано, счёт уменьшен для читаемости, настоящее "
             "облако ~24-мерно — на плоскости те же пропорции выглядят теснее, "
             "чем есть. Это схема, а не проекция.",
             ha="center", fontsize=8.5, color="#999")
    fig.subplots_adjust(bottom=0.26, top=0.93, wspace=0.04)
    out = os.path.join(BASE, "figures", "frame_schematic.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"рисунок: {out}")


if __name__ == "__main__":
    main()
