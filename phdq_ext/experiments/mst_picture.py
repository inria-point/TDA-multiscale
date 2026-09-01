"""Draw the MST itself, so the long edges can be seen for what they are.

Every band is a statement about a slice of the edge lengths, and the tables say
what tokens those edges join. What the tables do not say is the *shape*: whether
a long edge is a single point hanging off the tree, or a bridge holding a whole
group on. Those are different geometries and they would mean different things.

Three views, in order of how much they can lie to you.

  дендрограмма   exact. An MST is single-linkage clustering -- the same tree --
                 so a single-linkage dendrogram whose merge height is the edge
                 length IS the MST, redrawn. Cutting at height h removes every
                 edge longer than h. No projection, no distortion.

  что отваливается   exact. Removing one edge splits a tree in two; the size of
                 the smaller side says whether that edge held a point or a
                 cluster. Over all texts this settles the question numerically.

  рисунок дерева   approximate. Kamada-Kawai over the tree's own path metric,
                 which a tree tolerates far better than a general graph, but it
                 is still 768 dimensions pressed into 2 and edge lengths on the
                 page are only roughly the real ones. It is here to look at,
                 not to measure.

Colours are the ones already in use, so the three figures read together: nodes
and dendrogram leaves take the six token classes of token_geometry.py, tree edges
take the thirteen cells of edge_taxonomy.py.

Sinks are marked rather than removed: the point mass and the long bridge the MST
needs to attach it are exactly the kind of structure this picture is for.
"""
import os
import sys
from collections import Counter, deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import (COLORS, COS_CUT, NORM_CUT, SKIP, cell_of,
                           corpus_ranks, token_class)
from embedder import Embedder
from sink_cluster import sink_direction

BASE = os.path.join(HERE, "..")
TEXT = int(os.environ.get("TEXT", 0))
N_TEXTS = int(os.environ.get("N_TEXTS", 120))
CCOL = {"пункт": "#9e9ac8", "служ": "#6baed6", "смысл-част": "#e6550d",
        "смысл-редк": "#08306b", "подслово": "#74c476", "сток": "#000000"}
CLASSES = list(CCOL)


def one_text(text, emb, ranks, u_sink, L=cfg.L_DEFAULT, seed=0):
    """The estimator's own sample, plus the MST on it."""
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    wid = np.cumsum([x.startswith(("Ġ", "▁")) for x in toks])[idx]
    v = e[idx]
    t = [toks[i] for i in idx]
    nrm = np.linalg.norm(v, axis=1)
    sink = ((v @ u_sink) / nrm > COS_CUT) & (nrm < NORM_CUT)
    cls = ["сток" if s_ else token_class(x, ranks) for x, s_ in zip(t, sink)]
    d = squareform(pdist(v))
    m = minimum_spanning_tree(d).tocoo()
    return v, t, cls, wid, d, m


def smaller_side(rows, cols, lens, n):
    """For every tree edge, the size of the smaller of the two halves.

    One BFS gives every node's subtree size under an arbitrary root; an edge to
    a child then splits the tree into that subtree and everything else. O(n),
    against O(n) connected-component calls if each edge were cut separately.
    """
    adj = [[] for _ in range(n)]
    for k, (a, b) in enumerate(zip(rows, cols)):
        adj[a].append((b, k))
        adj[b].append((a, k))
    parent_edge = [-1] * n
    order, seen = [], np.zeros(n, bool)
    q = deque([0])
    seen[0] = True
    while q:
        u = q.popleft()
        order.append(u)
        for w, k in adj[u]:
            if not seen[w]:
                seen[w] = True
                parent_edge[w] = k
                q.append(w)
    size = np.ones(n, int)
    child_of_edge = np.full(len(lens), -1)
    for u in reversed(order):
        k = parent_edge[u]
        if k >= 0:
            child_of_edge[k] = u
    for u in reversed(order):
        k = parent_edge[u]
        if k >= 0:
            # the parent is whichever endpoint of edge k is not u
            p = rows[k] if cols[k] == u else cols[k]
            size[p] += size[u]
    sub = np.array([size[c] if c >= 0 else 1 for c in child_of_edge])
    return np.minimum(sub, n - sub)


def survey(emb, ranks, u_sink, hum):
    """Over many texts: are the long edges leaves or bridges?"""
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        r = one_text(s, emb, ranks, u_sink, seed=i)
        if r is None:
            continue
        _, t, cls, _, _, m = r
        o = np.argsort(m.data)
        lens, ra, ca = m.data[o], m.row[o], m.col[o]
        n = len(t)
        small = smaller_side(ra, ca, lens, n)
        k = int(0.2 * len(lens))
        for j in range(len(lens)):
            rows.append({"текст": i, "перцентиль": j / len(lens) * 100,
                         "длина": lens[j], "отн": lens[j] / lens.mean(),
                         "меньшая сторона": small[j],
                         "длинное": j >= len(lens) - k,
                         "сток на конце": ("сток" in (cls[ra[j]], cls[ca[j]]))})
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{N_TEXTS}", flush=True)
    return pd.DataFrame(rows)


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    ranks = corpus_ranks(hum, emb)
    u_sink, _ = sink_direction(hum[300:340], emb)

    print(f"обзор по {N_TEXTS} текстам: чем держится длинное ребро\n",
          flush=True)
    S = survey(emb, ranks, u_sink, hum)
    S.to_csv(os.path.join(BASE, "results", "mst_cuts.csv"), index=False)
    long = S[S["длинное"]]
    pd.set_option("display.width", 200)
    print("\nразмер меньшей стороны при разрезании ребра, 20% самых длинных\n")
    q = long["меньшая сторона"].value_counts(normalize=True).sort_index() * 100
    tbl = pd.DataFrame({"доля рёбер, %": q}).head(10)
    tbl.index.name = "точек в отваливающейся части"
    print(tbl.round(1).to_string())
    print(f"\nодна точка (лист):          {(long['меньшая сторона'] == 1).mean() * 100:.1f}%")
    print(f"2-5 точек:                  {long['меньшая сторона'].between(2, 5).mean() * 100:.1f}%")
    print(f"больше 5 точек (кластер):   {(long['меньшая сторона'] > 5).mean() * 100:.1f}%")
    print(f"медиана:                    {long['меньшая сторона'].median():.0f}")
    ss = long[long["сток на конце"]]
    if len(ss):
        print(f"\nиз длинных рёбер со стоком на конце ({len(ss) / len(long) * 100:.1f}% всех "
              f"длинных): медиана меньшей стороны {ss['меньшая сторона'].median():.0f}, "
              f"листьев {(ss['меньшая сторона'] == 1).mean() * 100:.0f}%")

    r = one_text(hum[TEXT], emb, ranks, u_sink, seed=TEXT)
    v, t, cls, wid, d, m = r
    draw(v, t, cls, wid, d, m, S, long)


def draw(v, t, cls, wid, d, m, S, long):
    n = len(t)
    o = np.argsort(m.data)
    lens, ra, ca = m.data[o], m.row[o], m.col[o]
    small = smaller_side(ra, ca, lens, n)
    cells = [cell_of(t[a], t[b], cls[a], cls[b], wid[a] == wid[b])
             for a, b in zip(ra, ca)]

    print(f"\n\nтекст {TEXT}: 12 самых длинных рёбер\n")
    tab = []
    for j in range(len(lens) - 1, len(lens) - 13, -1):
        a, b = ra[j], ca[j]
        tab.append({"длина": lens[j], "отн": lens[j] / lens.mean(),
                    "токен A": t[a], "токен B": t[b], "ячейка": cells[j],
                    "отваливается точек": small[j]})
    print(pd.DataFrame(tab).round(2).to_string(index=False))

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])

    # --- dendrogram: exact redrawing of this MST -------------------------
    ax = fig.add_subplot(gs[0, :])
    Z = linkage(squareform(d, checks=False), method="single")
    dd = dendrogram(Z, no_labels=True, color_threshold=0,
                    above_threshold_color="#555555", ax=ax)
    ax.set_ylabel("длина ребра MST")
    ax.set_title(f"Дендрограмма одинарной связи — это тот же MST, текст {TEXT}. "
                 "Высота слияния = длина ребра")
    for p, lab, col in [(40, "граница мелкой полосы", "#08306b"),
                        (50, "начало крупной", "#a63603")]:
        h = np.percentile(lens, p)
        ax.axhline(h, color=col, lw=1.2, ls="--")
        ax.text(0.004, h, f"{lab} ({p}-й перц.)", color=col, va="bottom",
                ha="left", fontsize=9, transform=ax.get_yaxis_transform())
    # a strip of class colours under the leaves, in dendrogram order
    y0 = -0.055 * lens.max()
    ax.scatter([5 + 10 * i for i in range(n)], [y0] * n, marker="s", s=9,
               c=[CCOL[cls[i]] for i in dd["leaves"]], linewidths=0,
               clip_on=False)
    ax.set_ylim(y0 * 1.6, None)
    ax.set_xticks([])
    ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=CCOL[c],
                                  label=c) for c in CLASSES],
              fontsize=8.5, loc="upper left", ncol=2)

    # --- what comes off when an edge is cut ------------------------------
    ax = fig.add_subplot(gs[1, 0])
    bins = np.arange(1, 22) - 0.5
    ax.hist(long["меньшая сторона"].clip(upper=20), bins=bins,
            color="#a63603", edgecolor="white")
    ax.set_xlabel("точек в отваливающейся части (20+ склеены)")
    ax.set_ylabel("рёбер")
    ax.set_title("Разрезаем 20% самых длинных рёбер:\nотваливается точка или "
                 f"кластер? ({S['текст'].nunique()} текстов)")
    ax.axvline(1.5, color="#333", lw=1, ls=":")
    frac = (long["меньшая сторона"] == 1).mean() * 100
    ax.text(0.97, 0.9, f"одиночная точка: {frac:.0f}%", transform=ax.transAxes,
            ha="right", fontsize=11)

    # --- the tree drawn ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, w in zip(ra, ca, lens):
        G.add_edge(int(a), int(b), weight=float(w))
    pos = nx.kamada_kawai_layout(G, weight="weight")
    P = np.array([pos[i] for i in range(n)])
    # edge colour is the taxonomy cell, exactly as in edge_taxonomy.png; edge
    # width is the length, so the two readings do not compete for one channel
    w_scale = 0.5 + 3.0 * (lens / lens.max())
    for k, (a, b) in enumerate(zip(ra, ca)):
        ax.plot(*zip(P[a], P[b]), color=COLORS[cells[k]], lw=w_scale[k],
                zorder=1, solid_capstyle="round")
    deg = np.bincount(np.concatenate([ra, ca]), minlength=n)
    ax.scatter(P[:, 0], P[:, 1], s=12 + 9 * deg, zorder=2, linewidths=0.4,
               edgecolors="white", c=[CCOL[c] for c in cls])
    seen = [c for c in COLORS if c in set(cells)]
    ax.legend(handles=[plt.Line2D([], [], color=COLORS[c], lw=2.5, label=c)
                       for c in seen], fontsize=8, loc="upper left",
              bbox_to_anchor=(1.0, 1.0), title="ребро", title_fontsize=8.5,
              frameon=False)
    ax.set_title("Само дерево (Kamada-Kawai по метрике дерева).\n"
                 "Цвет ребра — ячейка таксономии, толщина — длина")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    fig.tight_layout()
    out = os.path.join(BASE, "figures", f"mst_picture_{TEXT}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nрисунок: {out}")


if __name__ == "__main__":
    main()
