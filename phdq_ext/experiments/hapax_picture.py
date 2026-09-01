"""One picture for one claim -- but not the claim it started as.

The image to test was: the cloud has a body and the hapaxes stick out of it.
Measured over 11 779 vertices, that is false. A hapax is no further from the
centre of its cloud than a repeated token (22.55 against 23.08, Cohen d = -0.18,
and in the wrong direction), and no further from everything else on average
(32.30 against 32.79). One thing separates them and it is enormous: the distance
to the *nearest* neighbour, 23.63 against 15.32, d = +1.81.

So hapaxes are not on the outskirts. They sit inside the cloud like everyone
else, with nobody near. Which is why no projection of the point cloud will ever
show this -- in 2D they land in the general mass -- and only the tree can, since
a tree is built out of nearest neighbours.

mst_picture.py draws the same tree four ways and answers four questions; this
answers one, and everything that does not serve it is removed. No taxonomy
colours on the edges, no dendrogram, no legend larger than the thing it labels.

  тело        tokens that occur more than once in the document. They are each
              other's nearest neighbours, so they link up at short range.
  черешки     the long edges. A token with no twin has no near neighbour of its
              own and attaches to a stranger, far away.
  врезка      the one distribution that actually separates the two populations,
              since the drawing alone cannot carry a claim about distance.
  подписи     the longest stalks are labelled with their actual word, so the
              picture reads as a text and not as an abstract graph.

The layout is Kamada-Kawai over the tree's own path metric. A tree survives that
far better than a general graph does, but it is still 768 dimensions pressed
into two: lengths on the page are approximately, not exactly, the real ones. The
claim itself does not rest on the drawing -- it rests on leaf_class.py, where a
long-stalk leaf is hapax 65.6% of the time against a base rate of 34.3%.
"""
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import (COS_CUT, NORM_CUT, SKIP, corpus_ranks,
                           token_class)
from embedder import Embedder
from sink_cluster import sink_direction

BASE = os.path.join(HERE, "..")
TEXT = int(os.environ.get("TEXT", 0))
N_LABELS = int(os.environ.get("N_LABELS", 10))
# CONTENT=1 drops punctuation, function words, subword pieces and sinks, leaving
# only whole content words. It is the control against the obvious confound --
# function words repeat by necessity and content words often do not, so "body vs
# hapax" could have been "function vs content" in disguise. It is not: among
# content words alone the separation is larger, d = 2.80 against 1.81.
CONTENT_ONLY = os.environ.get("CONTENT", "0") == "1"
CONTENT = {"смысл-част", "смысл-редк"}
BODY = "#8fb8d6"
HAPAX = "#d7191c"
SINK = "#000000"
# short edges fade into the background, long ones come forward in red: the
# length is the whole point, so it gets both the colour and the width channel
EDGE_CMAP = LinearSegmentedColormap.from_list(
    "stalk", ["#dcdcdc", "#c9c9c9", "#f0a58a", "#d7191c"])


def build(text, emb, u_sink, ranks, L=cfg.L_DEFAULT, seed=0):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    # hapax-ness is always judged against the whole text, never against whatever
    # subset is being drawn: a word used twice is not a hapax because one of its
    # occurrences was filtered out
    full = Counter(toks[i] for i in keep)
    if CONTENT_ONLY:
        nrm0 = np.linalg.norm(e[keep], axis=1)
        sk = ((e[keep] @ u_sink) / nrm0 > COS_CUT) & (nrm0 < NORM_CUT)
        keep = [k for k, x in zip(keep, sk)
                if not x and token_class(toks[k], ranks) in CONTENT]
    if len(keep) < 60:
        return None
    L = min(L, len(keep))
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v, t = e[idx], [toks[i] for i in idx]
    nrm = np.linalg.norm(v, axis=1)
    sink = ((v @ u_sink) / nrm > COS_CUT) & (nrm < NORM_CUT)
    hap = np.array([full[x] == 1 for x in t]) & ~sink
    D = squareform(pdist(v))
    m = minimum_spanning_tree(D).tocoo()
    np.fill_diagonal(D, np.inf)
    return t, hap, sink, m, D.min(1)


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    u_sink, _ = sink_direction(hum[300:340], emb)
    ranks = corpus_ranks(hum, emb)
    r = build(hum[TEXT], emb, u_sink, ranks, seed=TEXT)
    if r is None:
        raise SystemExit(f"в тексте {TEXT} слишком мало подходящих токенов")
    t, hap, sink, m, nn = r
    n = len(t)
    ra, ca, lens = m.row, m.col, m.data
    deg = np.bincount(np.concatenate([ra, ca]), minlength=n)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, w in zip(ra, ca, lens):
        G.add_edge(int(a), int(b), weight=float(w))
    P = np.array([p for _, p in sorted(
        nx.kamada_kawai_layout(G, weight="weight").items())])

    # a dedicated left column for legend and inset: the Kamada-Kawai outline
    # changes shape from text to text, so anything placed inside the tree's own
    # axes lands on top of it for some texts and not others
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 3.4],
                          height_ratios=[1.15, 1], wspace=0.02, hspace=0.05)
    ax = fig.add_subplot(gs[:, 1])
    lax = fig.add_subplot(gs[0, 0])
    lax.axis("off")
    ins = fig.add_subplot(gs[1, 0])

    rel = lens / lens.mean()
    order = np.argsort(rel)                    # long edges drawn last, on top
    norm = np.clip((rel - 0.6) / 0.9, 0, 1)
    for k in order:
        a, b = ra[k], ca[k]
        ax.plot(*zip(P[a], P[b]), color=EDGE_CMAP(norm[k]),
                lw=0.6 + 2.6 * norm[k], zorder=1 + norm[k],
                solid_capstyle="round")

    body = ~hap & ~sink
    ax.scatter(P[body, 0], P[body, 1], s=18 + 10 * deg[body], c=BODY,
               zorder=4, linewidths=0.5, edgecolors="white")
    ax.scatter(P[hap, 0], P[hap, 1], s=26 + 10 * deg[hap], c=HAPAX,
               zorder=5, linewidths=0.5, edgecolors="white")
    if sink.any():
        ax.scatter(P[sink, 0], P[sink, 1], s=22, c=SINK, zorder=5,
                   linewidths=0.5, edgecolors="white")

    # label the longest stalks -- only leaves, and only hapax ones, since those
    # are the thing the picture is about. adjust_text is not available, so
    # labels are pushed radially outward and any that still collide are dropped
    leaf = deg == 1
    stalk = {}
    for a, b, w in zip(ra, ca, lens):
        for x, y in ((a, b), (b, a)):
            if leaf[x] and hap[x]:
                stalk[x] = (w, y)
    centre = P.mean(0)
    span = np.ptp(P, axis=0).max()
    placed = []
    for i, (w, _) in sorted(stalk.items(), key=lambda kv: -kv[1][0]):
        if len(placed) >= N_LABELS:
            break
        d = P[i] - centre
        d = d / (np.linalg.norm(d) + 1e-9)
        pos = P[i] + d * 0.09 * span
        if any(np.linalg.norm(pos - q) < 0.075 * span for q in placed):
            continue
        placed.append(pos)
        ax.annotate(t[i].lstrip("Ġ▁"), P[i], pos, fontsize=11.5,
                    color="#7f1113", ha="center", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#d7191c", lw=0.7,
                                    shrinkA=0, shrinkB=3))
    ax.set_aspect("equal")
    ax.margins(0.12)

    share_h = hap.mean() * 100
    cut = np.percentile(lens, 80)
    long_h = (hap[ra[lens >= cut]] | hap[ca[lens >= cut]]).mean() * 100
    short_h = (hap[ra[lens < cut]] | hap[ca[lens < cut]]).mean() * 100
    fig.suptitle(("Только смысловые слова: служебные, пунктуация и сток убраны.\n"
                  if CONTENT_ONLY else "")
                 + "Хапаксы не с краю облака — они внутри него, но вокруг "
                 "каждого пусто.\nПоэтому в дереве они листья на длинных "
                 "черешках", fontsize=15, y=0.985)

    lax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color=BODY, markersize=10,
                   label="повторяется в тексте"),
        plt.Line2D([], [], marker="o", ls="", color=HAPAX, markersize=10,
                   label="встречается один раз"),
        plt.Line2D([], [], marker="o", ls="", color=SINK, markersize=8,
                   label="сток внимания"),
        plt.Line2D([], [], color="#d7191c", lw=3.5, label="длинное ребро"),
        plt.Line2D([], [], color="#dcdcdc", lw=1.4, label="короткое ребро"),
    ], fontsize=11, loc="upper left", frameon=False,
        bbox_to_anchor=(0.0, 0.92))

    # the inset carries the actual claim: same place in the cloud, different
    # local neighbourhood. Without it the drawing is only suggestive.
    for lab, msk, col in [("повторяется", ~hap & ~sink, BODY),
                          ("один раз", hap, HAPAX)]:
        ins.hist(nn[msk], bins=np.linspace(0, 40, 45), density=True,
                 histtype="stepfilled", alpha=0.6, color=col, label=lab)
    ins.set_xlabel("расстояние до ближайшего соседа", fontsize=9)
    ins.set_yticks([])
    ins.tick_params(labelsize=8)
    ins.legend(fontsize=9, frameon=False)
    ins.set_title("до центра облака — одинаково (d = −0.16),\n"
                  "до ближайшего — врозь (d = +2.80)" if CONTENT_ONLY else
                  "до центра облака — одинаково (d = −0.18),\n"
                  "до ближайшего — врозь (d = +1.81)", fontsize=9.5)
    for sp in ("top", "right"):
        ins.spines[sp].set_visible(False)

    fig.text(0.5, 0.045,
             f"текст {TEXT}, {n} токенов.  Хапаксы — {share_h:.0f}% вершин, "
             f"но {long_h:.0f}% длинных рёбер против {short_h:.0f}% коротких.",
             ha="center", fontsize=10.5, color="#333")
    fig.text(0.5, 0.022,
             ("По 112 текстам, только смысловые: длинный лист — хапакс в 98.2% "
              "случаев при базовом уровне 52.8%; P(длинный лист | hapax) = 23.4% "
              "против 0.5% у повторяющихся."
              if CONTENT_ONLY else
              "По 120 текстам: лист на длинном черешке — хапакс в 65.6% случаев "
              "при базовом уровне 34.3%."),
             ha="center", fontsize=10, color="#444")
    fig.text(0.5, 0.004,
             "Раскладка Kamada-Kawai по метрике дерева: длины на бумаге "
             "приблизительные. Утверждение держится не на рисунке, а на врезке "
             "и на leaf_class.py.",
             ha="center", fontsize=8.5, color="#999")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.subplots_adjust(top=0.90, bottom=0.075)
    out = os.path.join(BASE, "figures",
                       f"hapax_picture_{TEXT}"
                       f"{'_content' if CONTENT_ONLY else ''}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"текст {TEXT}: хапаксов {share_h:.0f}% вершин, "
          f"на длинных рёбрах {long_h:.0f}%, на коротких {short_h:.0f}%")
    print(f"рисунок: {out}")


if __name__ == "__main__":
    main()
