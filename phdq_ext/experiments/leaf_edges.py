"""Are the long edges the ones that hold leaves?

leaf_class.py answered this from the vertex side: a token with no twin hangs on
a long stalk. This asks it from the edge side, along the whole scale, and
aggregates to the three bands -- so the answer arrives in the same currency as
every other statement about the bands.

Two axes, kept apart because they are different things:

  черешок      the edge has an endpoint of degree 1. In a tree of more than two
               vertices an edge cannot have two such endpoints (they would be a
               component of their own), so this is a yes/no, and the share of
               such edges over the whole tree is fixed at (number of leaves) /
               (n - 1). Only its distribution along the scale is free.

  hapax-концы  how many of the edge's two endpoints occur once in the document:
               0, 1 or 2. Unlike the first this is a property of the text, known
               before any tree is built.

The first is nearly a restatement of the geometry -- a point in a sparse region
gets one connection and that connection is long -- so the number to look at is
the size of the effect, not its sign. The second is not: nothing forces a
once-used word to sit far from everything.
"""
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from band_composition import QMAX, weights
from edge_taxonomy import COS_CUT, NORM_CUT, N_BINS, SKIP
from embedder import Embedder
from sink_cluster import sink_direction
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
N_TEXTS = int(os.environ.get("N_TEXTS", 120))
SEEDS = int(os.environ.get("SEEDS", 3))


def one_text(text, emb, u_sink, L=cfg.L_DEFAULT, seed=0):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    full = Counter(toks[i] for i in keep)
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v = e[idx]
    t = [toks[i] for i in idx]
    nrm = np.linalg.norm(v, axis=1)
    sink = ((v @ u_sink) / nrm > COS_CUT) & (nrm < NORM_CUT)
    hap = np.array([full[x] == 1 for x in t]) & ~sink
    solo = np.array([Counter(t)[x] == 1 for x in t]) & ~sink

    m = minimum_spanning_tree(squareform(pdist(v))).tocoo()
    o = np.argsort(m.data)
    lens, ra, ca = m.data[o], m.row[o], m.col[o]
    deg = np.bincount(np.concatenate([ra, ca]), minlength=len(t))
    leaf = deg == 1
    return pd.DataFrame({
        "bin": np.minimum((np.arange(len(lens)) * N_BINS) // len(lens),
                          N_BINS - 1),
        "rel_len": lens / lens.mean(),
        "черешок": leaf[ra] | leaf[ca],
        "hapax-концов": hap[ra].astype(int) + hap[ca].astype(int),
        "одиночек в выборке": solo[ra].astype(int) + solo[ca].astype(int),
    })


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    u_sink, _ = sink_direction(hum[300:340], emb)

    frames = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for seed in range(SEEDS):
            r = one_text(s, emb, u_sink, seed=1000 * seed + i)
            if r is not None:
                r["текст"] = i
                frames.append(r)
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{N_TEXTS}", flush=True)
    E = pd.concat(frames, ignore_index=True)
    E.to_csv(os.path.join(BASE, "results", "leaf_edges.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"\n{len(E)} рёбер, {E['текст'].nunique()} текстов\n")

    by = E.groupby("bin").agg(
        **{"длина/средняя": ("rel_len", "mean"),
           "черешок, %": ("черешок", lambda x: x.mean() * 100),
           "хотя бы 1 hapax, %": ("hapax-концов",
                                  lambda x: (x > 0).mean() * 100),
           "оба hapax, %": ("hapax-концов", lambda x: (x == 2).mean() * 100),
           "хотя бы 1 одиночка, %": ("одиночек в выборке",
                                     lambda x: (x > 0).mean() * 100)})
    print("по 5%-бинам длины ребра\n")
    print(by.round(1).to_string())
    by.round(3).to_csv(os.path.join(BASE, "results", "leaf_edges_bins.csv"))

    print(f"\nпо всему дереву: черешков {E['черешок'].mean() * 100:.1f}%, "
          f"хотя бы 1 hapax {(E['hapax-концов'] > 0).mean() * 100:.1f}%, "
          f"оба hapax {(E['hapax-концов'] == 2).mean() * 100:.1f}%")

    # the same numbers weighted into the three bands, exactly as in
    # band_composition.py, so they can be read beside the taxonomy tables
    cols = ["черешок, %", "хотя бы 1 hapax, %", "оба hapax, %",
            "хотя бы 1 одиночка, %"]
    rows = {}
    for name, (mode, qmin) in BANDS.items():
        w = weights(mode, qmin, QMAX[mode])
        rows[name] = {c: float(w @ by[c].values) for c in cols}
    B = pd.DataFrame(rows).T
    print("\nсведено к полосам\n")
    print(B.round(1).to_string())
    B.round(3).to_csv(os.path.join(BASE, "results", "leaf_edges_bands.csv"))

    # if hapax ends were placed at random over the edges, how many edges would
    # carry one? The gap between that and the observed share is the effect.
    p = E["hapax-концов"].sum() / (2 * len(E))
    print(f"\nдоля hapax среди концов рёбер: {p * 100:.1f}%")
    print(f"случайное размещение дало бы «хотя бы один» "
          f"{(1 - (1 - p) ** 2) * 100:.1f}% рёбер и «оба» {p ** 2 * 100:.1f}%; "
          f"наблюдается {(E['hapax-концов'] > 0).mean() * 100:.1f}% и "
          f"{(E['hapax-концов'] == 2).mean() * 100:.1f}%")
    plot(by, B)


def plot(by, B):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    x = (by.index.values + 0.5) * (100 / N_BINS)
    a = ax[0]
    a.plot(x, by["черешок, %"], "o-", color="#333", label="ребро — черешок листа")
    a.plot(x, by["хотя бы 1 hapax, %"], "o-", color="#d7191c",
           label="хотя бы один конец hapax")
    a.plot(x, by["оба hapax, %"], "o-", color="#fdae61", label="оба конца hapax")
    a.set_xlim(0, 100)
    a.set_ylim(0, 100)
    a.set_xlabel("перцентиль длины ребра MST")
    a.set_ylabel("% рёбер бина")
    a.set_title("Листья и хапаксы вдоль масштаба")
    a.legend(fontsize=9)
    for p_, lab, col in [(40, "мелкая до", "#08306b"),
                         (50, "крупная от", "#a63603")]:
        a.axvline(p_, color=col, ls="--", lw=1)
        a.text(p_, 96, f" {lab}", color=col, fontsize=8.5)

    a = ax[1]
    w = 0.26
    xs = np.arange(len(B))
    for k, (c, col) in enumerate([("черешок, %", "#333"),
                                  ("хотя бы 1 hapax, %", "#d7191c"),
                                  ("оба hapax, %", "#fdae61")]):
        a.bar(xs + (k - 1) * w, B[c], w, color=col, label=c)
    a.set_xticks(xs)
    a.set_xticklabels(B.index)
    a.set_ylabel("% рёбер полосы")
    a.set_title("То же, сведённое к трём полосам")
    a.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(BASE, "figures", "leaf_edges.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nрисунок: {out}")


if __name__ == "__main__":
    main()
