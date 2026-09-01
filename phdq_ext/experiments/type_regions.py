"""Does each token type own a region of fixed size that its copies fill up?

The model to test: a type occupies a patch of space whose size does not depend
on how many times the text happens to use it, and the k occurrences are spread
inside that patch, so the spacing between them falls as k grows.

It has numerical consequences, and one of them already looks wrong. If spacing
were R * k^(-1/d), the curve would be a single power law; measured, it drops
from 0.731 at k = 1 to 0.486 at k = 2 and then only to 0.371 by k = 10. A knee,
not a line. The likely reason is that k = 1 is a different regime altogether:
with no second copy, the nearest neighbour is not a room-mate but whatever
foreign region happens to lie closest, so k = 1 measures the gap *between*
regions and k >= 2 measures the packing *inside* one.

Three measurements settle it.

  чей сосед     for each k, how often the nearest neighbour is the same token.
                The model needs this near zero at k = 1 and high above it.
  размер        the mean pairwise distance among the copies of one type, against
                k. "Fixed region" means flat; a region that grows with use means
                rising.
  показатель    the log-log slope of spacing against k for k >= 2, which is
                -1/d and so names the local dimension the packing implies.

All distances are divided by the text's own mean pairwise distance, so texts of
different spread pool.
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

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import COS_CUT, NORM_CUT, SKIP, corpus_ranks, token_class
from embedder import Embedder
from sink_cluster import sink_direction

BASE = os.path.join(HERE, "..")
N_TEXTS = int(os.environ.get("N_TEXTS", 120))
SEEDS = int(os.environ.get("SEEDS", 3))
CLASSES = ["пункт", "служ", "смысл-част", "смысл-редк", "подслово"]


def one(text, emb, ranks, u_sink, L=cfg.L_DEFAULT, seed=0):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v, t = e[idx], [toks[i] for i in idx]
    nrm = np.linalg.norm(v, axis=1)
    sink = ((v @ u_sink) / nrm > COS_CUT) & (nrm < NORM_CUT)
    cls = ["сток" if s_ else token_class(x, ranks) for x, s_ in zip(t, sink)]
    cnt = Counter(t)
    D = squareform(pdist(v))
    scale = D[np.triu_indices(len(t), 1)].mean()
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)

    where = defaultdict(list)
    for j, x in enumerate(t):
        where[x].append(j)

    vert, reg = [], []
    for j, x in enumerate(t):
        vert.append({"токен": x, "класс": cls[j], "k": cnt[x],
                     "до ближайшего": D[j, nn[j]] / scale,
                     "сосед свой": t[nn[j]] == x})
    for x, js in where.items():
        if len(js) < 2 or cls[js[0]] == "сток":
            continue
        sub = squareform(pdist(v[js]))
        iu = np.triu_indices(len(js), 1)
        reg.append({"токен": x, "класс": cls[js[0]], "k": len(js),
                    "размах": sub[iu].mean() / scale,
                    "диаметр": sub[iu].max() / scale,
                    "мин. пара": sub[iu].min() / scale})
    return pd.DataFrame(vert), pd.DataFrame(reg)


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    ranks = corpus_ranks(hum, emb)
    u_sink, _ = sink_direction(hum[300:340], emb)

    A, B = [], []
    for i, s in enumerate(hum[:N_TEXTS]):
        for seed in range(SEEDS):
            r = one(s, emb, ranks, u_sink, seed=1000 * seed + i)
            if r is not None:
                A.append(r[0])
                B.append(r[1])
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{N_TEXTS}", flush=True)
    V = pd.concat(A, ignore_index=True)
    R = pd.concat(B, ignore_index=True)
    V.to_csv(os.path.join(BASE, "results", "type_regions_vertices.csv"),
             index=False)
    R.to_csv(os.path.join(BASE, "results", "type_regions.csv"), index=False)
    pd.set_option("display.width", 220)
    W = V[V["класс"] != "сток"]
    print(f"\n{len(W)} вершин, {len(R)} типов с k>=2, "
          f"{N_TEXTS} текстов\n")

    print("=" * 64)
    print("чей сосед оказывается ближайшим\n")
    t = W.groupby(W["k"].clip(upper=8)).agg(
        **{"сосед — свой же токен, %": ("сосед свой", lambda x: x.mean() * 100),
           "до ближайшего": ("до ближайшего", "mean"),
           "вершин": ("k", "size")})
    t.index.name = "k в выборке (8 = 8+)"
    print(t.round(2).to_string())

    print("\n" + "=" * 64)
    print("размер области типа против числа экземпляров в ней\n")
    g = R.groupby(R["k"].clip(upper=8)).agg(
        **{"размах (ср. парное)": ("размах", "mean"),
           "диаметр": ("диаметр", "mean"),
           "ближайшая пара": ("мин. пара", "mean"),
           "типов": ("k", "size")})
    g.index.name = "k"
    print(g.round(3).to_string())
    kk = g.index.values.astype(float)
    for c in ["размах", "диаметр", "мин. пара"]:
        col = {"размах": "размах (ср. парное)", "диаметр": "диаметр",
               "мин. пара": "ближайшая пара"}[c]
        b = np.polyfit(np.log(kk), np.log(g[col].values), 1)[0]
        print(f"  {c:14s} ~ k^{b:+.3f}"
              + ("   (постоянен — область не растёт)" if abs(b) < 0.12 else ""))

    print("\n" + "=" * 64)
    print("показатель упаковки: спад расстояния до ближайшего при k>=2\n")
    for name, sub in [("все классы", W)] + [(c, W[W["класс"] == c])
                                            for c in CLASSES]:
        m = sub[sub["k"] >= 2].groupby(sub["k"].clip(upper=9))[
            "до ближайшего"].mean().dropna()
        if len(m) < 3:
            continue
        b = np.polyfit(np.log(m.index.values.astype(float)),
                       np.log(m.values), 1)[0]
        print(f"  {name:12s}  спад ~ k^{b:+.3f}   -> локальная d = "
              f"{-1 / b:5.1f}")
    m1 = W[W["k"] == 1]["до ближайшего"].mean()
    m2 = W[W["k"] == 2]["до ближайшего"].mean()
    print(f"\n  переход k=1 -> k=2 даёт k^{np.log(m2 / m1) / np.log(2):+.3f}, "
          f"то есть d = {-1 / (np.log(m2 / m1) / np.log(2)):.1f} — "
          "другой режим,\n  как и предполагалось: при k=1 ближайший сосед чужой.")
    plot(t, g, W)


def plot(t, g, W):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    a = ax[0]
    a.plot(t.index, t["сосед — свой же токен, %"], "o-", color="#08306b")
    a.set_xlabel("вхождений типа в выборке")
    a.set_ylabel("%")
    a.set_ylim(0, 100)
    a.set_title("Ближайший сосед — то же самое слово?")

    a = ax[1]
    for c, col, lab in [("размах (ср. парное)", "#e6550d", "среднее парное"),
                        ("диаметр", "#08306b", "диаметр"),
                        ("ближайшая пара", "#74c476", "ближайшая пара")]:
        a.plot(g.index, g[c], "o-", color=col, label=lab)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("экземпляров типа в выборке")
    a.set_ylabel("расстояние / средняя парная")
    a.set_title("Область типа: растёт ли она с числом экземпляров?")
    a.legend(fontsize=9)

    a = ax[2]
    m = W.groupby(W["k"].clip(upper=9))["до ближайшего"].mean()
    a.plot(m.index, m.values, "o-", color="#333")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("экземпляров типа в выборке")
    a.set_ylabel("до ближайшего / средняя парная")
    a.set_title("Колено на k = 1 → 2: смена режима")
    a.annotate("свой сосед появляется", (2, m.loc[2]), (2.6, m.loc[2] * 1.22),
               fontsize=9, color="#a63603",
               arrowprops=dict(arrowstyle="->", color="#a63603"))
    fig.tight_layout()
    out = os.path.join(BASE, "figures", "type_regions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nрисунок: {out}")


if __name__ == "__main__":
    main()
