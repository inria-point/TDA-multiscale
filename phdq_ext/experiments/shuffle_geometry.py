"""What does shuffling the words do to the geometry, given that it does nothing
to the vocabulary?

Word shuffling is the cleanest control this project has: the multiset of tokens
survives exactly, so every lexical statistic -- hapax share, the multiplicity k
of every type, the class composition -- is unchanged by construction. What is
destroyed is context. Under the model in GEOMETRY.md the identity part of a
token's vector should therefore stay put and the contextual part should move, so
the cells should shift and the centres should not.

Which way the cells move is genuinely open. Occurrences of one word now land in
unrelated surroundings, which argues the cell should swell. But the surroundings
are also uniformly meaningless, so every occurrence gets the same kind of
non-context, which argues it should shrink. The two predictions are opposite and
the measurement decides.

Reported per class, on the same texts and the same seeds, original against
shuffled:

  радиус       mean distance from a type's occurrences to their centroid
  центры       spread of the centroids
  зазор        nearest foreign centre over the sum of the two radii
  до ближайшего   by multiplicity k, the quantity the whole report turns on

Plus the sanity check that the token multiset really did survive.
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
from perturb import apply as perturb
from sink_cluster import sink_direction

BASE = os.path.join(HERE, "..")
N_TEXTS = int(os.environ.get("N_TEXTS", 100))
GRP = {"смысловые": {"смысл-част", "смысл-редк"}, "служебные": {"служ"},
       "пунктуация": {"пункт"}, "подслова": {"подслово"}}


def geometry(text, emb, ranks, u_sink, seed, L=cfg.L_DEFAULT):
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
    ok = np.array([c != "сток" for c in cls])
    scale = pdist(v[ok]).mean()

    D = squareform(pdist(v))
    np.fill_diagonal(D, np.inf)
    cnt = Counter(t)
    out = {"стоков, %": sink.mean() * 100, "масштаб": scale,
           "типов k>=2": sum(1 for x, c in cnt.items() if c >= 2)}
    for kk, lab in [(1, "до бл. при k=1"), (2, "до бл. при k=2"),
                    (3, "до бл. при k>=3")]:
        m = np.array([(cnt[x] == kk if kk < 3 else cnt[x] >= 3)
                      and c != "сток" for x, c in zip(t, cls)])
        out[lab] = D.min(1)[m].mean() / scale if m.any() else np.nan

    where = defaultdict(list)
    for j, x in enumerate(t):
        if cls[j] != "сток":
            where[(cls[j], x)].append(j)
    for name, cc in GRP.items():
        g = [ix for (c, _), ix in where.items() if c in cc and len(ix) >= 2]
        if len(g) < 4:
            continue
        cen = np.array([v[ix].mean(0) for ix in g])
        rad = np.array([np.linalg.norm(v[ix] - v[ix].mean(0), axis=1).mean()
                        for ix in g])
        Dc = squareform(pdist(cen))
        np.fill_diagonal(Dc, np.inf)
        nn, j2 = Dc.min(1), Dc.argmin(1)
        iu = np.triu_indices(len(cen), 1)
        out[f"{name}: радиус"] = rad.mean() / scale
        out[f"{name}: центры"] = squareform(pdist(cen))[iu].mean() / scale
        out[f"{name}: зазор"] = float(np.mean(nn / (rad + rad[j2] + 1e-9)))
    return out, Counter(toks[i] for i in keep)


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    ranks = corpus_ranks(hum, emb)
    u_sink, _ = sink_direction(hum[300:340], emb)

    rows, kept = [], []
    for i, s in enumerate(hum[:N_TEXTS]):
        a = geometry(s, emb, ranks, u_sink, seed=i)
        b = geometry(perturb("shuffle_words", s, seed=i), emb, ranks, u_sink,
                     seed=i)
        if a is None or b is None:
            continue
        rows.append({**{f"было: {k}": v for k, v in a[0].items()},
                     **{f"стало: {k}": v for k, v in b[0].items()}})
        # did the bag of tokens really survive?
        c1, c2 = a[1], b[1]
        common = sum((c1 & c2).values())
        kept.append(common / max(sum(c1.values()), 1) * 100)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{N_TEXTS}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "shuffle_geometry.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"\n{len(D)} текстов. Мешок токенов уцелел на "
          f"{np.mean(kept):.1f}% (перетокенизация на стыках меняет остальное)\n")

    keys = [k[len("было: "):] for k in D.columns if k.startswith("было: ")]
    t = pd.DataFrame({
        "было": [D[f"было: {k}"].mean() for k in keys],
        "перемешано": [D[f"стало: {k}"].mean() for k in keys]}, index=keys)
    t["сдвиг, %"] = (t["перемешано"] - t["было"]) / t["было"] * 100
    from scipy.stats import wilcoxon
    t["p"] = [wilcoxon(D[f"было: {k}"].dropna(),
                       D[f"стало: {k}"].dropna()).pvalue
              if D[f"было: {k}"].notna().sum() > 10 else np.nan for k in keys]
    print(t.round(3).to_string())
    t.round(4).to_csv(os.path.join(BASE, "results", "shuffle_geometry_summary.csv"))
    plot(t, keys)


def plot(t, keys):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    sel = [k for k in keys if "радиус" in k or "центры" in k or "зазор" in k]
    y = np.arange(len(sel))
    a.barh(y, [t.loc[k, "сдвиг, %"] for k in sel],
           color=["#08306b" if "радиус" in k else
                  "#6baed6" if "центры" in k else "#e6550d" for k in sel])
    a.set_yticks(y)
    a.set_yticklabels(sel, fontsize=9)
    a.axvline(0, color="#333", lw=1)
    a.set_xlabel("сдвиг после перемешивания слов, %")
    a.set_title("Ячейки и их расположение")

    a = ax[1]
    sel = [k for k in keys if k.startswith("до бл.")]
    y = np.arange(len(sel))
    w = 0.38
    a.barh(y - w / 2, [t.loc[k, "было"] for k in sel], w, color="#6baed6",
           label="было")
    a.barh(y + w / 2, [t.loc[k, "перемешано"] for k in sel], w, color="#e6550d",
           label="перемешано")
    a.set_yticks(y)
    a.set_yticklabels(sel, fontsize=9)
    a.set_xlabel("расстояние до ближайшего / средняя парная")
    a.set_title("Два режима до и после")
    a.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(BASE, "figures", "shuffle_geometry.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nрисунок: {out}")


if __name__ == "__main__":
    main()
