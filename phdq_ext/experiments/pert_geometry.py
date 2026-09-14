"""Do the band shifts of the big perturbations follow from the cell model?

GEOMETRY.md ends with a picture: a frame of type centres, a cell of radius
~0.23 around each repeated type, and singletons hanging off foreign cells on
long petioles. Shuffling the words was explained inside that picture -- cells
swell 17%, the frame stands still, and a synthetic arrangement with no text in
it reproduces the signs of both bands.

Shuffling is one transformation. This runs the same battery over the dozen
mechanical perturbations that move the bands hardest, so that every one of them
can be stated as a displacement in the same coordinates:

  радиус        how tightly the occurrences of one type sit together
  зазор         nearest foreign centre over the two radii: do cells overlap
  одиночек, %   the share of the sample with no twin -- what feeds the coarse band
  черешок       how far a singleton hangs from whatever holds it
  p90/p50       how uneven the edge lengths are, which is what d actually reads
  состав        class shares and the sink, because a perturbation moves those too

Bands are recomputed here rather than read from the catalogue, so that the
geometry and the dimension come from the same sample of the same text, and the
pairing is exact. They will differ a little from catalogue.md, which does not
drop the special tokens.
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import COS_CUT, NORM_CUT, SKIP, corpus_ranks, token_class
from embedder import Embedder
from perturb import apply as perturb
from qphd import qphd
from sink_cluster import bands_of, sink_direction

BASE = os.path.join(HERE, "..")
N_TEXTS = int(os.environ.get("N_TEXTS", 50))
SEEDS = int(os.environ.get("SEEDS", 3))
OUT = os.path.join(BASE, "results", os.environ.get("OUT", "pert_geometry.csv"))

# Chosen from catalogue.md: the mechanical perturbations with the largest and
# most interpretable band shifts, arranged as four questions.
PERTS = [
    ("исходный", "identity"),
    # 1. the radius axis: shuffling inflates the cell, echoing should crush it
    ("перемешивание", "shuffle_words"),
    ("эхо x2", "loop_local_1"),
    ("эхо x4", "loop_local_3"),
    ("эхо на четверти", "echo_p25"),
    # 2. the singleton axis, from none to all
    ("словарь к 25", "collapse_vocab_25"),
    ("словарь к 60", "collapse_vocab_60"),
    ("все слова разные", "expand_vocab_100"),
    # 3. singleton count held fixed, only their fit to the context changes
    ("чужие хапаксы", "hapax_swap_wide"),
    ("хапаксы одного домена", "hapax_swap_local"),
    ("20 чужих хапаксов", "hapax_swap_wide_20w"),
    # 4. composition: dilute the sample, or remove a whole class
    ("маркер каждые 3", "marker_p3"),
    ("без пунктуации", "strip_punctuation"),
    ("без служебных", "drop_function_words"),
]

GRP = {"смысловые": {"смысл-част", "смысл-редк"}, "служебные": {"служ"},
       "пунктуация": {"пункт"}, "подслова": {"подслово"}}


def geometry(v, t, cls, single):
    """The model's coordinates for one L-token sample."""
    ok = np.array([c != "сток" for c in cls])
    scale = pdist(v[ok]).mean()
    D = squareform(pdist(v))
    np.fill_diagonal(D, np.inf)
    cnt = Counter(t)

    out = {"масштаб": scale,
           "стоков, %": (~ok).mean() * 100,
           "одиночек, %": single[ok].mean() * 100,
           "типов всего": len(cnt),
           "типов k>=2": sum(1 for x, c in cnt.items() if c >= 2)}
    for kk, lab in [(1, "до бл. при k=1"), (2, "до бл. при k=2"),
                    (3, "до бл. при k>=3")]:
        m = np.array([(cnt[x] == kk if kk < 3 else cnt[x] >= 3) and c != "сток"
                      for x, c in zip(t, cls)])
        out[lab] = D.min(1)[m].mean() / scale if m.any() else np.nan
    for name, cc in GRP.items():
        out[f"доля {name}, %"] = np.mean([c in cc for c in cls]) * 100

    # the tree: what holds the singletons, and how far away it is
    mst = minimum_spanning_tree(squareform(pdist(v))).tocoo()
    w = mst.data / scale
    deg = np.zeros(len(v), int)
    for a, b in zip(mst.row, mst.col):
        deg[a] += 1
        deg[b] += 1
    longest = np.zeros(len(v))
    for a, b, ww in zip(mst.row, mst.col, w):
        longest[a] = max(longest[a], ww)
        longest[b] = max(longest[b], ww)
    leaf = deg == 1
    cut = np.percentile(w, 80)
    # the shape of the edge-length distribution, which is what a dimension
    # estimate actually reads: a uniform d-dimensional cloud gives lengths
    # that concentrate, and any clumping shows up as a heavier spread
    out["ребро, среднее"] = w.mean()
    out["рёбра: CV"] = w.std() / w.mean()
    top = w[w >= np.percentile(w, 80)]
    out["длинные: CV"] = top.std() / top.mean()
    q10, q50, q90, q99 = np.percentile(w, [10, 50, 90, 99])
    out["p90/p50"] = q90 / q50
    out["p99/p50"] = q99 / q50
    out["p10/p50"] = q10 / q50
    out["черешок одиночки"] = (longest[leaf & single].mean()
                               if (leaf & single).any() else np.nan)
    out["длинных листьев, %"] = np.mean(leaf & (longest > cut)) * 100
    both = np.mean([single[a] and single[b] for a, b in zip(mst.row, mst.col)])
    p = single.mean()
    out["оба одиночки / случай"] = both / (p * p) if p > 0 else np.nan

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
        out[f"{name}: ячеек"] = len(g)
        out[f"{name}: радиус"] = rad.mean() / scale
        out[f"{name}: центры"] = squareform(pdist(cen))[iu].mean() / scale
        out[f"{name}: зазор"] = float(np.mean(nn / (rad + rad[j2] + 1e-9)))
    return out


def measure(text, emb, ranks, u_sink, seed, key, L=cfg.L_DEFAULT):
    """Geometry and bands from one sample of one text."""
    e = emb.embed_cached(text, cache_key=key)
    toks = emb.tokenizer.tokenize(text)
    if len(toks) != e.shape[0]:  # cache holds the embeddings only
        toks = toks[:e.shape[0]]
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    e, toks = e[keep], [toks[i] for i in keep]
    if e.shape[0] < L:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(e.shape[0], size=L, replace=False)
    v, t = e[idx], [toks[i] for i in idx]
    nrm = np.linalg.norm(v, axis=1)
    sink = ((v @ u_sink) / nrm > COS_CUT) & (nrm < NORM_CUT)
    cls = ["сток" if s_ else token_class(x, ranks) for x, s_ in zip(t, sink)]
    cnt = Counter(t)
    single = np.array([cnt[x] == 1 for x in t])
    g = geometry(v, t, cls, single)
    df = qphd(v, q_list=cfg.Q_GRID, rng=rng,
              **cfg.qphd_kwargs(L=L, pool=cfg.make_pool(e, L, rng)))
    return {**bands_of(df), **g}


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    ranks = corpus_ranks(hum, emb)
    u_sink, _ = sink_direction(hum[300:340], emb)
    print(f"{len(PERTS)} условий x {N_TEXTS} текстов x {SEEDS} зёрен",
          flush=True)

    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in PERTS:
            try:
                txt = s if name == "identity" else perturb(name, s, seed=i)
            except Exception as exc:                     # noqa: BLE001
                print(f"  {label}: {exc}", flush=True)
                continue
            for seed in range(SEEDS):
                r = measure(txt, emb, ranks, u_sink, seed=1000 + seed,
                            key=f"pg_{name}_{i}")
                if r is None:
                    continue
                rows.append({"текст": i, "условие": label, "зерно": seed, **r})
        if (i + 1) % 5 == 0:
            print(f"  {i + 1} текстов, {len(rows)} строк", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT, index=False)
    print(f"\n{OUT}: {len(D)} строк")


if __name__ == "__main__":
    main()
