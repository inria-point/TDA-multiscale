"""Two dimensions in one cloud, and what decides which of them the band shows.

The model: the cloud carries two different geometries at once. The arrangement of
cell centres -- one per token type, with a singleton being a cell of one point --
has its own dimension. The packing of occurrences inside a cell has another, and
a lower one. What a band reports is a mixture, and the mixing weight is set by
the text: many singletons and tight cells push it toward the centre dimension,
swollen cells and few singletons push it toward the packing dimension.

Two consequences, tested separately.

  d центров > d упаковки   qPHD run on the centroid cloud -- each type collapsed
                           to the mean of its occurrences -- against the same
                           estimator on the ordinary cloud. Paired, same texts,
                           same seeds. The packing exponent measured in
                           type_regions.py gives the other end for comparison.

  полоса как смесь         across texts, the coarse band should fall as the
                           cells swell and rise with the share of singletons.
                           The second half is already known (partial r = +0.41);
                           the first is the new prediction and the sharper one,
                           since cell radius and singleton share are close to
                           independent.

The centroid cloud needs as many types as the estimator needs points, so only
texts with at least L distinct token types can be measured that way; that
selection is reported rather than hidden.
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import COS_CUT, NORM_CUT, SKIP, corpus_ranks, token_class
from embedder import Embedder
from qphd import qphd
from sink_cluster import bands_of, sink_direction
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
B = list(BANDS)
N_TEXTS = int(os.environ.get("N_TEXTS", 100))
N_SEEDS = int(os.environ.get("N_SEEDS", 4))
REPLICATES = int(os.environ.get("REPLICATES", 16))


def clouds(text, emb, ranks, u_sink):
    """Full cloud without sinks, and the centroid cloud built from it."""
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if not keep:
        return None
    v = e[keep]
    t = [toks[i] for i in keep]
    nrm = np.linalg.norm(v, axis=1)
    sink = ((v @ u_sink) / nrm > COS_CUT) & (nrm < NORM_CUT)
    v, t = v[~sink], [x for x, s_ in zip(t, sink) if not s_]
    cls = [token_class(x, ranks) for x in t]

    where = defaultdict(list)
    for j, x in enumerate(t):
        where[x].append(j)
    cen = np.array([v[ix].mean(0) for ix in where.values()])
    # cell radius, on the text's own scale, over types that have a cell at all
    rng = np.random.default_rng(0)
    sub = v[rng.choice(len(v), min(len(v), 400), replace=False)]
    scale = pdist(sub).mean()
    rad = [np.linalg.norm(v[ix] - v[ix].mean(0), axis=1).mean()
           for ix in where.values() if len(ix) >= 2]
    cnt = Counter(t)
    return dict(
        полное=v, центры=cen,
        радиус=float(np.mean(rad)) / scale if rad else np.nan,
        радиус_смысл=float(np.mean(
            [np.linalg.norm(v[ix] - v[ix].mean(0), axis=1).mean()
             for x, ix in where.items()
             if len(ix) >= 2 and token_class(x, ranks).startswith("смысл")]
            or [np.nan])) / scale,
        одиночек=float(np.mean([cnt[x] == 1 for x in t])),
        типов=len(where), токенов=len(t))


def run(v, seed, L=cfg.L_DEFAULT):
    if v.shape[0] < L:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(v.shape[0], size=L, replace=False)
    return qphd(v[idx], q_list=cfg.Q_GRID, rng=rng,
                **cfg.qphd_kwargs(L=L, pool=cfg.make_pool(v, L, rng),
                                  replicates=REPLICATES))


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = pool[pool.is_human].reset_index(drop=True)
    texts = list(hum["text"])
    emb = Embedder()
    ranks = corpus_ranks(texts, emb)
    u_sink, _ = sink_direction(texts[300:340], emb)

    rows = []
    for i, s in enumerate(texts[:N_TEXTS]):
        c = clouds(s, emb, ranks, u_sink)
        if c is None:
            continue
        rec = {"текст": i, "id": str(hum.loc[i, "id"]),
               "радиус": c["радиус"], "радиус_смысл": c["радиус_смысл"],
               "одиночек": c["одиночек"] * 100,
               "типов": c["типов"], "токенов": c["токенов"]}
        for name, cloud in [("полное", c["полное"]), ("центры", c["центры"])]:
            acc = []
            for seed in range(N_SEEDS):
                df = run(cloud, seed=3000 + seed)
                if df is not None:
                    acc.append(bands_of(df))
            if acc:
                m = pd.DataFrame(acc).mean()
                rec.update({f"{name}: {k}": v for k, v in m.items()})
        rows.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{N_TEXTS}", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "two_dimensions.csv"), index=False)
    pd.set_option("display.width", 200)
    both = D.dropna(subset=[f"центры: {b}" for b in B])
    print(f"\n{len(D)} текстов посчитано, из них {len(both)} имеют >= "
          f"{cfg.L_DEFAULT} различных типов и потому дают облако центров")
    print(f"типов на текст {D['типов'].mean():.0f} из "
          f"{D['токенов'].mean():.0f} токенов\n")

    from scipy.stats import wilcoxon
    print("=" * 60)
    print("облако центров против полного облака (парно)\n")
    out = []
    for b in B:
        a, c = both[f"полное: {b}"], both[f"центры: {b}"]
        out.append({"полоса": b, "полное облако": a.mean(),
                    "облако центров": c.mean(),
                    "сдвиг, %": ((c - a) / a * 100).mean(),
                    "p": wilcoxon(a, c).pvalue})
    print(pd.DataFrame(out).set_index("полоса").round(4).to_string())
    print("\nдля сравнения: показатель упаковки внутри ячеек (type_regions.py)")
    print("  спад до ближайшего ~ k^-0.161  ->  d упаковки = 6.2")

    print("\n" + "=" * 60)
    print("полоса как смесь: что её двигает по текстам\n")
    T = pd.read_csv(os.path.join(BASE, "results", "human_band_dev.csv"))
    T["id"] = T["id"].astype(str)
    M = D.merge(T[["id"] + B], on="id", suffixes=("", "_ref"))
    M["логN"] = np.log(M["токенов"])

    def resid(y, x):
        X = np.c_[np.ones(len(x)), x]
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    print(f"{len(M)} текстов с готовыми полосами\n")
    print(f"{'предиктор':16s} {'полоса':10s} {'r':>8s} {'r|N':>8s}")
    for name in ["радиус", "радиус_смысл", "одиночек"]:
        for b in B:
            x, y = M[name].values, M[b].values
            ok = np.isfinite(x) & np.isfinite(y)
            r = np.corrcoef(x[ok], y[ok])[0, 1]
            rp = np.corrcoef(resid(x[ok], M["логN"].values[ok]),
                             resid(y[ok], M["логN"].values[ok]))[0, 1]
            print(f"{name:16s} {b:10s} {r:+8.3f} {rp:+8.3f}")
    print("\nпредсказание модели: радиус — отрицательно, одиночек — положительно")
    print(f"\nрадиус и доля одиночек между собой: "
          f"r = {M[['радиус', 'одиночек']].corr().iloc[0, 1]:+.3f} "
          "(если близко к нулю, это два независимых рычага)")


if __name__ == "__main__":
    main()
