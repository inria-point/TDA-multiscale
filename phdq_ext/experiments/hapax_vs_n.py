"""Does the estimator's own fit axis change the cloud it is fitting?

qPHD reads d off the slope of log S(n) against log n over subsample sizes
n = 41 ... 201. That is only legitimate if the thing being sampled looks the
same at every n. It does not: the share of tokens with no twin in the subsample
falls as n grows, because a second occurrence becomes more likely to be drawn.
Measured over the actual grid it is a clean power law, 73.8% at n = 41 down to
48.0% at n = 201, share ~ n^-0.27 with R^2 = 0.996.

That matters because a power-law drift in composition is exactly what a log-log
fit cannot tell apart from dimension: it adds to the slope, and d = alpha/(1-s)
turns the slope into the answer. And the tokens whose share is drifting are the
ones that hang on the long edges (leaf_class.py), so the coarse band is where it
should land.

Three measurements, in increasing strength.

  состав по n     how the MST changes across the fit grid: singleton share, the
                  share of long edges with a singleton end, mean edge length.

  кривизна        the local slope of log S between consecutive grid points. A
                  clean power law gives a flat sequence; a composition that
                  drifts with n gives a sloped one, and the direction says which
                  way d is pushed.

  контроль        d on a cloud with one vector per token type. There every token
                  is a singleton at every n, so the drift is gone by
                  construction. It removes the fine scale as well, so it is a
                  control for the coarse band only -- but that is where the
                  effect is predicted.
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
from edge_taxonomy import SKIP
from embedder import Embedder
from qphd import qphd, trimmed_sum
from sink_cluster import bands_of
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
N_TEXTS = int(os.environ.get("N_TEXTS", 60))
REPS = int(os.environ.get("REPS", 8))
GRID = cfg.aligned_grid(cfg.L_DEFAULT)
QS = [0.5, 0.7, 0.9]        # coarse-band trimming levels, q_small
B = list(BANDS)


def cloud(text, emb):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    return e[keep], [toks[i] for i in keep]


def measure(v, t, rng):
    """Across the fit grid: composition of the cloud and of its MST, and S(q)."""
    out = []
    pool_n = min(cfg.POOL_FACTOR * cfg.L_DEFAULT, len(t))
    for rep in range(REPS):
        p = rng.choice(len(t), pool_n, replace=False)
        for n in GRID:
            idx = p[rng.choice(pool_n, n, replace=False)]
            tt = [t[k] for k in idx]
            c = Counter(tt)
            solo = np.array([c[x] == 1 for x in tt])
            d = squareform(pdist(v[idx]))
            m = minimum_spanning_tree(d).tocoo()
            lens = np.sort(m.data)
            o = np.argsort(m.data)
            ra, ca = m.row[o], m.col[o]
            k = max(1, int(0.2 * len(lens)))
            rec = {"n": n, "rep": rep,
                   "одиночек, %": solo.mean() * 100,
                   "длинных с одиночкой, %":
                       (solo[ra[-k:]] | solo[ca[-k:]]).mean() * 100,
                   "средняя длина": lens.mean()}
            for q in QS:
                rec[f"S(q={q})"] = trimmed_sum(lens, q, "q_small")
            out.append(rec)
    return pd.DataFrame(out)


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()

    frames, ctrl = [], []
    for i, s in enumerate(hum[:N_TEXTS]):
        v, t = cloud(s, emb)
        if len(t) < cfg.POOL_FACTOR * cfg.L_DEFAULT:
            continue
        rng = np.random.default_rng(i)
        f = measure(v, t, rng)
        f["текст"] = i
        frames.append(f)

        # control: one vector per token type. Singleton share is 100% at every
        # n, so nothing about the cloud drifts along the fit axis.
        first = {}
        for k, x in enumerate(t):
            first.setdefault(x, k)
        u = np.array(sorted(first.values()))
        if len(u) >= cfg.L_DEFAULT:
            rng2 = np.random.default_rng(1000 + i)
            a = qphd(v[u][rng2.choice(len(u), cfg.L_DEFAULT, replace=False)],
                     q_list=cfg.Q_GRID, rng=rng2,
                     **cfg.qphd_kwargs(L=cfg.L_DEFAULT,
                                       pool=cfg.make_pool(v[u], cfg.L_DEFAULT,
                                                          rng2),
                                       replicates=16))
            rng3 = np.random.default_rng(1000 + i)
            b = qphd(v[rng3.choice(len(v), cfg.L_DEFAULT, replace=False)],
                     q_list=cfg.Q_GRID, rng=rng3,
                     **cfg.qphd_kwargs(L=cfg.L_DEFAULT,
                                       pool=cfg.make_pool(v, cfg.L_DEFAULT,
                                                          rng3),
                                       replicates=16))
            ctrl.append({"текст": i, "типов": len(u), "токенов": len(t),
                         **{f"без двойников: {k}": x
                            for k, x in bands_of(a).items()},
                         **{f"как есть: {k}": x
                            for k, x in bands_of(b).items()}})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{N_TEXTS}", flush=True)

    D = pd.concat(frames, ignore_index=True)
    D.to_csv(os.path.join(BASE, "results", "hapax_vs_n.csv"), index=False)
    pd.set_option("display.width", 220)
    by = D.groupby("n").mean(numeric_only=True).drop(columns=["rep", "текст"])
    print(f"\n{D['текст'].nunique()} текстов, {REPS} повторов\n")
    print("состав по сетке подвыборок\n")
    print(by.round(2).to_string())

    ln = np.log(by.index.values.astype(float))
    print("\nстепенные показатели по сетке (наклон log y по log n)\n")
    for c in by.columns:
        b_ = np.polyfit(ln, np.log(by[c].values), 1)[0]
        print(f"  {c:26s} n^{b_:+.4f}")

    # curvature: the local slope between consecutive grid points. A clean power
    # law gives a flat sequence; a drifting composition gives a trend.
    print("\nлокальный наклон log S между соседними точками сетки\n")
    rows = []
    for q in QS:
        y = np.log(by[f"S(q={q})"].values)
        loc = np.diff(y) / np.diff(ln)
        rows.append({"q": q, **{f"{GRID[i]}→{GRID[i + 1]}": loc[i]
                                for i in range(len(loc))},
                     "тренд": np.polyfit(ln[:-1], loc, 1)[0]})
    print(pd.DataFrame(rows).set_index("q").round(3).to_string())
    print("\n«тренд» — наклон локального наклона по log n. Ноль означает чистый\n"
          "степенной закон; положительный — S растёт всё быстрее, и подгонка\n"
          "одной прямой занижает наклон на краю сетки.")

    if ctrl:
        C = pd.DataFrame(ctrl)
        C.to_csv(os.path.join(BASE, "results", "hapax_vs_n_control.csv"),
                 index=False)
        from scipy.stats import wilcoxon
        print(f"\n\nконтроль: облако из одного вектора на тип токена "
              f"({len(C)} текстов)\n")
        print(f"типов на текст {C['типов'].mean():.0f} из "
              f"{C['токенов'].mean():.0f} токенов\n")
        out = []
        for k in B:
            a, b_ = C[f"как есть: {k}"], C[f"без двойников: {k}"]
            out.append({"полоса": k, "как есть": a.mean(),
                        "без двойников": b_.mean(),
                        "сдвиг, %": ((b_ - a) / a * 100).mean(),
                        "p": wilcoxon(a, b_).pvalue})
        print(pd.DataFrame(out).set_index("полоса").round(4).to_string())
    plot(by)


def plot(by):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    a.plot(by.index, by["одиночек, %"], "o-", color="#d7191c",
           label="одиночек в подвыборке")
    a.plot(by.index, by["длинных с одиночкой, %"], "o-", color="#333",
           label="длинных рёбер с одиночкой на конце")
    a.set_xscale("log")
    a.set_xticks(list(by.index))
    a.set_xticklabels([str(n) for n in by.index])
    a.set_xlabel("размер подвыборки n (сетка подгонки qPHD)")
    a.set_ylabel("% ")
    a.set_ylim(0, 100)
    a.set_title("Состав облака меняется вдоль оси,\nпо которой считается наклон")
    a.legend(fontsize=9)

    a = ax[1]
    ln = np.log(by.index.values.astype(float))
    for q, col in zip(QS, ["#08306b", "#2171b5", "#6baed6"]):
        y = np.log(by[f"S(q={q})"].values)
        a.plot(by.index[1:], np.diff(y) / np.diff(ln), "o-", color=col,
               label=f"q_small = {q}")
    a.set_xscale("log")
    a.set_xticks(list(by.index)[1:])
    a.set_xticklabels([str(n) for n in by.index[1:]])
    a.set_xlabel("правый конец интервала n")
    a.set_ylabel("локальный наклон log S / log n")
    a.set_title("Прямая ли на самом деле log S(n)?")
    a.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(BASE, "figures", "hapax_vs_n.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nрисунок: {out}")


if __name__ == "__main__":
    main()
