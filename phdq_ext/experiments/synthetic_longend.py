"""Two more sweeps of the same synthetic arrangement, for the other two bands.

synthetic_inflate.py swept one knob -- the radius of the cells -- and showed
that the shuffle signature (fine up, coarse down) follows from geometry alone.
The perturbation battery (pert_geometry.py) says each band answers to a
different coordinate, across twelve transformations:

  мелкая   cell radius                      r = +0.92
  средняя  share of singletons              r = +0.95
  крупная  spread of the long edges         rho = -0.74, and nothing better

The first is already established synthetically. The other two are not, and both
are cheap to settle here, because the arrangement has a knob for each.

Sweep A: the share of points that are singletons, cells and radius fixed. If
the middle band rides on it, d should climb with the share.

Sweep B: heterogeneity of the long end at a fixed number of singletons. A
fraction of the singletons is pulled in towards the nearest cell centre, so
that the petioles stop being one population and become two. Nothing else moves
-- not the count, not the radius, not the frame. If the coarse band reads the
evenness of the long edges rather than their length, it must fall as the
fraction grows, and fall hardest where the mixture is most even.

This is the model's account of hapax_swap_wide, which drops the coarse band 17%
with every lexical statistic and every cell coordinate unchanged, and shortens
a part of the petioles while raising their spread 23%.
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, "experiments")
sys.path.insert(0, "scripts")
import config as cfg
from qphd import qphd
from sink_cluster import bands_of
from three_bands import BANDS

B = list(BANDS)
L = cfg.L_DEFAULT
DC = 24
KS = [2, 2, 2, 3, 3, 3, 4, 4, 5, 6]
RAD = 0.232          # the radius measured on real text
REPS = 12


def build(n_sing, n_cell, r_target, pull_share=0.0, pull=0.35, push=0.0,
          glue_share=0.0, glue=0.15, rng=None):
    """n_sing singletons and n_cell cells of radius r_target, L points total.

    pull_share of the singletons are moved to a point `pull` of the way from
    the frame towards the nearest centre, which shortens their petiole without
    changing anything else.

    push moves ALL singletons that same fraction further out from the nearest
    centre instead: the second knob of the model, the one semantic enrichment
    turns. Everything stays comparable between singletons, only the whole
    population goes further from the frame.

    glue_share pairs up that share of the singletons and brings each pair to
    `glue` of its original separation -- they stick to each other rather than
    move relative to the frame. This is the operation a defect appears to
    perform: in real corrupted text the inappropriate singletons end up 14%
    nearer one another (40 texts of 40) while their distance to the frame
    hardly moves, and 74.8% of singleton petioles lead to another singleton
    rather than to a cell centre. Gluing removes long edges without shortening
    the ones that remain, which is the signature the other two knobs cannot
    produce.
    """
    need = L - n_sing
    # every cell needs at least two points, so the number of cells cannot
    # exceed half of what the singletons leave -- without this the trimming
    # loop below has no way to reach `need` and spins forever
    n_cell = min(n_cell, max(1, need // 2))
    ks = rng.choice(KS, n_cell)
    while ks.sum() < need:
        ks[rng.integers(n_cell)] += 1
    while ks.sum() > need:
        j = rng.integers(n_cell)
        if ks[j] > 2:
            ks[j] -= 1
    cen = rng.normal(size=(n_sing + n_cell, DC))
    sing, ccen = cen[:n_sing].copy(), cen[n_sing:]
    blobs = [rng.normal(size=(k, DC)) for k in ks]

    # scale the blobs so that radius / mean pairwise distance = r_target
    X0 = np.vstack([sing] + [c + b for c, b in zip(ccen, blobs)])
    r0 = np.mean([np.linalg.norm(b - b.mean(0), axis=1).mean() for b in blobs])
    f = r_target * pdist(X0).mean() / r0
    blobs = [b * f for b in blobs]

    if pull_share > 0 or push != 0 or glue_share > 0:
        d = squareform(pdist(np.vstack([sing, ccen])))[:n_sing, n_sing:]
        near = ccen[d.argmin(1)]
        if push != 0:                      # все одиночки дальше от каркаса
            sing = sing + push * (sing - near)
        if pull_share > 0:                 # часть одиночек внутрь
            m = int(round(pull_share * n_sing))
            idx = rng.choice(n_sing, m, replace=False)
            sing[idx] = sing[idx] + pull * (near[idx] - sing[idx])
    if glue_share > 0:                     # часть одиночек слипается попарно
        m = int(round(glue_share * n_sing / 2)) * 2
        idx = rng.choice(n_sing, m, replace=False)
        for a, b in zip(idx[::2], idx[1::2]):
            mid = (sing[a] + sing[b]) / 2
            sing[a] = mid + glue * (sing[a] - mid)
            sing[b] = mid + glue * (sing[b] - mid)
    X = np.vstack([sing] + [c + b for c, b in zip(ccen, blobs)])
    return X


def edge_stats(X, n_sing=None):
    D = squareform(pdist(X))
    mst = minimum_spanning_tree(D).tocoo()
    w = mst.data / pdist(X).mean()
    top = w[w >= np.percentile(w, 80)]
    out = {"длинные: CV": top.std() / top.mean(),
           "p10/p50": np.percentile(w, 10) / np.percentile(w, 50),
           "p99/p50": np.percentile(w, 99) / np.percentile(w, 50)}
    if n_sing:
        # черешок вершины — её самое длинное ребро; одиночки по построению
        # занимают первые n_sing строк
        longest = np.zeros(len(X))
        for a, b, ww in zip(mst.row, mst.col, w):
            longest[a] = max(longest[a], ww)
            longest[b] = max(longest[b], ww)
        out["средний черешок"] = float(longest[:n_sing].mean())
        out["разброс черешков"] = float(longest[:n_sing].std())
    return out


def run(X, rng, n_sing=None):
    df = qphd(X, q_list=cfg.Q_GRID, rng=rng,
              **cfg.qphd_kwargs(L=L, pool=X, replicates=16))
    return {**bands_of(df), **edge_stats(X, n_sing)}


def sweep(name, cases):
    rows = []
    for tag, kw in cases:
        for rep in range(REPS):
            rng = np.random.default_rng(hash((name, tag, rep)) % 2 ** 31)
            rows.append({name: tag,
                         **run(build(rng=rng, **kw), rng,
                               n_sing=kw.get("n_sing"))})
    D = pd.DataFrame(rows)
    print(f"  {name}: {len(D)} прогонов", flush=True)
    return D, D.groupby(name).mean()


def show(g, base, title):
    out = g.copy()
    for c in B + ["длинные: CV", "p10/p50", "p99/p50"]:
        out[c + " %"] = (g[c] - g.loc[base, c]) / g.loc[base, c] * 100
    print(f"\n{title}")
    keep = B + ["длинные: CV"] + [c for c in ("средний черешок",
                                              "разброс черешков")
                                  if c in out]
    print(out[keep].round(3).to_string())
    print(out[[c + " %" for c in B] + ["длинные: CV %"]].round(1).to_string())


def main():
    print(f"синтетика: {L} точек, центры из {DC}-мерного гаусса, "
          f"{REPS} повторов, ничего лексического", flush=True)

    cases = [(n, dict(n_sing=n, n_cell=31, r_target=RAD))
             for n in [40, 60, 80, 94, 110, 130, 150]]
    D1, g1 = sweep("одиночек", cases)
    D1.to_csv("results/synthetic_singletons.csv", index=False)
    show(g1, 94, "A. доля одиночек при том же радиусе (94 = реальный текст)")

    cases = [(s, dict(n_sing=94, n_cell=31, r_target=RAD, pull_share=s))
             for s in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]]
    D2, g2 = sweep("подтянуто", cases)
    D2.to_csv("results/synthetic_longend.csv", index=False)
    show(g2, 0.0, "B. часть одиночек подтянута к ближайшему центру на 35%")

    cases = [(p, dict(n_sing=94, n_cell=31, r_target=RAD, push=p))
             for p in [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.25]]
    D3, g3 = sweep("отодвинуто", cases)
    D3.to_csv("results/synthetic_push.csv", index=False)
    show(g3, 0.0, "C. ВСЕ одиночки равномерно отодвинуты от каркаса")
    print("\nреальность: художественный пересказ даёт крупная +13.0 при "
          "среднем черешке −0.0% и разбросе −5.0%")

    # сила склейки взята из замера: у неуместных одиночек взаимное расстояние
    # падает на 14% (0.869 против 1.009 у уместных, 40 текстов из 40), то есть
    # коэффициент около 0.86, а не 0.15. При 0.15 пара почти совмещается, даёт
    # сверхкороткое ребро и обваливает мелкую полосу, чего в текстах нет
    cases = [(f"{g:.1f} на {k:.2f}", dict(n_sing=94, n_cell=31, r_target=RAD,
                                          glue_share=g, glue=k))
             for k in (0.85, 0.70, 0.50)
             for g in (0.0, 0.5, 1.0)]
    D4, g4 = sweep("слиплось", cases)
    D4.to_csv("results/synthetic_glue.csv", index=False)
    show(g4, "0.0 на 0.85", "D. одиночки слипаются попарно; метка — доля "
                            "слипшихся и коэффициент сближения")
    print("\nреальность: чужие хапаксы дают крупная −17.2 при расстоянии до "
          "каркаса −1.1% и среднем черешке −3.6%;\n"
          "             словарь к 25 даёт −31.2 при расстоянии до каркаса "
          "−1.8%")
    print("\nреальность: чужие хапаксы дают крупная −17.2, мелкая +13.0, "
          "средняя −10.0 при CV длинного конца +23%")


if __name__ == "__main__":
    main()
