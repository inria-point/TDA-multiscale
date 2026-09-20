"""Работают ли наши две ручки там, где теорема чиста?

report_1.pdf доказывает: если ячейки отделены (δ > 4r, то есть g = δ/2r > 2),
спектр MST расщепляется точно на q* = (n−N)/(n−1), ниже лежат только
внутриячеечные рёбра, выше только каркасные. В настоящем тексте g = 1.60, то
есть ниже порога, и страты перемешаны. Наши синтетические развёртки тоже шли
при радиусе 0.232, то есть в том же грязном режиме.

Отсюда альтернатива, которую надо исключить. Две ручки, которыми мы объясняли
крупную полосу — отодвигание всех одиночек от каркаса и попарная склейка
одиночек, — могут быть не свойством каркаса, а очередным лицом потери
отделимости. Различить можно, прогнав те же ручки при малом радиусе, где
страты чисты:

  сохранятся  → ручки про неоднородность каркаса, и её надо вводить в теорию,
                потому что каркас у теоремы однороден по построению
                (центры — i.i.d. выборка из d_g-регулярной меры)
  исчезнут    → всё сводится к q* плюс отделимость, новой сущности не нужно

Отделимость здесь измеряется, а не подставляется: g = (среднее расстояние до
ближайшего чужого центра) / 2r на самом построенном облаке. Стока в синтетике
нет по построению.
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
N_SING, N_CELL = 94, 31
REPS = 10


def build(r_target, push=0.0, pull_share=0.0, pull=0.35,
          glue_share=0.0, glue=0.70, rng=None):
    ks = rng.choice(KS, N_CELL)
    need = L - N_SING
    while ks.sum() < need:
        ks[rng.integers(N_CELL)] += 1
    while ks.sum() > need:
        j = rng.integers(N_CELL)
        if ks[j] > 2:
            ks[j] -= 1
    cen = rng.normal(size=(N_SING + N_CELL, DC))
    sing, ccen = cen[:N_SING].copy(), cen[N_SING:]
    blobs = [rng.normal(size=(k, DC)) for k in ks]
    X0 = np.vstack([sing] + [c + b for c, b in zip(ccen, blobs)])
    r0 = np.mean([np.linalg.norm(b - b.mean(0), axis=1).mean() for b in blobs])
    f = r_target * pdist(X0).mean() / r0
    blobs = [b * f for b in blobs]
    rad = np.mean([np.linalg.norm(b - b.mean(0), axis=1).mean() for b in blobs])

    d = squareform(pdist(np.vstack([sing, ccen])))[:N_SING, N_SING:]
    near = ccen[d.argmin(1)]
    if push:
        sing = sing + push * (sing - near)
    if pull_share > 0:
        m = int(round(pull_share * N_SING))
        idx = rng.choice(N_SING, m, replace=False)
        sing[idx] = sing[idx] + pull * (near[idx] - sing[idx])
    if glue_share > 0:
        m = int(round(glue_share * N_SING / 2)) * 2
        idx = rng.choice(N_SING, m, replace=False)
        for a, b in zip(idx[::2], idx[1::2]):
            mid = (sing[a] + sing[b]) / 2
            sing[a] = mid + glue * (sing[a] - mid)
            sing[b] = mid + glue * (sing[b] - mid)

    X = np.vstack([sing] + [c + b for c, b in zip(ccen, blobs)])
    # отделимость, измеренная: расстояние до ближайшего чужого центра
    Dc = squareform(pdist(ccen))
    np.fill_diagonal(Dc, np.inf)
    delta = Dc.min(1).mean()
    scale = pdist(X).mean()
    return X, delta / (2 * rad), rad / scale


def run(X, rng):
    df = qphd(X, q_list=cfg.Q_GRID, rng=rng,
              **cfg.qphd_kwargs(L=L, pool=X, replicates=16))
    mst = minimum_spanning_tree(squareform(pdist(X))).tocoo()
    w = mst.data / pdist(X).mean()
    longest = np.zeros(len(X))
    for a, b, ww in zip(mst.row, mst.col, w):
        longest[a] = max(longest[a], ww)
        longest[b] = max(longest[b], ww)
    return {**bands_of(df),
            "черешок": float(longest[:N_SING].mean()),
            "разброс": float(longest[:N_SING].std())}


def main():
    knobs = [("база", {}),
             ("отодвинуть +10%", dict(push=0.10)),
             ("отодвинуть −10%", dict(push=-0.10)),
             ("подтянуть 30%", dict(pull_share=0.30)),
             ("подтянуть 60%", dict(pull_share=0.60)),
             ("склеить 50%", dict(glue_share=0.50)),
             ("склеить 100%", dict(glue_share=1.00))]
    rows = []
    for r_target in (0.06, 0.10, 0.232):
        for name, kw in knobs:
            for rep in range(REPS):
                rng = np.random.default_rng(hash((r_target, name, rep)) % 2**31)
                X, g, rad = build(r_target, rng=rng, **kw)
                rows.append({"радиус": r_target, "ручка": name, "g": g,
                             **run(X, rng)})
        print(f"  радиус {r_target}: готово", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv("results/synthetic_regimes.csv", index=False)

    pd.set_option("display.width", 200)
    for r_target, g0 in D.groupby("радиус"):
        base = g0[g0["ручка"] == "база"].mean(numeric_only=True)
        t = g0.groupby("ручка").mean(numeric_only=True)
        out = pd.DataFrame({
            "g": t["g"].round(2),
            "крупная": t["крупный"].round(2),
            "Δкрупная, %": ((t["крупный"] / base["крупный"] - 1) * 100).round(1),
            "Δмелкая, %": ((t["мелкий"] / base["мелкий"] - 1) * 100).round(1),
            "Δчерешок, %": ((t["черешок"] / base["черешок"] - 1) * 100).round(1),
            "Δразброс, %": ((t["разброс"] / base["разброс"] - 1) * 100).round(1),
        }).reindex([n for n, _ in knobs])
        reg = "СТРАТЫ ЧИСТЫ" if base["g"] > 2 else "страты перемешаны"
        print(f"\n=== радиус {r_target}, измеренное g = {base['g']:.2f} "
              f"({reg}) ===")
        print(out.to_string())


if __name__ == "__main__":
    main()
