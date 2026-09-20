"""Складывается ли перемешивание из двух геометрических слагаемых?

Измерено на тексте (без стоков): перемешивание даёт крупную −12.9%, мелкую
+43.8%. Синтетика с раздуванием ячеек при ПРИНУДИТЕЛЬНО неподвижном каркасе даёт
−7.2% и +31.2%, то есть 56% и 71% эффекта. Остаток должен приходить от того, что
каркас на самом деле не стоит.

Чем он деформируется, теперь известно: снос вдоль оси определённости
класс-зависим (token_projection.csv, все p = 4·10⁻¹⁸),

    пунктуация  +0.187      подслова  +0.218
    смысловые   +0.32       служебные +0.389

в долях средней парной дистанции. Разность крайних классов 0.20 при типичном
каркасном ребре 0.76 — четверть ребра. Общая часть этого сноса есть перенос и
геометрически инертна; работает только разность между классами, то есть
анизотропная деформация каркаса.

Здесь оба слагаемых задаются в синтетике по отдельности и вместе:

  только ячейки       радиус ×1.17
  только классы       сдвиг вдоль одного направления, своя величина на класс
  оба                 и то, и другое

Если «оба» даёт около −12.9% и +43.8%, перемешивание объяснено целиком.
"""
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

sys.path.insert(0, "experiments")
sys.path.insert(0, "scripts")
import config as cfg
from qphd import qphd
from sink_cluster import bands_of
from synthetic_regimes import DC, KS, N_CELL, N_SING

REPS = 12
RAD = 0.232
# доли классов среди токенов и их сдвиг вдоль оси, измеренные на тексте
CLASSES = [("пунктуация", 0.14, 0.187), ("служебные", 0.37, 0.389),
           ("подслова", 0.10, 0.218), ("смысловые", 0.39, 0.320)]


def build(rng, r_mult=1.0, class_shift=0.0):
    ks = rng.choice(KS, N_CELL)
    need = cfg.L_DEFAULT - N_SING
    while ks.sum() < need:
        ks[rng.integers(N_CELL)] += 1
    while ks.sum() > need:
        j = rng.integers(N_CELL)
        if ks[j] > 2:
            ks[j] -= 1
    cen = rng.normal(size=(N_SING + N_CELL, DC))
    sing, ccen = cen[:N_SING], cen[N_SING:]
    blobs = [rng.normal(size=(k, DC)) for k in ks]
    X0 = np.vstack([sing] + [c + b for c, b in zip(ccen, blobs)])
    r0 = np.mean([np.linalg.norm(b - b.mean(0), axis=1).mean() for b in blobs])
    f = RAD * r_mult * pdist(X0).mean() / r0
    parts = [sing] + [c + b * f for c, b in zip(ccen, blobs)]
    owner = [None] * N_SING + list(range(N_CELL))
    X = np.vstack(parts)

    if class_shift:
        # класс приписывается ТИПУ, значит вся ячейка едет целиком
        p = np.array([c[1] for c in CLASSES])
        s = np.array([c[2] for c in CLASSES])
        n_types = N_SING + N_CELL
        cls = rng.choice(len(CLASSES), size=n_types, p=p / p.sum())
        u = rng.normal(size=DC)
        u /= np.linalg.norm(u)
        scale = pdist(X).mean()
        shift = np.zeros_like(X)
        k = 0
        for j in range(N_SING):
            shift[k] = s[cls[j]] * class_shift * scale * u
            k += 1
        for i, kk in enumerate(ks):
            shift[k:k + kk] = s[cls[N_SING + i]] * class_shift * scale * u
            k += kk
        X = X + shift
    return X


def bands(X, rng):
    return bands_of(qphd(X, q_list=cfg.Q_GRID, rng=rng,
                         **cfg.qphd_kwargs(L=cfg.L_DEFAULT, pool=X,
                                           replicates=16)))


def main():
    cases = [("база", dict()),
             ("только ячейки ×1.17", dict(r_mult=1.17)),
             ("только классы", dict(class_shift=1.0)),
             ("оба", dict(r_mult=1.17, class_shift=1.0)),
             ("оба, классы ×1.5", dict(r_mult=1.17, class_shift=1.5))]
    rows = []
    for name, kw in cases:
        for rep in range(REPS):
            rng = np.random.default_rng(1000 + rep)
            X = build(np.random.default_rng(1000 + rep), **kw)
            rows.append({"случай": name, **bands(X, rng)})
    D = pd.DataFrame(rows).groupby("случай").mean()
    base = D.loc["база"]
    out = pd.DataFrame({
        "крупная": D["крупный"].round(2),
        "Δкрупная, %": ((D["крупный"] / base["крупный"] - 1) * 100).round(1),
        "мелкая": D["мелкий"].round(2),
        "Δмелкая, %": ((D["мелкий"] / base["мелкий"] - 1) * 100).round(1),
        "Δсредняя, %": ((D["средний"] / base["средний"] - 1) * 100).round(1),
    }).reindex([n for n, _ in cases])
    D.to_csv("results/shuffle_synthesis.csv")
    print(out.to_string())
    print("\n  реальность (перемешивание, без стоков): "
          "крупная −12.9%, мелкая +43.8%, средняя −2.6%")


if __name__ == "__main__":
    main()
