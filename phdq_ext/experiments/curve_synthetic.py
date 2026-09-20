"""Положительный контроль: где живёт влияние каркаса, когда страты чисты.

На тексте отделимость g = 1.60, расщепление размыто (4.5% рёбер не на своей
стороне), и предсказание «каркасные величины влияют только выше q*» не
гарантировано. Здесь тот же замер делается на синтетике при g ≈ 5, где теорема
выполняется точно: если там кривая влияния — ступенька на q*, а на тексте
пологий склон, разница и есть цена размытости.

Меряется так: строятся пары облаков «база» и «возмущённое» с одним зерном,
считаются полные кривые d̂(q), и для каждой точки (режим, q) берётся средний
относительный сдвиг d̂ по повторам. Возмущения — те, что действуют на каркас:
раздвинуть все одиночки и спарить их. Для сравнения берётся возмущение ячеек
(раздуть радиус), которое обязано действовать по другую сторону q*.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
sys.path.insert(0, "scripts")
import config as cfg
from qphd import qphd
from synthetic_regimes import N_CELL, N_SING, build

REPS = 10
R_CLEAN = 0.06


def curve(X, rng):
    df = qphd(X, q_list=cfg.Q_GRID, rng=rng,
              **cfg.qphd_kwargs(L=cfg.L_DEFAULT, pool=X, replicates=16))
    return df.set_index(["mode", "q"])["d_hat"]


def main():
    # q* для этой конструкции: N = число ячеек + число одиночек
    # ячейки дают N_CELL центров, одиночки — N_SING; n = L
    q_star = (cfg.L_DEFAULT - (N_CELL + N_SING)) / (cfg.L_DEFAULT - 1)
    print(f"синтетика: {N_SING} одиночек + {N_CELL} ячеек, "
          f"q* = {q_star:.3f}\n")
    cases = [("раздвинуть +15%", dict(r_target=R_CLEAN, push=0.15)),
             ("спарить 100%", dict(r_target=R_CLEAN, glue_share=1.00)),
             ("раздуть ячейки ×2", dict(r_target=2 * R_CLEAN))]
    out = {}
    for name, kw in cases:
        acc = []
        for rep in range(REPS):
            s = hash((name, rep)) % 2 ** 31
            X0, g0, _ = build(R_CLEAN, rng=np.random.default_rng(s))
            X1, g1, _ = build(rng=np.random.default_rng(s), **kw)
            c0 = curve(X0, np.random.default_rng(s))
            c1 = curve(X1, np.random.default_rng(s))
            acc.append((c1 - c0) / c0 * 100)
        out[name] = pd.concat(acc, axis=1).mean(axis=1)
        print(f"{name}: g базы {g0:.2f} → {g1:.2f}")
    D = pd.DataFrame(out)
    D.to_csv("results/curve_synthetic.csv")
    pd.set_option("display.width", 200)
    for mode in ("q_small", "q_large", "q0.5_range"):
        sub = D.loc[mode]
        print(f"\n=== {mode} ===   (q* = {q_star:.3f})")
        print("  сдвиг d̂, %, по точкам q")
        print(sub.round(1).to_string())


if __name__ == "__main__":
    main()
