"""What the perturbation geometry run says.

Three tables.

1. Shifts. Every coordinate of the cell model, per condition, as a paired
   percentage against the same texts unperturbed.

2. Levers. The two quantities the model says the bands ride on -- the cell
   radius and the supply of singletons -- against the three bands, so that a
   condition can be read as a displacement in that plane.

3. Extrapolation. A regression fitted on the unperturbed texts alone, then
   asked to predict the bands of the perturbed ones from their geometry. Where
   it succeeds the perturbation needs no explanation beyond the model; where it
   fails, the residual is the thing to explain, and its sign says in which
   direction.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..")
SRC_NAME = os.environ.get("SRC", "pert_geometry.csv")
SRC = os.path.join(BASE, "results", SRC_NAME)
TAG = SRC_NAME.replace("pert_geometry", "").replace(".csv", "")
BANDS = ["крупный", "мелкий", "средний"]
BASELINE = "исходный"

COORDS = ["одиночек, %", "тип-одиночек, %", "смысловые: радиус",
          "служебные: радиус", "смысловые: зазор", "смысловые: ячеек",
          "черешок одиночки", "до бл. при k=1", "до бл. при k=2",
          "длинных листьев, %", "оба одиночки / случай", "p10/p50", "p90/p50",
          "p99/p50", "длинные: CV", "рёбра: CV", "стоков, %",
          "доля смысловые, %", "доля служебные, %", "доля пунктуация, %"]


def load():
    D = pd.read_csv(SRC)
    D["тип-одиночек, %"] = ((D["типов всего"] - D["типов k>=2"])
                            / D["типов всего"] * 100)
    return D


def shifts(D):
    """Paired percentage change against the unperturbed text, per condition."""
    base = D[D["условие"] == BASELINE].groupby("текст").mean(numeric_only=True)
    rows = []
    for cond in D["условие"].unique():
        if cond == BASELINE:
            continue
        cur = D[D["условие"] == cond].groupby("текст").mean(numeric_only=True)
        common = base.index.intersection(cur.index)
        r = {"условие": cond, "n": len(common)}
        for col in BANDS + COORDS:
            if col not in cur:
                continue
            a, b = base.loc[common, col], cur.loc[common, col]
            m = a.notna() & b.notna() & (a != 0)
            if m.sum() < 8:
                r[col] = np.nan
                continue
            r[col] = float(((b[m] - a[m]) / a[m]).median() * 100)
            try:
                p = wilcoxon(a[m], b[m]).pvalue
            except ValueError:
                p = 1.0
            if p > 0.01:
                r[col] = 0.0 if abs(r[col]) < 3 else r[col]
            r[col + "_p"] = p
        rows.append(r)
    return pd.DataFrame(rows).set_index("условие")


def extrapolate(D):
    """Fit the bands on the unperturbed texts, predict the perturbed ones."""
    feats = ["одиночек, %", "смысловые: радиус", "служебные: радиус",
             "типов k>=2"]
    W = D.dropna(subset=feats + BANDS).copy()
    X = np.log(W[feats].clip(lower=1e-6))
    X["const"] = 1.0
    fit = W["условие"] == BASELINE
    out, coef = [], {}
    for b in BANDS:
        y = np.log(W[b].clip(lower=1e-6))
        beta, *_ = np.linalg.lstsq(X[fit].values, y[fit].values, rcond=None)
        coef[b] = dict(zip(X.columns, beta.round(3)))
        W["pred_" + b] = np.exp(X.values @ beta)
        r = np.corrcoef(y[fit], np.log(W.loc[fit, "pred_" + b]))[0, 1]
        out.append((b, r ** 2))
    base = W[fit].groupby("текст")[BANDS + ["pred_" + b for b in BANDS]].mean()
    rows = []
    for cond in W["условие"].unique():
        if cond == BASELINE:
            continue
        cur = W[W["условие"] == cond].groupby("текст")[
            BANDS + ["pred_" + b for b in BANDS]].mean()
        common = base.index.intersection(cur.index)
        r = {"условие": cond, "n": len(common)}
        for b in BANDS:
            obs = ((cur.loc[common, b] - base.loc[common, b])
                   / base.loc[common, b]).median() * 100
            prd = ((cur.loc[common, "pred_" + b] - base.loc[common, "pred_" + b])
                   / base.loc[common, "pred_" + b]).median() * 100
            r[b + " набл."] = obs
            r[b + " модель"] = prd
            r[b + " остаток"] = obs - prd
        rows.append(r)
    return pd.DataFrame(rows).set_index("условие"), dict(out), coef


def main():
    D = load()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    n_texts = D[D["условие"] == BASELINE]["текст"].nunique()
    print(f"{SRC}: {n_texts} текстов, {D['условие'].nunique()} условий, "
          f"{D['зерно'].nunique()} зёрен\n")

    print("Исходные тексты, средние значения координат:")
    base = D[D["условие"] == BASELINE].mean(numeric_only=True)
    print(base[BANDS + [c for c in COORDS if c in base]].round(3).to_string())

    S = shifts(D)
    S.to_csv(os.path.join(BASE, "results", f"pert_geometry{TAG}_shifts.csv"))
    print("\n\nСдвиг в процентах, парно, медиана; 0 = незначимо (p>0.01, |сдвиг|<3%)")
    print("\n== полосы ==")
    print(S[BANDS].round(1).to_string())
    print("\n== чем модель их объясняет ==")
    print(S[["одиночек, %", "тип-одиночек, %", "смысловые: радиус",
             "служебные: радиус", "смысловые: зазор", "смысловые: ячеек",
             "черешок одиночки"]].round(1).to_string())
    print("\n== форма распределения длин рёбер ==")
    print(S[["p10/p50", "p90/p50", "p99/p50", "длинные: CV", "рёбра: CV",
             "до бл. при k=1", "до бл. при k=2", "длинных листьев, %",
             "оба одиночки / случай"]].round(1).to_string())
    print("\n== состав выборки ==")
    print(S[["стоков, %", "доля смысловые, %", "доля служебные, %",
             "доля пунктуация, %"]].round(1).to_string())

    E, r2, coef = extrapolate(D)
    E.to_csv(os.path.join(BASE, "results", f"pert_geometry{TAG}_model.csv"))
    print("\n\nМодель: log(полоса) ~ log(одиночки, радиусы, число ячеек),")
    print("подогнана только на неизменённых текстах;")
    print("  R² на них: " + ", ".join(f"{b} {v:.2f}" for b, v in r2.items()))
    for b in BANDS:
        print(f"  {b}: " + ", ".join(f"{k} {v:+.2f}" for k, v in coef[b].items()))
    print()
    print(E[[f"{b} {w}" for b in BANDS for w in ("набл.", "модель", "остаток")]]
          .round(1).to_string())


if __name__ == "__main__":
    main()
