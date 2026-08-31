"""Subtract what the counted statistics explain, and see what is left.

The absurdity experiment worked this way by hand: the middle band appeared not
to respond because a lexical enrichment of +3.1% masked a real -3.9%, and only
subtracting the prediction revealed it. If that is worth doing once it is worth
doing everywhere -- the residual band is the part of the displacement that no
word count accounts for, which is where a semantic effect would have to live.

Predictions are out-of-fold, so a text never contributes to the model that
explains it and the residual is not shrunk by overfitting.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from judge import PROPS
from judge_shift import MIN_N, load, weighted_within
from mech_props import NAMES as MECH
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from three_bands import BANDS as _B, SUF

BANDS = list(_B)
BASE = os.path.join(HERE, "..")


def main():
    D = load()
    for m in MECH:
        if f"src_{m}" in D:
            D[f"d_{m}"] = D[m] - D[f"src_{m}"]
    mech = [f"d_{m}" for m in MECH if f"d_{m}" in D]
    judge = [f"d_{k}" for k in PROPS if f"d_{k}" in D]
    D = D.dropna(subset=BANDS + mech + judge).reset_index(drop=True)
    print(f"{len(D)} текстов, {len(mech)} механических признаков\n")

    cv = KFold(5, shuffle=True, random_state=0)
    for band in BANDS:
        pred = cross_val_predict(LinearRegression(), D[mech], D[band], cv=cv)
        D[f"ост_{band}"] = D[band] - pred
        var = 1 - D[f"ост_{band}"].var() / D[band].var()
        print(f"{band:8s} механика объясняет {var:5.0%} дисперсии; "
              f"sd полосы {D[band].std():.1f} -> остатка "
              f"{D[f'ост_{band}'].std():.1f}")

    rows = []
    for k in PROPS:
        c = f"d_{k}"
        if c not in D:
            continue
        r = {"признак": PROPS[k]["ru"]}
        for band in BANDS:
            for tag, col in (("полоса", band), ("остаток", f"ост_{band}")):
                g = D[[c, col]].dropna()
                r[f"{band}: {tag}"] = g[c].corr(g[col], method="spearman")
                r[f"{band}: {tag}, внутри"] = weighted_within(D, c, col)
        rows.append(r)
    T = pd.DataFrame(rows)
    T.round(3).to_csv(os.path.join(BASE, "results",
                                   f"band_residual{SUF}.csv"), index=False)
    pd.set_option("display.width", 240)
    for band in BANDS:
        cols = [f"{band}: полоса", f"{band}: остаток",
                f"{band}: полоса, внутри", f"{band}: остаток, внутри"]
        S = T[["признак"] + cols].copy()
        S.columns = ["признак", "полоса", "остаток", "полоса, внутри",
                     "остаток, внутри"]
        S = S.reindex(S["остаток"].abs().sort_values(ascending=False).index)
        print(f"\n=== {band.upper()}: судейские сдвиги против полосы и остатка")
        print(S.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
