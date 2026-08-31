"""How far off is the fitted band, in the units the band is measured in.

R2 is a ratio and hides the scale. A model can hold R2 0.75 and still miss a
given text by twenty percentage points if the band itself varies by forty. So
the same fits are reported here as typical error: the median absolute residual,
the 90th percentile, and the same figures for the do-nothing baseline that
always predicts the average. The ratio between them is what R2 summarises.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from band_models import demean, features
from judge_shift import load
from mech_props import NAMES as MECH
from three_bands import BANDS as _B, SUF

BANDS = list(_B)
BASE = os.path.join(HERE, "..")


def main():
    D = load()
    for m in MECH:
        if f"src_{m}" in D:
            D[f"d_{m}"] = D[m] - D[f"src_{m}"]
    cols, label, _ = features(D)
    M = pd.read_csv(os.path.join(BASE, "results", f"band_models_random{SUF}.csv"))
    back = {v: k for k, v in label.items()}

    rows = []
    for band in BANDS:
        for mode in ("все тексты", "внутри"):
            pick = M[(M["полоса"] == band) & (M["режим"] == mode)]["признак"]
            use = [back[p] for p in pick]
            sub = D[[band, "perturbation"] + cols].dropna()
            if mode == "внутри":
                sub = demean(sub, [band] + cols, "perturbation")
            X, y = sub[use], sub[band]
            pred = cross_val_predict(LinearRegression(), X, y,
                                     cv=KFold(5, shuffle=True, random_state=0))
            err = np.abs(y - pred)
            base = np.abs(y - y.mean())
            rows.append({
                "полоса": band, "режим": mode, "n": len(y),
                "разброс полосы (sd)": y.std(),
                "ошибка: медиана": err.median(),
                "ошибка: 90-й проц.": err.quantile(0.9),
                "без модели: медиана": base.median(),
                "без модели: 90-й проц.": base.quantile(0.9),
                "R2": 1 - (err ** 2).sum() / (base ** 2).sum(),
            })
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("всё в процентах сдвига полосы; ошибка — по отложенным текстам\n")
    print(T.round(1).to_string(index=False))
    T.round(2).to_csv(os.path.join(BASE, "results", f"model_error{SUF}.csv"),
                      index=False)


if __name__ == "__main__":
    main()
