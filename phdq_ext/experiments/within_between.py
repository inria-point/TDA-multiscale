"""Does a relation between perturbations survive inside one?

Every correlation in this project so far has been over perturbation means. A
relation can be strong between group means and absent within groups: that
would say the operations differ in both the property and the dimension, not
that the property moves the dimension. The two are told apart by computing the
correlation separately inside each perturbation and averaging.

Properties are taken as paired shifts against the same source text wherever a
source measurement exists, so a text that began richer than another does not
masquerade as an effect.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from judge import PROPS
from mech_props import NAMES as MECH
from three_bands import BANDS, SUF

BASE = os.path.join(HERE, "..")
MIN_N = 15


def main():
    D = pd.read_csv(os.path.join(BASE, "results", f"annotated{SUF}.csv"))
    # mechanical properties as paired shifts where the source was measured
    for m in MECH:
        if f"src_{m}" in D:
            D[f"d_{m}"] = D[m] - D[f"src_{m}"]
    feats = ([(k, PROPS[k]["ru"], "судья") for k in PROPS] +
             [(f"d_{m}", MECH[m] + " (сдвиг)", "механика") for m in MECH
              if f"d_{m}" in D] +
             [(m, MECH[m], "механика") for m in MECH if m in D])

    rows = []
    for key, ru, kind in feats:
        if key not in D:
            continue
        rec = {"признак": ru, "тип": kind}
        for band in BANDS:
            s = D[[key, band, "perturbation"]].dropna()
            if len(s) < 50:
                continue
            rec[f"{band} общая"] = spearmanr(s[key], s[band])[0]
            inner, weights = [], []
            for _, g in s.groupby("perturbation"):
                if g[key].nunique() < 3 or len(g) < MIN_N:
                    continue
                r = spearmanr(g[key], g[band])[0]
                if not np.isnan(r):
                    inner.append(r)
                    weights.append(len(g))
            rec[f"{band} внутри"] = (np.average(inner, weights=weights)
                                     if inner else np.nan)
            rec[f"{band} групп"] = len(inner)
        rows.append(rec)
    R = pd.DataFrame(rows).set_index("признак")
    R.round(3).to_csv(os.path.join(BASE, "results",
                                   f"within_between{SUF}.csv"))

    pd.set_option("display.width", 210)
    pd.set_option("display.max_rows", 60)
    for band in BANDS:
        cols = [c for c in R.columns if c.startswith(band) or c == "тип"]
        sub = R[cols].copy()
        sub.columns = ["тип", "общая", "внутри", "групп"]
        sub = sub.dropna(subset=["общая"]).reindex(
            sub["общая"].abs().sort_values(ascending=False).index)
        print(f"\n=== {band.upper()}")
        print(sub.head(12).round(2).to_string())


if __name__ == "__main__":
    main()
