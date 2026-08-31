"""Three band coordinates for every individual perturbed text.

Everything so far has correlated properties against perturbation means -- forty
to eighty points, each an average over a hundred texts. A relation that holds
between the means need not hold within them, and the only way to find out is to
give each text its own coordinates.

The baseline is the same text untouched, so each row is a paired displacement:
what this operation did to this document, not to documents in general.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from three_bands import BANDS, FILES, SUF

BASE = os.path.join(HERE, "..")


def main():
    frames = []
    for fn in FILES:
        path = os.path.join(BASE, "results", fn)
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        d = d[d["d_hat"] > 0]
        base = (d[d["perturbation"] == "identity"]
                .set_index(["text_id", "mode", "q"])["d_hat"].rename("d0"))
        if base.empty:
            continue
        p = d[d["perturbation"] != "identity"].join(
            base, on=["text_id", "mode", "q"]).dropna(subset=["d0"])
        p["rel"] = (p["d_hat"] - p["d0"]) / p["d0"] * 100
        cols = {}
        for name, (mode, qmin) in BANDS.items():
            s = p[(p["mode"] == mode) & (p["q"] >= qmin)]
            cols[name] = s.groupby(["perturbation", "text_id"])["rel"].mean()
        frames.append(pd.DataFrame(cols))
    T = pd.concat(frames)
    T = T[~T.index.duplicated(keep="last")].dropna()
    T = T.reset_index()
    path = os.path.join(BASE, "results", f"per_text_bands{SUF}.csv")
    T.round(2).to_csv(path, index=False)

    pd.set_option("display.width", 200)
    print(f"текстов с координатами: {len(T)}, "
          f"пертурбаций: {T['perturbation'].nunique()}, "
          f"исходников: {T['text_id'].nunique()}")
    print(f"\nразброс по полосам:")
    print(T[list(BANDS)].describe().loc[
        ["mean", "std", "min", "25%", "50%", "75%", "max"]].round(1).to_string())
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
