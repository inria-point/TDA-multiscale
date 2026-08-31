"""Absolute dimension per band per text, with no reference to a source.

Everything so far has been a paired displacement, which is the right frame for
asking what an operation did. It cannot answer the other question: given a text
and nothing else, what is its dimension. That needs d itself averaged over each
band's q range rather than its per cent change, and it needs the untouched
sources in the sample as ordinary rows rather than as baselines.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
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
        cols = {}
        for name, (mode, qmin) in BANDS.items():
            s = d[(d["mode"] == mode) & (d["q"] >= qmin)]
            cols[name] = s.groupby(["perturbation", "text_id"])["d_hat"].mean()
        frames.append(pd.DataFrame(cols))
    T = pd.concat(frames)
    T = T[~T.index.duplicated(keep="last")].dropna().reset_index()
    path = os.path.join(BASE, "results", f"per_text_bands_abs{SUF}.csv")
    T.round(3).to_csv(path, index=False)
    pd.set_option("display.width", 200)
    print(f"строк: {len(T)}, пертурбаций: {T['perturbation'].nunique()}, "
          f"текстов: {T['text_id'].nunique()}")
    print("\nразмерность d по полосам:")
    print(T[list(BANDS)].describe().loc[
        ["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2).to_string())
    print("\nисходники (identity):")
    print(T[T.perturbation == "identity"][list(BANDS)].describe()
          .loc[["mean", "std", "min", "max"]].round(2).to_string())
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
