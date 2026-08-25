"""Is a model's point a real place, or the mean of a mixture?

A generator is drawn on the map at the mean of its texts. If it loops on some
of them and writes normally on the rest, that mean can sit where no text of
its own actually is, and matching a perturbation to it would be matching an
artefact of averaging.

Each text is therefore given its own coordinates in the same basis, and each
text is scored for repetition directly from its words. The question is whether
the per-text cloud of a model is one blob around its mean or two.

Repetition is measured as the share of word trigrams that occur more than once
-- a loop drives it towards one, ordinary prose keeps it near zero, and it
needs no embedding.
"""
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from pc_space import ALL_MODES, generator_profiles, perturbation_profiles

BASE = os.path.join(HERE, "..")
WORD = re.compile(r"[a-z']+")


def basis():
    pert = perturbation_profiles(os.path.join(BASE, "results",
                                              "perturb_L201_coling_all.csv.gz"))
    gen, _ = generator_profiles(os.path.join(BASE, "results",
                                             "coling_qphd_L201.csv.gz"))
    cols = [c for c in pert.columns if c in gen.columns]
    X = pd.concat([pert[cols], gen[cols]]).fillna(0)
    _, _, vt = np.linalg.svd(X.values, full_matrices=False)
    return vt[:3], cols


def trigram_repeat(text):
    w = WORD.findall(text.lower())
    if len(w) < 30:
        return np.nan
    tri = list(zip(w, w[1:], w[2:]))
    c = Counter(tri)
    return sum(v for v in c.values() if v > 1) / len(tri)


def main():
    vt, cols = basis()
    d = pd.read_csv(os.path.join(BASE, "results", "coling_qphd_L201.csv.gz"))
    d = d[d["d_hat"] > 0].copy()
    h = (d[d["is_human"]].groupby(["sub_source", "mode", "q"])["d_hat"]
         .mean().rename("h"))
    d = d.join(h, on=["sub_source", "mode", "q"]).dropna(subset=["h"])
    d["rel"] = (d["d_hat"] - d["h"]) / d["h"] * 100

    parts = []
    for m in ALL_MODES:
        piv = (d[d["mode"] == m]
               .pivot_table(index=["model", "id"], columns="q",
                            values="rel"))
        piv.columns = [f"{m}@{c:g}" for c in piv.columns]
        parts.append(piv)
    P = pd.concat(parts, axis=1).reindex(columns=cols).fillna(0)
    C = pd.DataFrame(P.values @ vt.T, index=P.index,
                     columns=["PC1", "PC2", "PC3"]).reset_index()

    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    rep = {str(r.id): trigram_repeat(r.text) for r in pool.itertuples()}
    C["rep"] = C["id"].astype(str).map(rep)
    C = C.dropna(subset=["rep"])
    C.round(3).to_csv(os.path.join(BASE, "results", "per_text_pc.csv"),
                      index=False)

    # the cut is the 95th percentile of human text, so "repetitive" means
    # repetitive by the standard of the corpus and not by a round number
    THRESH = C.loc[C["model"] == "human", "rep"].quantile(0.95)
    print(f"порог повторяемости = 95-й процентиль человека = {THRESH:.3f}\n")

    n = C.groupby("model").size()
    keep = n[n >= 20].index.difference(["human"])
    pd.set_option("display.width", 220)
    rows = []
    for m in keep:
        g = C[C["model"] == m]
        loop = g["rep"] > THRESH
        rows.append({
            "модель": m, "n": len(g),
            "PC1": g["PC1"].mean(), "PC2": g["PC2"].mean(),
            "PC2 медиана": g["PC2"].median(), "PC2 σ": g["PC2"].std(),
            "повторных": int(loop.sum()),
            "rep медиана": g["rep"].median(),
            "PC1 без них": g.loc[~loop, "PC1"].mean(),
            "PC2 без них": g.loc[~loop, "PC2"].mean(),
        })
    R = pd.DataFrame(rows).sort_values("PC2")
    print("покоординатно по текстам; повтор = доля триграмм, "
          "встретившихся больше раза\n")
    print(R.round(1).to_string(index=False))
    R.round(2).to_csv(os.path.join(BASE, "results", "model_mixture.csv"),
                      index=False)


if __name__ == "__main__":
    main()
