"""Repair the formatting and see whether the coarse band comes back.

The thirteen human documents with the coarse band down alone are full of
detokenisation: spaces before punctuation, clitics split off, stray list
punctuation left behind. I read that as the cause. The alternative reading is
that these documents are fragmentary in their sense and the spacing is
incidental.

The two readings differ in a way that can be settled: repair the formatting and
nothing else. If the band recovers, the writing was the cause. If it does not,
the writing was a symptom and something in the content is doing the work.

The repair is mechanical rather than a rewrite, so that it provably touches
only spacing and punctuation -- no word is added, removed or reordered, and
that is checked afterwards by comparing the sequence of words. The same repair
is applied to the control documents, since a repair that moves clean text too
would prove nothing.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from embedder import Embedder
from qphd import qphd
from three_bands import BANDS

BASE = os.path.join(HERE, "..")
B = list(BANDS)
CLITICS = r"n't|'s|'re|'ve|'ll|'d|'m|'t"


def repair(t):
    """Undo detokenisation and scrape residue. Words are never touched."""
    t = re.sub(r"\s+(" + CLITICS + r")\b", r"\1", t)      # do n't -> don't
    t = re.sub(r"\s+([,.;:!?%])", r"\1", t)               # word , -> word,
    t = re.sub(r"([(\[])\s+", r"\1", t)                   # ( x -> (x
    t = re.sub(r"\s+([)\]])", r"\1", t)                   # x ) -> x)
    t = re.sub(r"\s+-\s+", "-", t)                        # story - relevant
    t = re.sub(r"([.!?]);\s*,", r"\1", t)                 # wall.; ,  -> wall.
    t = re.sub(r"(?<=[.!?])\s*,+\s*", " ", t)             # . , ,  -> .
    t = re.sub(r",{2,}", ",", t)                          # ,, -> ,
    t = re.sub(r"\s*,\s*(?=[A-Z][a-z]+ (?:you|the|it|if|are|is))", ". ", t)
    t = re.sub(r"([a-z])\.([A-Z])", r"\1. \2", t)         # film.The -> film. The
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def letters(t):
    """Everything but letters is stripped, so the check is blind to exactly
    what the repair changes and sensitive to everything it must not.

    Comparing word sequences instead silently dropped the worst-damaged texts:
    joining `do n't` into `don't` changes the word split, so the documents the
    repair helped most were the ones excluded from the sample.
    """
    return re.sub(r"[^a-z]", "", t.lower())


def bands_of(text, emb, L=cfg.L_DEFAULT, seed=0):
    e = emb.embed_cached(text)
    if e.shape[0] < L:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(e.shape[0], size=L, replace=False)
    df = qphd(e[idx], q_list=cfg.Q_GRID, rng=rng,
              **cfg.qphd_kwargs(L=L, pool=cfg.make_pool(e, L, rng),
                                replicates=10))
    out = {}
    for name, (mode, qmin) in BANDS.items():
        qmax = 0.5 if mode == "q0.5_range" else 0.9
        s = df[(df["mode"] == mode) & df["q"].between(qmin, qmax)
               & (df["d_hat"] > 0)]
        out[name] = float(s["d_hat"].mean()) if len(s) else np.nan
    return out


def main():
    H = pd.concat([pd.read_csv(os.path.join(BASE, "results", f)) for f in
                   ("coarse_sign.csv", "coarse_sign_мелкий.csv",
                    "coarse_sign_средний.csv")]).drop_duplicates("id")
    P = H[[f"откл_{b}" for b in B]].values
    H["профиль"] = ["".join("0" if v == 0 else ("+" if v > 0 else "−")
                            for v in r)
                    for r in np.sign(P) * np.clip(np.abs(P) - 20, 0, None)]
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    pool["id"] = pool["id"].astype(str)
    H["id"] = H["id"].astype(str)
    H = H.merge(pool[["id", "text"]], on="id")
    S = H[H["профиль"].isin(["−00", "−0−", "−−−", "000"])].copy()
    S["группа"] = np.where(S["профиль"] == "000", "контроль", S["профиль"])
    emb = Embedder()

    rows = []
    for i, r in enumerate(S.itertuples()):
        fixed = repair(r.text)
        if letters(fixed) != letters(r.text):
            print(f"  пропуск {r.id[:8]}: правка изменила буквы")
            continue
        a, b = bands_of(r.text, emb, seed=i), bands_of(fixed, emb, seed=i)
        if not a or not b:
            continue
        rows.append({"id": r.id, "группа": r.группа, "sub_source": r.sub_source,
                     "правок": len(r.text) - len(fixed),
                     **{f"до_{k}": v for k, v in a.items()},
                     **{f"после_{k}": v for k, v in b.items()}})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(S)}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "detok_repair.csv"), index=False)

    from scipy import stats as st
    pd.set_option("display.width", 200)
    print(f"\n{len(D)} документов; поток букв не изменился ни в одном\n")
    for g, d in D.groupby("группа"):
        print(f"=== {g}  (n={len(d)}, снято символов: медиана "
              f"{d['правок'].median():.0f})")
        for b in B:
            x, y = d[f"до_{b}"], d[f"после_{b}"]
            rel = (y - x) / x * 100
            print(f"   {b:8s} {x.mean():6.2f} -> {y.mean():6.2f}  "
                  f"({rel.mean():+5.1f}%, медиана {rel.median():+5.1f}%, "
                  f"p={st.wilcoxon(y - x).pvalue:.4f})")


if __name__ == "__main__":
    main()
