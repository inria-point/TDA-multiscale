"""Do texts exist that keep their structure and still collapse the short edges?

Every perturbation that pulls the short MST edges together in the text is a
loop, and every loop destroys the text. The question is whether that is
necessary or merely how our perturbations happen to work. Genres that repeat
identical tokens a few positions apart while being perfectly well formed --
source code, numeric tables -- answer it directly, and they need no
generation: real examples are on disk.

These are not perturbations, so there is no paired baseline. Absolute values
are reported for each genre against human prose measured the same way.
"""
import glob
import os
import random
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from embedder import Embedder
from qphd import qphd

BASE = os.path.join(HERE, "..")
SKIP = {"[CLS]", "[SEP]", "[PAD]"}
NEAR, FRAC = 10, 0.20


CAP = 520   # every genre truncated to the same token count: the gap is
            # measured in tokens, so a longer text inflates it mechanically


def edge_stats(e, toks, L=cfg.L_DEFAULT, seed=0):
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    d = np.linalg.norm(e[idx][:, None] - e[idx][None], axis=-1)
    m = minimum_spanning_tree(d).tocoo()
    o = np.argsort(m.data)
    k = max(1, int(FRAC * len(m.data)))
    g = np.abs(idx[m.row[o[:k]]] - idx[m.col[o[:k]]])
    return {"разрыв": float(g.mean()), "медиана": float(np.median(g)),
            "разрыв, % длины": float(g.mean()) / len(keep) * 100,
            "рёбер <10": float((g < NEAR).sum()),
            "схлопнувшихся, %": float((m.data < 0.33 * m.data.mean()).mean())
            * 100}


def dims(e, seed=0, L=cfg.L_DEFAULT):
    if e.shape[0] < int(1.3 * L):
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(e.shape[0], size=L, replace=False)
    df = qphd(e[idx], q_list=(0.0, 0.8), rng=rng,
              **cfg.qphd_kwargs(L=L, pool=cfg.make_pool(e, L, rng),
                                replicates=16))
    g = df.set_index(["mode", "q"])["d_hat"]
    return {"d общий": g[("q_small", 0.0)],
            "d мелкий": g[("q_large", 0.8)],
            "d крупный": g[("q_small", 0.8)]}


def python_sources(n=40, min_words=320):
    """Chunks of real library source, whole functions where possible."""
    files = []
    for pat in ["numpy", "pandas", "scipy", "sklearn", "matplotlib"]:
        files += glob.glob(os.path.join(BASE, ".venv", "lib", "python*",
                                        "site-packages", pat, "**", "*.py"),
                           recursive=True)
    random.Random(0).shuffle(files)
    out = []
    for f in files:
        try:
            src = open(f, encoding="utf-8").read()
        except Exception:
            continue
        lines = [l for l in src.splitlines() if l.strip()]
        if len(" ".join(lines).split()) < min_words:
            continue
        out.append("\n".join(lines[:400]))
        if len(out) >= n:
            break
    return out


def numeric_tables(n=40, min_words=320):
    """Real result tables rendered as aligned text: labels and numbers repeat
    at a fixed short period, which is what a table is."""
    out = []
    for f in sorted(glob.glob(os.path.join(BASE, "results", "*.csv"))):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if d.shape[0] < 15 or d.shape[1] < 3:
            continue
        rep = []
        for _, r in d.iterrows():
            rep.append("; ".join(f"{c}: {r[c]}" for c in d.columns))
        t = "\n".join(rep)
        while len(t.split()) < min_words and len(rep) > 1:
            t = t + "\n" + t
        out.append(t)
        if len(out) >= n:
            break
    return out


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    human = [r.text for r in pool.itertuples() if r.is_human][:40]
    genres = {"человеческая проза": human, "код (Python)": python_sources(),
              "таблица чисел": numeric_tables()}

    emb = Embedder()
    rows = []
    for name, texts in genres.items():
        acc = []
        for i, t in enumerate(texts):
            e, toks = emb.embed(t, return_tokens=True)
            e, toks = e[:CAP], toks[:CAP]
            s = edge_stats(e, toks, seed=i)
            d = dims(e, seed=i)
            if s and d:
                acc.append({**s, **d})
        if not acc:
            print(f"  {name}: нет подходящих текстов")
            continue
        rec = pd.DataFrame(acc).mean().to_dict()
        rec["текстов"] = len(acc)
        rows.append(pd.Series(rec, name=name))
        print(f"  {name:22s} {len(acc)} текстов", flush=True)
    tbl = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\nабсолютные значения по жанрам, L=201\n")
    print(tbl.round(1).to_string())
    tbl.round(2).to_csv(os.path.join(BASE, "results", "genre_reference.csv"))


if __name__ == "__main__":
    main()
