"""Did the long edges get longer gaps, or did every edge?

The long edges of the absurd text span 13 more tokens than their counterparts
in the true one. Two different things produce that. Either the whole graph
reaches further -- every edge joins more distant positions, and the long end
inherits it -- or the total travel is unchanged and the long-gap edges simply
climbed the length ranking, displacing something else.

The second is what the account predicts: with no subject holding it, a distant
repetition stops being drawn together, so it moves outward while nothing else
about the text's positional structure changes.

Measured over the whole MST, not a slice of it, so the two can be told apart.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.sparse.csgraph import minimum_spanning_tree

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from embedder import Embedder
from halluc_result import CJK
from long_edge_types import SKIP

BASE = os.path.join(HERE, "..")
PAIR = ("chained", "absurd")
NEAR = 10


def per_text(text, emb, L=cfg.L_DEFAULT, seed=0):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v = e[idx]
    d = np.linalg.norm(v[:, None] - v[None], axis=-1)
    m = minimum_spanning_tree(d).tocoo()
    gap = np.abs(idx[m.row] - idx[m.col]).astype(float)
    order = np.argsort(m.data)
    n = len(order)
    third = max(1, int(0.30 * n))
    # texts run from 269 to 1235 tokens, so a gap of 150 means different things
    # in different documents; the share of the text spanned is comparable and
    # the absolute count is kept alongside for reference
    span = float(len(keep))
    g = gap / span * 100
    return {
        "все рёбра: разрыв, % длины": g.mean(),
        "все рёбра: медиана, % длины": float(np.median(g)),
        "длинные 30%: разрыв, % длины": g[order[-third:]].mean(),
        "короткие 30%: разрыв, % длины": g[order[:third]].mean(),
        "рёбер с разрывом < 10 токенов, %": float((gap < NEAR).mean()) * 100,
        "связь длины и разрыва (rho)": st.spearmanr(m.data, gap).statistic,
        "длина текста, токенов": span,
        "все рёбра: разрыв, токенов": gap.mean(),
    }


def main():
    texts = {v: json.load(open(os.path.join(
        BASE, "results", f"halluc_{v}.json")))["answers"] for v in PAIR}
    bad = {q for q, t in json.load(open(os.path.join(
        BASE, "results", "halluc_raw.json")))["answers"].items()
        if len(CJK.findall(t)) > 20}
    qids = sorted((set(texts[PAIR[0]]) & set(texts[PAIR[1]])) - bad)
    emb = Embedder()

    rows = []
    for i, q in enumerate(qids):
        a, b = (per_text(texts[v][q], emb, seed=i) for v in PAIR)
        if a and b:
            rows.append({"qid": q, **{f"верно::{k}": v for k, v in a.items()},
                         **{f"абсурд::{k}": v for k, v in b.items()}})
        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{len(qids)}", flush=True)
    D = pd.DataFrame(rows).set_index("qid")
    D.to_csv(os.path.join(BASE, "results", "absurd_gap.csv"))

    keys = [k.split("::", 1)[1] for k in D.columns if k.startswith("верно::")]
    out = []
    for k in keys:
        a, b = D[f"верно::{k}"], D[f"абсурд::{k}"]
        d = (b - a).dropna()
        out.append({"признак": k, "факты верны": a.mean(),
                    "абсурд": b.mean(), "сдвиг": d.mean(),
                    "выросло у": (d > 0).mean(),
                    "p": st.wilcoxon(d).pvalue})
    pd.set_option("display.width", 200)
    print(f"\n{len(D)} пар, весь MST (L={cfg.L_DEFAULT} точек, "
          f"{cfg.L_DEFAULT - 1} рёбер)\n")
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
