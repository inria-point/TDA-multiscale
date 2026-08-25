"""What are the shortest MST edges actually made of?

The fine scale is what survives when the longest edges are trimmed away, so
whatever populates the short end of the edge distribution is what q_large
measures. The objection to test: close pairs are always present and are mostly
function words, in which case the fine scale is largely a property of the
grammatical skeleton rather than of the content.

Each token is classified from its surface form -- function word, punctuation
or subword fragment, or content word -- and the edges of the MST are sorted by
length. The share of each class among the shortest edges is reported against
its share among all edges, so a class that is merely frequent does not look
enriched.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from embedder import Embedder
from perturb import FUNCTION_WORDS

BASE = os.path.join(HERE, "..")
ALPHA = re.compile(r"^[A-Za-z]+$")


def classify(tok):
    w = tok.lstrip("Ġ▁").strip()
    if not ALPHA.match(w):
        return "punct"
    if w.lower() in FUNCTION_WORDS:
        return "function"
    if not tok.startswith(("Ġ", "▁")):
        return "subword"
    return "content"


def edge_classes(text, emb, L=cfg.L_DEFAULT, seed=0):
    e, toks = emb.embed(text, return_tokens=True)
    if len(e) < L:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(e), size=L, replace=False)
    v, t = e[idx], [classify(toks[i]) for i in idx]
    d = np.linalg.norm(v[:, None] - v[None], axis=-1)
    mst = minimum_spanning_tree(d).tocoo()
    order = np.argsort(mst.data)
    rows, cols = mst.row[order], mst.col[order]

    out = {}
    n = len(order)
    for name, sel in [("все", slice(None)), ("10% самых коротких",
                                            slice(0, max(1, n // 10))),
                      ("20% самых коротких", slice(0, max(1, n // 5)))]:
        pairs = [(t[a], t[b]) for a, b in zip(rows[sel], cols[sel])]
        share = {}
        for k in ["function", "punct", "subword", "content"]:
            share[k] = np.mean([(x == k) + (y == k) for x, y in pairs]) / 2
        share["обе служебные/пункт."] = np.mean(
            [x in ("function", "punct") and y in ("function", "punct")
             for x, y in pairs])
        out[name] = share
    return out


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    texts = [r.text for r in pool.itertuples() if r.is_human][:40]
    emb = Embedder()
    acc = {}
    for i, s in enumerate(texts):
        r = edge_classes(s, emb, seed=i)
        if not r:
            continue
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    tbl = pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in acc.items()}).T
    pd.set_option("display.width", 200)
    print(f"доля токенов каждого класса среди концов рёбер MST, "
          f"{len(texts)} человеческих текстов, L={cfg.L_DEFAULT}\n")
    print(tbl.round(3).to_string())
    tbl.round(4).to_csv(os.path.join(BASE, "results", "short_edges.csv"))


if __name__ == "__main__":
    main()
