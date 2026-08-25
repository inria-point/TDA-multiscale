"""Paired before/after for the positional reach of the short MST edges.

Same texts on both sides, so the difference is within-text and does not carry
between-text variation. Two quantities per text:

  gap_tokens  mean distance in the text, in tokens, between the ends of an
              edge, over the 20% shortest edges of the MST
  n_near      how many of those edges have their ends under 10 tokens apart,
              as a count out of the 40 edges taken (L=201 gives 200 MST edges)

The count is reported rather than the share because the denominator is fixed
by construction, and a count says directly how many near-duplicate pairs the
estimator is looking at.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from embedder import Embedder
from perturb import apply

BASE = os.path.join(HERE, "..")
SKIP = {"[CLS]", "[SEP]", "[PAD]"}
NEAR = 10
FRAC = 0.20


def gaps(text, emb, L=cfg.L_DEFAULT, seed=0):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    d = np.linalg.norm(e[idx][:, None] - e[idx][None], axis=-1)
    m = minimum_spanning_tree(d).tocoo()
    o = np.argsort(m.data)[:max(1, int(FRAC * len(m.data)))]
    g = np.abs(idx[m.row[o]] - idx[m.col[o]])
    return {"gap_mean": float(g.mean()), "gap_median": float(np.median(g)),
            "n_near": int((g < NEAR).sum()), "n_edges": len(o)}


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    texts = [r.text for r in pool.itertuples() if r.is_human][:40]
    emb = Embedder()
    variants = [("loop/echo6w_x2", "loop_local_1"),
                ("loop/echo6w_x4", "loop_local_3"),
                ("loop/phrase12w_x4", "loop_phrase"),
                ("loop/tail_keep40_unit3w", "loop_tail_mid"),
                ("vocab/top60", "collapse_vocab_60"),
                ("vocab/top10", "collapse_vocab_10"),
                ("vocab/expand100", "expand_vocab_100"),
                ("shuffle/words", "shuffle_words"),
                ("surface/punct_add", "add_punctuation")]

    base = [gaps(s, emb, seed=i) for i, s in enumerate(texts)]
    rows = []
    for name, pert in variants:
        pairs = []
        for i, s in enumerate(texts):
            a = base[i]
            b = gaps(apply(pert, s, seed=i), emb, seed=i)
            if a and b:
                pairs.append((a, b))
        if not pairs:
            continue
        A = pd.DataFrame([p[0] for p in pairs]).mean()
        B = pd.DataFrame([p[1] for p in pairs]).mean()
        rows.append({
            "пертурбация": name, "n": len(pairs),
            "разрыв до": A["gap_mean"], "разрыв после": B["gap_mean"],
            "разрыв Δ": B["gap_mean"] - A["gap_mean"],
            "медиана до": A["gap_median"], "медиана после": B["gap_median"],
            "рёбер <10 до": A["n_near"], "рёбер <10 после": B["n_near"],
            "рёбер <10 Δ": B["n_near"] - A["n_near"],
            "всего взято рёбер": B["n_edges"],
        })
        print(f"  {name:24s} {len(pairs)}", flush=True)
    tbl = pd.DataFrame(rows).set_index("пертурбация")
    pd.set_option("display.width", 240)
    print(f"\nпарно по {len(texts)} человеческим текстам, "
          f"{int(FRAC * 100)}% кратчайших рёбер MST\n")
    print(tbl.round(1).to_string())
    tbl.round(2).to_csv(os.path.join(BASE, "results", "gap_paired.csv"))


if __name__ == "__main__":
    main()
