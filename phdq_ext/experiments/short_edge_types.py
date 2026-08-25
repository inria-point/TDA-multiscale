"""Composition of the short MST edges, by token identity as well as class.

Short is defined against the text's own mean MST edge length: an edge counts
if it is under THRESH of that mean, so texts of different overall spread are
compared on the same footing.

Every short edge falls into exactly one of six cells, so the table sums to
100% and nothing hides in a remainder:

  both sides function or punctuation, different tokens
  both sides function or punctuation, the same token
  both sides content, the same token
  both sides content, different tokens
  one side function/punctuation, one side content
  (subword fragments are counted as content: they are pieces of content words)
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
from perturb import FUNCTION_WORDS, apply

BASE = os.path.join(HERE, "..")
ALPHA = re.compile(r"^[A-Za-z]+$")
THRESH = 0.33
SKIP = {"[CLS]", "[SEP]", "[PAD]"}


CELLS = ["служ.-служ., один токен", "служ.-служ., разные",
         "смысл.-смысл., один токен", "смысл.-смысл., разные",
         "часть-часть, один токен", "часть-часть, разные", "смешанные"]


def token_class(tok):
    """служ. = punctuation, digits, symbols and function words;
    часть = a word-piece continuation (the tokeniser marks word starts);
    смысл. = a whole content word."""
    w = tok.lstrip("Ġ▁").strip()
    if not ALPHA.match(w):
        return "служ."
    if not tok.startswith(("Ġ", "▁")):
        return "часть"
    return "служ." if w.lower() in FUNCTION_WORDS else "смысл."


def mst_edges(text, emb, L=cfg.L_DEFAULT, seed=0):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v = e[idx]
    t = [toks[i] for i in idx]
    d = np.linalg.norm(v[:, None] - v[None], axis=-1)
    m = minimum_spanning_tree(d).tocoo()
    o = np.argsort(m.data)
    # idx are positions in the original token stream, so the gap between the
    # ends of an edge is how far apart in the text the two tokens stand
    return m.data[o], m.row[o], m.col[o], t, idx, len(keep)


def tally(rows, cols, t, sel, idx=None, n_tokens=None):
    """Every edge lands in exactly one cell, so the row sums to 100%.

    Alongside the shares, the gap in tokens between the two ends of an edge.
    Short in embedding space says the two tokens are alike; this says whether
    they also stand next to each other in the text or a page apart.
    """
    c = dict.fromkeys(CELLS, 0)
    gaps = {k: [] for k in CELLS}
    allg = []
    for a, b in zip(rows[sel], cols[sel]):
        ta, tb = t[a], t[b]
        ca, cb = token_class(ta), token_class(tb)
        key = ("смешанные" if ca != cb
               else f"{ca}-{ca}, " + ("один токен" if ta == tb else "разные"))
        c[key] += 1
        if idx is not None:
            g = abs(int(idx[a]) - int(idx[b]))
            gaps[key].append(g)
            allg.append(g)
    n = max(1, sum(c.values()))
    out = {k: v / n * 100 for k, v in c.items()}
    if idx is not None:
        out["разрыв, токенов"] = float(np.mean(allg)) if allg else np.nan
        out["медиана разрыва"] = float(np.median(allg)) if allg else np.nan
        out["доля рёбер ближе 10 токенов"] = (
            float(np.mean(np.array(allg) < 10)) * 100 if allg else np.nan)
        for k in ["служ.-служ., один токен", "смысл.-смысл., один токен"]:
            out[f"разрыв: {k}"] = (float(np.mean(gaps[k])) if gaps[k]
                                   else np.nan)
    return out


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    texts = [r.text for r in pool.itertuples() if r.is_human][:40]
    emb = Embedder()
    variants = [("человек", None), ("loop/echo6w_x2", "loop_local_1"),
                ("loop/echo6w_x4", "loop_local_3"),
                ("vocab/top60", "collapse_vocab_60"),
                ("vocab/top10", "collapse_vocab_10"),
                ("vocab/expand100", "expand_vocab_100"),
                ("shuffle/words", "shuffle_words"),
                ("surface/punct_add", "add_punctuation")]
    # a cut at a fraction of the mean length also says how many edges have
    # collapsed; a percentile cut always takes the same number, so only the
    # composition varies
    views = {"короче 0.33 средней": None, "20% самых коротких": 0.20,
             "30% самых коротких": 0.30}
    acc = {v: {} for v in views}
    for name, pert in variants:
        got = {v: [] for v in views}
        for i, s_ in enumerate(texts):
            r = mst_edges(s_ if pert is None else apply(pert, s_, seed=i),
                          emb, seed=i)
            if not r:
                continue
            lens, rows, cols, t, idx, n_tok = r
            for vname, frac in views.items():
                if frac is None:
                    sel = lens < THRESH * lens.mean()
                    if sel.sum() == 0:
                        continue
                    rec = tally(rows, cols, t, sel, idx, n_tok)
                    rec["коротких рёбер, % от всех"] = sel.mean() * 100
                else:
                    rec = tally(rows, cols, t,
                                slice(0, max(1, int(frac * len(lens)))),
                                idx, n_tok)
                got[vname].append(rec)
        for vname in views:
            if got[vname]:
                acc[vname][name] = pd.DataFrame(got[vname]).mean()
        print(f"  {name:18s} готово", flush=True)

    pd.set_option("display.width", 240)
    for vname in views:
        tbl = pd.DataFrame(acc[vname]).T
        print(f"\n=== {vname}; проценты — от числа взятых рёбер\n")
        share = [c for c in tbl.columns if not c.startswith(("разрыв",
                                                             "медиана",
                                                             "доля рёбер"))]
        print(tbl[share].round(1).to_string())
        gap = [c for c in tbl.columns if c not in share]
        print("\n    расстояние в тексте между концами ребра, в токенах:")
        print(tbl[gap].round(1).to_string())
        tag = {"короче 0.33 средней": "mean033", "20% самых коротких": "p20",
               "30% самых коротких": "p30"}[vname]
        tbl.round(2).to_csv(os.path.join(BASE, "results",
                                         f"short_edge_types_{tag}.csv"))


if __name__ == "__main__":
    main()
