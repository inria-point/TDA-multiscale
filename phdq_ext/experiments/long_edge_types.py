"""What are the long MST edges made of?

The coarse scale is what survives when the short edges are trimmed, and it is
the half of PC2 that nothing in the perturbation set explains: the instruct
models raise it by 15-23% and our strongest lever reaches 12%. The short end
turned out to be almost entirely repetition -- 94.5% of the shortest edges
join two occurrences of one token, and two different content words are never
close. This asks the mirror question.

Reported per text, over the longest 20% of the MST edges:

  class composition   the same six cells as at the short end, so the two ends
                      of the distribution are directly comparable
  rank                mean corpus frequency rank of the endpoints, against the
                      mean over all tokens of the same text: a long edge
                      anchored on rare vocabulary would show up here
  gap                 distance in the text between the ends, in tokens
  hub share           the share of long edges that meet at the busiest 10% of
                      tokens: a few outlying tokens carrying most of the long
                      edges is a different geometry from a diffuse spread
"""
import os
import re
import sys
from collections import Counter

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
WORD = re.compile(r"[a-z']+")
SKIP = {"[CLS]", "[SEP]", "[PAD]"}
FRAC = 0.20
# the same seven cells as at the short end, so both ends of the distribution
# are read off one partition. Splitting same-token from different-token inside
# every class is what matters: at the short end almost every service-service
# edge joins a token to itself, at the long end almost none do.
CELLS = ["служ.-служ., один токен", "служ.-служ., разные",
         "смысл.-смысл., один токен", "смысл.-смысл., разные",
         "часть-часть, один токен", "часть-часть, разные", "смешанные"]


def token_class(tok):
    w = tok.lstrip("Ġ▁").strip()
    if not ALPHA.match(w):
        return "служ."
    if not tok.startswith(("Ġ", "▁")):
        return "часть"
    return "служ." if w.lower() in FUNCTION_WORDS else "смысл."


def stats(text, emb, ranks, L=cfg.L_DEFAULT, seed=0, long_end=True):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v, t = e[idx], [toks[i] for i in idx]
    d = np.linalg.norm(v[:, None] - v[None], axis=-1)
    m = minimum_spanning_tree(d).tocoo()
    o = np.argsort(m.data)
    k = max(1, int(FRAC * len(o)))
    sel = o[-k:] if long_end else o[:k]
    rows, cols = m.row[sel], m.col[sel]

    c = dict.fromkeys(CELLS, 0)
    for a, b in zip(rows, cols):
        ta, tb, ca, cb = t[a], t[b], token_class(t[a]), token_class(t[b])
        if ca != cb:
            c["смешанные"] += 1
        else:
            c[f"{ca}-{ca}, " + ("один токен" if ta == tb else "разные")] += 1
    n = max(1, sum(c.values()))
    out = {kk: vv / n * 100 for kk, vv in c.items()}

    def rank_of(tok):
        w = tok.lstrip("Ġ▁").lower()
        return ranks.get(w, np.nan)

    # rank is compared against the content words of the same text, not all its
    # tokens: function words are so frequent that including them drags the
    # baseline down and makes any endpoint look rare by comparison
    ends = [rank_of(t[i]) for i in np.concatenate([rows, cols])
            if token_class(t[i]) == "смысл."]
    allr = [rank_of(x) for x in t if token_class(x) == "смысл."]
    ends = [x for x in ends if x == x]
    allr = [x for x in allr if x == x]
    out["лог-ранг знам. концов"] = (float(np.mean(np.log(ends))) if ends
                                    else np.nan)
    out["лог-ранг знам. текста"] = (float(np.mean(np.log(allr))) if allr
                                    else np.nan)
    full = Counter(toks[j] for j in keep)
    hap = np.array([full[x] == 1 for x in t])
    out["хотя бы один hapax, %"] = float((hap[rows] | hap[cols]).mean()) * 100
    out["разрыв"] = float(np.mean(np.abs(idx[rows] - idx[cols])))
    deg = Counter(np.concatenate([rows, cols]).tolist())
    top = sorted(deg.values(), reverse=True)[:max(1, L // 10)]
    out["на 10% узлов, % рёбер"] = sum(top) / (2 * len(rows)) * 100
    return out


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    counts = Counter()
    for t in hum:
        counts.update(WORD.findall(t.lower()))
    ranks = {w: i + 1 for i, (w, _) in enumerate(counts.most_common())}
    texts = hum[:40]

    emb = Embedder()
    variants = [("человек, ДЛИННЫЕ", None, True),
                ("человек, короткие", None, False),
                ("loop/echo6w_x4", "loop_local_3", True),
                ("vocab/top10", "collapse_vocab_10", True),
                ("vocab/expand100", "expand_vocab_100", True),
                ("shuffle/words", "shuffle_words", True)]
    rows = []
    for name, pert, long_end in variants:
        acc = []
        for i, s in enumerate(texts):
            r = stats(s if pert is None else apply(pert, s, seed=i), emb,
                      ranks, seed=i, long_end=long_end)
            if r:
                acc.append(r)
        if acc:
            rows.append(pd.Series(pd.DataFrame(acc).mean(), name=name))
        print(f"  {name:22s} {len(acc)}", flush=True)
    tbl = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    print(f"\nсостав {int(FRAC * 100)}% самых длинных рёбер MST\n")
    print(tbl.round(1).to_string())
    tbl.round(2).to_csv(os.path.join(BASE, "results", "long_edge_types.csv"))


if __name__ == "__main__":
    main()
