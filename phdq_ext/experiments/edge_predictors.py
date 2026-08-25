"""Which edge statistic predicts the dimension response, over all perturbations?

Four candidates, each computed on the perturbed text minus the same statistic
on its source, then correlated across perturbations with the dimension change:

  bush_radius     spread of one token's occurrences in embedding space, in
                  units of the text's own mean pairwise distance
  gap_tokens      mean distance in the text, in tokens, between the ends of a
                  short MST edge
  near_share      share of short edges whose ends stand under 10 tokens apart
  collapsed       share of all MST edges under a third of the mean length

The question they arbitrate: is a low dimension driven by how tightly the
occurrences of one token cluster, or by whether those occurrences also stand
next to each other in the text. The two come apart -- collapsing the
vocabulary onto ten words packs the clusters and scatters them across the
document, an echo packs them and puts them side by side.

Targets are the paired relative change of d at the fine scale (q_large 0.8,
the shortest edges kept), at the coarse scale (q_small 0.8) and untrimmed,
and the position on PC2. PC2 is kept separate on purpose: it is not "the
dimension is low" but "the two scales moved against each other", so a
statistic can predict the level well and PC2 not at all.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
import taxonomy as T
from embedder import Embedder
from pc_space import ALL_MODES, full_profile, paired, profiles

BASE = os.path.join(HERE, "..")
SKIP = {"[CLS]", "[SEP]", "[PAD]"}
NEAR = 10


def stats(text, emb, L=cfg.L_DEFAULT, seed=0):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v, t = e[idx], [toks[i] for i in idx]
    d = np.linalg.norm(v[:, None] - v[None], axis=-1)

    # spread inside each token's cluster, scaled by the text's own spread
    scale = float(d[np.triu_indices(L, 1)].mean())
    pos = {}
    for i, tok in enumerate(t):
        pos.setdefault(tok, []).append(i)
    rad, w = [], []
    for ii in pos.values():
        if len(ii) < 2:
            continue
        sub = d[np.ix_(ii, ii)]
        rad.append(sub[np.triu_indices(len(ii), 1)].mean() / scale)
        w.append(len(ii))
    bush = float(np.average(rad, weights=w)) if rad else np.nan

    m = minimum_spanning_tree(d).tocoo()
    o = np.argsort(m.data)
    lens, rows, cols = m.data[o], m.row[o], m.col[o]
    k = max(1, int(0.20 * len(lens)))
    gaps = np.abs(idx[rows[:k]] - idx[cols[:k]])
    return {
        "bush_radius": bush,
        "gap_tokens": float(gaps.mean()),
        "near_share": float((gaps < NEAR).mean()) * 100,
        "collapsed": float((lens < 0.33 * lens.mean()).mean()) * 100,
    }


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    originals = {f"coling::{r.id}": r.text for r in pool.itertuples()}
    texts = {}
    for fn in ["perturbed_texts_coling.json", "perturbed_texts_colingv2.json",
               "perturbed_texts_loops.json", "perturbed_texts_ngram.json",
               "perturbed_texts_expand.json", "perturbed_texts_hard.json",
               "coling_style_edits.json", "coling_v2_edits.json",
               "coling_hard_edits.json"]:
        path = os.path.join(BASE, "results", fn)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for k, v in json.load(f).items():
                if k in T.KEEP and k not in texts:
                    texts[k] = v

    emb = Embedder()
    n_texts = int(os.environ.get("N_TEXTS", 25))
    base = {}
    rows = []
    for pname, by_id in texts.items():
        keys = [k for k in by_id if k in originals][:n_texts]
        deltas = []
        for i, k in enumerate(keys):
            if k not in base:
                base[k] = stats(originals[k], emb, seed=i)
            a, b = base[k], stats(by_id[k], emb, seed=i)
            if a and b:
                deltas.append({c: b[c] - a[c] for c in a})
        if len(deltas) >= 8:
            rec = pd.DataFrame(deltas).mean().to_dict()
            rec["perturbation"] = T.KEEP[pname]
            rows.append(rec)
        print(f"  {T.KEEP[pname]:30s} {len(deltas):3d}", flush=True)
    E = pd.DataFrame(rows).set_index("perturbation")
    E.round(3).to_csv(os.path.join(BASE, "results", "edge_predictors.csv"))

    d = pd.read_csv(os.path.join(BASE, "results",
                                 "perturb_L201_coling_all.csv.gz"))
    prof = full_profile({m: profiles(paired(d[d["d_hat"] > 0]), m)
                         for m in ALL_MODES})
    prof.index = [T.KEEP.get(i, i) for i in prof.index]
    P = pd.read_csv(os.path.join(BASE, "results", "pert_map.csv"),
                    index_col=0)
    J = E.join(prof[["q_large@0.8", "q_small@0.8", "q_small@0"]].rename(
        columns={"q_large@0.8": "d_мелкий", "q_small@0.8": "d_крупный",
                 "q_small@0": "d_общий"})).join(P[["PC1", "PC2"]]).dropna()
    J["мелк.-круп."] = J["d_мелкий"] - J["d_крупный"]

    pd.set_option("display.width", 200)
    print(f"\nранговые корреляции по {len(J)} пертурбациям\n")
    out = []
    for f in ["bush_radius", "gap_tokens", "near_share", "collapsed"]:
        rec = {"признак": f}
        for tgt in ["d_общий", "d_мелкий", "d_крупный",
                    "мелк.-круп.", "PC1", "PC2"]:
            r, p = spearmanr(J[f], J[tgt])
            rec[tgt] = f"{r:+.2f}" + ("*" if p < 0.01 else "")
        out.append(rec)
    print(pd.DataFrame(out).to_string(index=False))
    print("\n* p < 0.01")
    J.round(3).to_csv(os.path.join(BASE, "results",
                                   "edge_predictors_joined.csv"))


if __name__ == "__main__":
    main()
