"""Does the spread of a token's occurrences explain the fine scale?

The mechanism under test: every occurrence of a token sits in the same
neighbourhood of embedding space -- it has to, or the model could not emit it
from that region -- but how tightly that cluster is packed depends on how much
the contexts and the positions of its occurrences differ. Occurrences that are
close together in the text and in the same syntactic and semantic frame give a
tight cluster and therefore a low dimension at the fine scale; occurrences
scattered across positions and frames give a loose one and a spike.

Three quantities are measured per text and averaged over tokens that occur at
least twice, weighted by the number of occurrences:

  bush_radius   mean pairwise distance inside a token's cluster, in units of
                the text's own mean pairwise distance, so it does not simply
                track how spread out the whole text is
  pos_spread    mean gap in tokens between the occurrences, over text length
  rep_share     share of tokens that occur more than once at all

If the mechanism holds, bush_radius should predict the fine-scale response
better than any of the 33 surface features did -- none of them reached 0.4.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import taxonomy as T
from embedder import Embedder
from pc_space import ALL_MODES, full_profile, paired, profiles

BASE = os.path.join(HERE, "..")
SKIP = set("[CLS] [SEP] [PAD]".split())


def bush_stats(text, emb, cap=1200):
    e, toks = emb.embed(text, return_tokens=True)
    e, toks = e[:cap], toks[:cap]
    if len(e) < 50:
        return None
    idx = defaultdict(list)
    for i, t in enumerate(toks):
        if t not in SKIP:
            idx[t].append(i)
    # scale: mean pairwise distance over a sample of the whole text
    rng = np.random.default_rng(0)
    s = e[rng.choice(len(e), size=min(300, len(e)), replace=False)]
    scale = float(np.mean(np.linalg.norm(s[:, None] - s[None], axis=-1)))
    if scale == 0:
        return None

    rad, pos, w = [], [], []
    for t, ii in idx.items():
        if len(ii) < 2:
            continue
        v = e[ii]
        d = np.linalg.norm(v[:, None] - v[None], axis=-1)
        iu = np.triu_indices(len(v), 1)
        rad.append(float(d[iu].mean()) / scale)
        gaps = np.diff(sorted(ii))
        pos.append(float(gaps.mean()) / len(e))
        w.append(len(ii))
    if not rad:
        return None
    w = np.array(w, float)
    return {
        "bush_radius": float(np.average(rad, weights=w)),
        "pos_spread": float(np.average(pos, weights=w)),
        "rep_share": float(w.sum() / len(e)),
    }


def main():
    import json

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
            d = json.load(f)
        for k, v in d.items():
            if k in T.KEEP and k not in texts:
                texts[k] = v

    emb = Embedder()
    n_texts = int(os.environ.get("N_TEXTS", 30))
    rows = []
    for pname, by_id in texts.items():
        keys = [k for k in by_id if k in originals][:n_texts]
        deltas = []
        for k in keys:
            a, b = bush_stats(originals[k], emb), bush_stats(by_id[k], emb)
            if a and b:
                deltas.append({c: b[c] - a[c] for c in a})
        if len(deltas) >= 10:
            rec = pd.DataFrame(deltas).mean().to_dict()
            rec["perturbation"] = T.KEEP[pname]
            rows.append(rec)
        print(f"  {T.KEEP[pname]:30s} {len(deltas):3d}", flush=True)
    B = pd.DataFrame(rows).set_index("perturbation")
    B.round(4).to_csv(os.path.join(BASE, "results", "token_bush.csv"))

    d = pd.read_csv(os.path.join(BASE, "results",
                                 "perturb_L201_coling_all.csv.gz"))
    prof = full_profile({m: profiles(paired(d[d["d_hat"] > 0]), m)
                         for m in ALL_MODES})
    prof.index = [T.KEEP.get(i, i) for i in prof.index]
    P = pd.read_csv(os.path.join(BASE, "results", "pert_map.csv"), index_col=0)
    J = B.join(P[["PC1", "PC2", "PC3"]]).join(
        prof[["q_large@0.8", "q_small@0.8"]].rename(
            columns={"q_large@0.8": "d_fine", "q_small@0.8": "d_coarse"}))
    J = J.dropna()

    from scipy.stats import spearmanr

    pd.set_option("display.width", 200)
    print(f"\nранговые корреляции по {len(J)} пертурбациям\n")
    out = []
    for f in ["bush_radius", "pos_spread", "rep_share"]:
        rec = {"признак": f}
        for tgt in ["d_fine", "d_coarse", "PC1", "PC2"]:
            r, p = spearmanr(J[f], J[tgt])
            rec[tgt] = f"{r:+.2f}" + ("*" if p < 0.01 else "")
        out.append(rec)
    print(pd.DataFrame(out).to_string(index=False))
    print("\n* p < 0.01")
    J.round(3).to_csv(os.path.join(BASE, "results", "token_bush_joined.csv"))


if __name__ == "__main__":
    main()
