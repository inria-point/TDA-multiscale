"""How big is the cloud, not just who sits on its edges.

Composition of the long end explains a third of the coarse-scale response and
its residuals are systematic: every perturbation that makes the text say less
-- loops, anaphora, field records, uniform syntax -- falls far below what the
shares predict. Shares say who forms the long edges; they cannot say how long
those edges are, and the dimension is a statement about lengths.

Four geometric quantities, all computed on the content tokens only, since the
long end is theirs, and all scaled by the text's own overall spread so they
describe shape rather than size:

  radius      mean distance of a content token from the content centroid
  eff_dim     participation ratio of the covariance eigenvalues,
              (sum l)^2 / sum l^2 -- how many directions the cloud really uses
  long_len    mean length of the longest 20% of MST edges
  spread_gap  ratio of the mean long edge to the mean short edge, the
              separation between the two ends of the distribution
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from embedder import Embedder
from perturb import FUNCTION_WORDS
from pc_space import paired, profiles

BASE = os.path.join(HERE, "..")
ALPHA = re.compile(r"^[A-Za-z]+$")
SKIP = {"[CLS]", "[SEP]", "[PAD]"}


def is_content(t):
    w = t.lstrip("Ġ▁").strip()
    return (ALPHA.match(w) is not None and t.startswith(("Ġ", "▁"))
            and w.lower() not in FUNCTION_WORDS)


def geom(text, emb, seed=0, L=cfg.L_DEFAULT):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [j for j, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v = e[idx]
    t = [toks[j] for j in idx]
    d = np.linalg.norm(v[:, None] - v[None], axis=-1)
    scale = float(d[np.triu_indices(L, 1)].mean())
    if scale == 0:
        return None

    m = minimum_spanning_tree(d).tocoo()
    o = np.argsort(m.data)
    k = max(1, int(0.20 * len(o)))
    long_len = float(m.data[o[-k:]].mean()) / scale
    short_len = float(m.data[o[:k]].mean()) / scale

    cm = np.array([is_content(x) for x in t])
    out = {"long_len": long_len, "spread_gap": long_len / max(short_len, 1e-9)}
    if cm.sum() >= 20:
        c = v[cm]
        c = c - c.mean(0)
        out["radius"] = float(np.linalg.norm(c, axis=1).mean()) / scale
        lam = np.linalg.svd(c, compute_uv=False) ** 2
        out["eff_dim"] = float(lam.sum() ** 2 / (lam ** 2).sum())
    return out


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    originals = {f"coling::{r.id}": r.text for r in pool.itertuples()}
    emb = Embedder()

    texts = {}
    for fn in ["perturbed_texts_coling.json", "perturbed_texts_colingv2.json",
               "perturbed_texts_loops.json", "perturbed_texts_ngram.json",
               "perturbed_texts_expand.json", "perturbed_texts_hapax.json",
               "perturbed_texts_pc2neg.json", "perturbed_texts_composed.json",
               "coling_style_edits.json", "coling_v2_edits.json",
               "coling_hard_edits.json", "coling_pc2_probes.json",
               "coling_collapse_probes.json"]:
        p = os.path.join(BASE, "results", fn)
        if os.path.exists(p):
            with open(p) as f:
                for k, v in json.load(f).items():
                    texts.setdefault(k, v)

    coarse = {}
    for fn in ["perturb_L201_coling_all.csv.gz", "perturb_L201_hapax.csv.gz",
               "perturb_L201_pc2neg.csv.gz", "perturb_L201_pc2probes.csv.gz",
               "perturb_L201_collapse.csv.gz",
               "perturb_L201_composed.csv.gz"]:
        p = os.path.join(BASE, "results", fn)
        if os.path.exists(p):
            d = pd.read_csv(p)
            pr = profiles(paired(d[d["d_hat"] > 0]), "q_small")
            for name in pr.index:
                coarse.setdefault(name, pr.loc[name, 0.8])

    base, rows = {}, []
    for pname, by_id in texts.items():
        if pname not in coarse:
            continue
        keys = [k for k in by_id if k in originals][:20]
        acc = []
        for i, k in enumerate(keys):
            if k not in base:
                base[k] = geom(originals[k], emb, seed=i)
            a, b = base[k], geom(by_id[k], emb, seed=i)
            if a and b:
                acc.append({c: b[c] - a[c] for c in a if c in b})
        if len(acc) >= 8:
            rec = pd.DataFrame(acc).mean().to_dict()
            rec["перт"] = pname
            rec["d крупный"] = coarse[pname]
            rows.append(rec)
        print(f"  {pname:26s} {len(acc)}", flush=True)
    G = pd.DataFrame(rows).set_index("перт")
    S = pd.read_csv(os.path.join(BASE, "results", "pert_edge_shares.csv"),
                    index_col=0)
    J = G.join(S[["швы", "hapax", "смысл."]], rsuffix="_s").dropna()
    J.round(3).to_csv(os.path.join(BASE, "results", "cloud_spread.csv"))

    feats = ["radius", "eff_dim", "long_len", "spread_gap", "швы", "смысл."]
    print(f"\n=== по {len(J)} пертурбациям, связь со сдвигом d крупный\n")
    for f in feats:
        if f in J:
            r, p = spearmanr(J[f], J["d крупный"])
            print(f"  {f:12s} {r:+.2f}  (p={p:.4f})")

    have = [f for f in feats if f in J]
    A = np.c_[np.ones(len(J)), J[have].values]
    coef, *_ = np.linalg.lstsq(A, J["d крупный"].values, rcond=None)
    pred = A @ coef
    r2 = 1 - ((J["d крупный"] - pred) ** 2).sum() / (
        (J["d крупный"] - J["d крупный"].mean()) ** 2).sum()
    print(f"\nсовместная регрессия по всем: R2 = {r2:.2f}")
    for f, c in zip(have, coef[1:]):
        print(f"   {f:12s} {c:+.2f}")
    A2 = np.c_[np.ones(len(J)), J[["швы", "смысл."]].values]
    c2, *_ = np.linalg.lstsq(A2, J["d крупный"].values, rcond=None)
    r22 = 1 - ((J["d крупный"] - A2 @ c2) ** 2).sum() / (
        (J["d крупный"] - J["d крупный"].mean()) ** 2).sum()
    print(f"только по долям (как было): R2 = {r22:.2f}")


if __name__ == "__main__":
    main()
