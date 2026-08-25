"""Do syntactic seams raise the coarse dimension and hapax edges lower it?

The hypothesis to test: the long end of the MST is shared between two kinds of
edge -- seams between the punctuation and function-word clusters, and hapax
words hanging off the tree as leaves -- and the coarse-scale dimension follows
which of the two occupies it.

Two tests, deliberately independent.

Across perturbations: each perturbation shifts the composition of the long end
and shifts the coarse dimension, both measured against the same source texts,
so the comparison is paired and the direction of the change is what is
correlated.

Across human texts, untouched: the same two shares and the absolute coarse
dimension, text by text, with nothing manipulated at all. If the relation is a
property of the geometry it should appear here too; if it appears only under
perturbation it is a property of our operations.

Reported as partial correlations as well, because the two shares are
mechanically linked -- an edge counted as a seam is not counted as a hapax
edge -- so each has to be tested with the other held fixed.
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
import taxonomy as T
from embedder import Embedder
from perturb import FUNCTION_WORDS
from pc_space import ALL_MODES, paired, profiles
from qphd import qphd

BASE = os.path.join(HERE, "..")
ALPHA = re.compile(r"^[A-Za-z]+$")
SKIP = {"[CLS]", "[SEP]", "[PAD]"}
SVC = {"пункт.", "служ.слово"}


def kind(t):
    w = t.lstrip("Ġ▁").strip()
    if not ALPHA.match(w):
        return "пункт."
    if not t.startswith(("Ġ", "▁")):
        return "часть"
    return "служ.слово" if w.lower() in FUNCTION_WORDS else "смысл."


def shares(text, emb, seed=0, L=cfg.L_DEFAULT, with_d=False):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [j for j, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    t = [toks[j] for j in idx]
    full = Counter(toks[j] for j in keep)
    hap = np.array([full[x] == 1 for x in t])
    k_ = [kind(x) for x in t]
    d = np.linalg.norm(e[idx][:, None] - e[idx][None], axis=-1)
    m = minimum_spanning_tree(d).tocoo()
    o = np.argsort(m.data)
    k = max(1, int(0.20 * len(o)))
    a, b = m.row[o[-k:]], m.col[o[-k:]]
    out = {
        "швы": float(np.mean([k_[i] in SVC and k_[j] in SVC
                              for i, j in zip(a, b)])) * 100,
        "hapax": float(np.mean(hap[a] | hap[b])) * 100,
        "смысл.": float(np.mean([k_[i] == "смысл." and k_[j] == "смысл."
                                 for i, j in zip(a, b)])) * 100,
    }
    if with_d and e.shape[0] >= int(1.3 * L):
        r2 = np.random.default_rng(seed)
        df = qphd(e[idx], q_list=(0.0, 0.8), rng=r2,
                  **cfg.qphd_kwargs(L=L, pool=cfg.make_pool(e, L, r2),
                                    replicates=16))
        g = df.set_index(["mode", "q"])["d_hat"]
        out["d крупный"] = g[("q_small", 0.8)]
        out["d общий"] = g[("q_small", 0.0)]
    return out


def partial(x, y, z):
    rx, ry, rz = (pd.Series(v).rank().values for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return spearmanr(ex, ey)


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    originals = {f"coling::{r.id}": r.text for r in pool.itertuples()}
    emb = Embedder()

    # --- test 2 first: human texts, nothing manipulated
    hum = [r.text for r in pool.itertuples() if r.is_human][:150]
    rows = [shares(s, emb, seed=i, with_d=True) for i, s in enumerate(hum)]
    H = pd.DataFrame([r for r in rows if r])
    print(f"=== по {len(H)} нетронутым человеческим текстам")
    for f in ["швы", "hapax", "смысл."]:
        r, p = spearmanr(H[f], H["d крупный"])
        rp, pp = partial(H[f], H["d крупный"], H["d общий"])
        print(f"  {f:8s} с d крупный: {r:+.2f} (p={p:.3f});  "
              f"с исключённым общим уровнем: {rp:+.2f} (p={pp:.3f})")
    H.round(3).to_csv(os.path.join(BASE, "results", "human_edge_shares.csv"),
                      index=False)

    # --- test 1: across perturbations
    texts = {}
    for fn in ["perturbed_texts_coling.json", "perturbed_texts_colingv2.json",
               "perturbed_texts_loops.json", "perturbed_texts_ngram.json",
               "perturbed_texts_expand.json", "perturbed_texts_hapax.json",
               "perturbed_texts_pc2neg.json", "perturbed_texts_composed.json",
               "coling_style_edits.json", "coling_v2_edits.json",
               "coling_hard_edits.json", "coling_pc2_probes.json",
               "coling_collapse_probes.json"]:
        path = os.path.join(BASE, "results", fn)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for k, v in json.load(f).items():
                texts.setdefault(k, v)

    coarse = {}
    for fn in ["perturb_L201_coling_all.csv.gz", "perturb_L201_hapax.csv.gz",
               "perturb_L201_pc2neg.csv.gz", "perturb_L201_pc2probes.csv.gz",
               "perturb_L201_collapse.csv.gz",
               "perturb_L201_composed.csv.gz"]:
        path = os.path.join(BASE, "results", fn)
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
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
                base[k] = shares(originals[k], emb, seed=i)
            a, b = base[k], shares(by_id[k], emb, seed=i)
            if a and b:
                acc.append({c: b[c] - a[c] for c in a})
        if len(acc) >= 8:
            rec = pd.DataFrame(acc).mean().to_dict()
            rec["перт"] = pname
            rec["d крупный"] = coarse[pname]
            rows.append(rec)
        print(f"  {pname:26s} {len(acc)}", flush=True)
    P = pd.DataFrame(rows).set_index("перт")
    P.round(2).to_csv(os.path.join(BASE, "results", "pert_edge_shares.csv"))

    print(f"\n=== по {len(P)} пертурбациям (сдвиги, парно)")
    for f in ["швы", "hapax", "смысл."]:
        r, p = spearmanr(P[f], P["d крупный"])
        print(f"  Δ{f:8s} с Δd крупный: {r:+.2f} (p={p:.4f})")
    for f, other in [("швы", "hapax"), ("hapax", "швы")]:
        rp, pp = partial(P[f], P["d крупный"], P[other])
        print(f"  Δ{f:8s} при фиксированном Δ{other}: {rp:+.2f} (p={pp:.4f})")


if __name__ == "__main__":
    main()
