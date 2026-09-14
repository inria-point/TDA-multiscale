"""Where does an inappropriate word go, if not further from the text?

fit_pmi.py left one thing not adding up. A word the context does not call for
sits *further* from its lexical home (смещение +13% under substitution, +44%
under shuffling, and PMI correlates with смещение at rho = -0.2..-0.3 inside
every text) and yet *closer* to the rest of the text -- its petiole shortens by
3.4% and the share of long leaves falls 15%. Distance to the centre of the
cloud does not move at all (0.706 in all three conditions), so it is not simply
falling into the middle.

The remaining reading: inappropriate tokens are pushed in a *common* direction.
Then they end up near each other rather than near the centre, which shortens
exactly their petioles and leaves the rest of the long end where it was -- which
is the mixture that raises CV and drops the coarse band.

Measured here, per text and condition, on the displacement vectors
d = v - дом(тип):

  общее направление   mean pairwise cosine between displacements of *different*
                      token types -- do they share a direction at all
  по терцилям PMI     the same inside the lowest and the highest third by PMI
  сближение           mean pairwise distance among the lowest-PMI singletons
                      against the highest-PMI ones, in units of the text scale

No encoder pass is needed: the embeddings are in the cache and the PMI values
are in results/fit_pmi.csv, whose rows are in sample order, so the two join by
position. The token strings are checked against each other row by row.
"""
import hashlib
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import SKIP
from embedder import CACHE_DIR, MODEL_NAME
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
HOME_TEXTS, HOME_MIN = 200, 5
CONDS = [("исходный", None), ("чужие хапаксы", "hapax_swap_wide"),
         ("перемешивание", "shuffle_words")]


def cached(text, key):
    digest = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, f"{key}_{digest[:10]}.npz")
    return np.load(path)["embeds"] if os.path.exists(path) else None


def home_vectors(texts, tok):
    tot, cnt = {}, Counter()
    for i, s in enumerate(texts):
        e = cached(s, f"home_{i}")
        t = tok.tokenize(s)
        if e is None or len(t) != e.shape[0]:
            continue
        for x, vv in zip(t, e):
            tot[x] = tot[x] + vv if x in tot else vv.copy()
            cnt[x] += 1
    return {x: tot[x] / cnt[x] for x in tot if cnt[x] >= HOME_MIN}


def pair_cos(M, types):
    """Mean cosine between displacements belonging to different types."""
    if len(M) < 3:
        return np.nan
    U = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    C = U @ U.T
    same = np.array(types)[:, None] == np.array(types)[None, :]
    C[same] = np.nan
    return float(np.nanmean(C))


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    home = home_vectors(hum[:HOME_TEXTS], tok)
    print(f"домашних векторов: {len(home)}", flush=True)

    P = pd.read_csv(os.path.join(BASE, "results", "fit_pmi.csv"))
    rows, checked, bad = [], 0, 0
    for i, s in enumerate(hum[:P["текст"].max() + 1]):
        for label, name in CONDS:
            txt = s if name is None else perturb(name, s, seed=i)
            e = cached(txt, f"pg_{name or 'identity'}_{i}")
            if e is None:
                continue
            toks = tok.tokenize(txt)
            keep = [j for j, x in enumerate(toks) if x not in SKIP]
            if len(toks) != e.shape[0] or len(keep) < cfg.L_DEFAULT:
                continue
            e, toks = e[keep], [toks[j] for j in keep]
            rng = np.random.default_rng(1000)
            idx = rng.choice(len(toks), size=cfg.L_DEFAULT, replace=False)
            v, t = e[idx], [toks[j] for j in idx]

            g = P[(P["текст"] == i) & (P["условие"] == label)]
            if len(g) != len(t):
                continue
            if list(g["токен"]) != t:                 # positions must line up
                bad += 1
                continue
            checked += 1
            pmi = g["PMI"].to_numpy()
            cnt = Counter(t)

            have = [j for j, x in enumerate(t) if x in home]
            if len(have) < 40:
                continue
            Dm = np.array([v[j] - home[t[j]] for j in have])
            types = [t[j] for j in have]
            pm = pmi[have]
            scale = pdist(v).mean()
            lo, hi = np.percentile(pm, [33, 67])
            r = {"текст": i, "условие": label, "n": len(have),
                 "общее направление": pair_cos(Dm, types),
                 "низкий PMI: направление": pair_cos(Dm[pm <= lo],
                                                     list(np.array(types)[pm <= lo])),
                 "высокий PMI: направление": pair_cos(Dm[pm >= hi],
                                                      list(np.array(types)[pm >= hi]))}
            # do the low-PMI singletons end up nearer one another?
            sing = np.array([cnt[t[j]] == 1 for j in have])
            for tag, m in [("низкий PMI", sing & (pm <= lo)),
                           ("высокий PMI", sing & (pm >= hi))]:
                pts = v[np.array(have)[m]]
                r[f"{tag}: взаимное расст."] = (pdist(pts).mean() / scale
                                                if len(pts) >= 3 else np.nan)
            rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "fit_pmi_direction.csv"), index=False)
    print(f"сверено по токенам: {checked} пар, несовпало {bad}\n")
    print(D.groupby("условие").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
