"""Одиночки действительно образуют ПАРЫ — или просто сгущаются?

Склейка введена как модель измеренного факта: у неуместных одиночек взаимное
расстояние на 14% меньше, чем у уместных (40 текстов из 40). Но «ближе друг к
другу» не значит «попарно». Это могло быть общим сгущением популяции, группами
по трое, чем угодно.

Различается прямо. Взаимная пара — это две одиночки, каждая из которых является
для другой ближайшим соседом среди ВСЕХ точек облака. Если склейка буквальна,
доля одиночек во взаимных парах должна расти. Если идёт общее сгущение, вырастет
лишь доля «ближайший сосед — тоже одиночка», а взаимность останется на месте.

Три числа на текст:

  сосед-одиночка, %     у скольких одиночек ближайший сосед — другая одиночка
  взаимных пар, %       у скольких это взаимно
  размер сгустка        средний размер связной компоненты в графе «ближайший
                        сосед», построенном только на одиночках: 2 значит пары,
                        больше — цепочки и группы

Стоки исключены. Сравниваются исходный текст, подстановка чужих хапаксов и
перемешивание.
"""
import hashlib
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wilcoxon
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import SKIP
from embedder import CACHE_DIR, MODEL_NAME
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
N_TEXTS, NORM_CUT, SEEDS = 50, 30.0, 3
CONDS = [("исходный", "identity"), ("чужие хапаксы", "hapax_swap_wide"),
         ("хапаксы одного домена", "hapax_swap_local"),
         ("перемешивание", "shuffle_words")]


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def measure(text, key, tok):
    e = cached(text, key)
    t_all = tok.tokenize(text)
    if e is None or len(t_all) != e.shape[0]:
        return None
    keep = [j for j, x in enumerate(t_all) if x not in SKIP]
    e, toks = e[keep], [t_all[j] for j in keep]
    ok = np.linalg.norm(e, axis=1) >= NORM_CUT
    e, toks = e[ok], [x for x, k in zip(toks, ok) if k]
    if e.shape[0] < cfg.L_DEFAULT:
        return None
    acc = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        idx = rng.choice(e.shape[0], size=cfg.L_DEFAULT, replace=False)
        v, t = e[idx], [toks[j] for j in idx]
        cnt = Counter(t)
        sg = np.array([cnt[x] == 1 for x in t])
        if sg.sum() < 20:
            continue
        D = squareform(pdist(v))
        np.fill_diagonal(D, np.inf)
        nn = D.argmin(1)
        s_idx = np.where(sg)[0]
        # ближайший сосед одиночки — тоже одиночка
        nn_is_single = sg[nn[s_idx]]
        # взаимность: nn(nn(i)) == i
        mutual = nn[nn[s_idx]] == s_idx
        # граф ближайших соседей внутри одиночек: размер компоненты
        pos = {j: k for k, j in enumerate(s_idx)}
        rr, cc = [], []
        for j in s_idx:
            if sg[nn[j]]:
                rr.append(pos[j])
                cc.append(pos[nn[j]])
        if rr:
            gph = coo_matrix((np.ones(len(rr)), (rr, cc)),
                             shape=(len(s_idx), len(s_idx)))
            ncomp, lab = connected_components(gph, directed=False)
            sizes = np.bincount(lab)
            linked = sizes[sizes >= 2]
            comp = float(linked.mean()) if len(linked) else np.nan
        else:
            comp = np.nan
        acc.append({"сосед-одиночка, %": float(nn_is_single.mean() * 100),
                    "взаимных пар, %": float((mutual & nn_is_single).mean() * 100),
                    "размер сгустка": comp,
                    "до ближайшей одиночки":
                        float(np.mean([D[j, nn[j]] for j in s_idx
                                       if sg[nn[j]]]) / pdist(v).mean())
                        if nn_is_single.any() else np.nan})
    if not acc:
        return None
    return {k: float(np.nanmean([a[k] for a in acc])) for k in acc[0]}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in CONDS:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            r = measure(txt, f"pg_{name}_{i}", tok)
            if r:
                rows.append({"текст": i, "условие": label, **r})
        if (i + 1) % 15 == 0:
            print(f"  {i+1} текстов", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "pairing_test.csv"), index=False)
    cols = [c for c in D.columns if c not in ("текст", "условие")]
    print("\nСредние значения:\n")
    print(D.groupby("условие")[cols].mean().round(3).to_string())
    print("\nПарно против исходного:\n")
    piv = {c: D.pivot(index="текст", columns="условие", values=c) for c in cols}
    for cond in ("чужие хапаксы", "хапаксы одного домена", "перемешивание"):
        print(f"  {cond}")
        for c in cols:
            a, b = piv[c]["исходный"], piv[c][cond]
            m = a.notna() & b.notna()
            print(f"    {c:24s} {a[m].mean():7.3f} → {b[m].mean():7.3f}  "
                  f"({(b[m] - a[m]).median():+.3f})  p={wilcoxon(a[m], b[m]).pvalue:.1e}")


if __name__ == "__main__":
    main()
