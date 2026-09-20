"""Каркасная страта ПОЛНОГО облака на всех условиях.

Модель крупной полосы строилась на величинах облака центроидов, а оценка
размерности видит полное облако. Для перемешивания это разные знаки: расстояния
между центроидами растут (+1.3%), а каркасные рёбра полного облака укорачиваются
(−0.8%), потому что раздувшаяся ячейка подставляет соседке своего крайнего члена
(предложение 3.2: w_ij ≈ ‖C_i − C_j‖ − c·r).

Здесь то же самое считается на всех 26 условиях: рёбра MST полного облака выше
q*, их среднее и коэффициент вариации. Стоки исключены.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from coling_data import human_texts
from edge_taxonomy import SKIP
from embedder import CACHE_DIR, MODEL_NAME
from pert_geometry import PERTS as MECH
from pert_geometry_llm import PERTS as LLM
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
N_TEXTS, NORM_CUT, SEEDS = 50, 30.0, 3


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
        n, N = len(v), len(set(t))
        q = (n - N) / (n - 1)
        scale = pdist(v).mean()
        w = np.sort(minimum_spanning_tree(squareform(pdist(v))).tocoo().data
                    / scale)
        k = int(round(q * len(w)))
        fr, cell = w[k:], w[:k]
        acc.append({"полн: каркас ребро": float(fr.mean()),
                    "полн: каркас CV": float(fr.std() / fr.mean()),
                    "полн: ячейки ребро": float(cell.mean()) if k else np.nan,
                    "q*": q})
    if not acc:
        return None
    return {k: float(np.nanmean([a[k] for a in acc])) for k in acc[0]}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in MECH:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            r = measure(txt, f"pg_{name}_{i}", tok)
            if r:
                rows.append({"набор": "механические", "текст": str(i),
                             "условие": label, **r})
        if (i + 1) % 20 == 0:
            print(f"  механические: {i+1}", flush=True)
    src = dict(human_texts(120, min_words=280))
    store = {}
    for label, key, fname in LLM:
        with open(os.path.join(BASE, "results", fname)) as f:
            store[label] = json.load(f)[key]
    for n_, (k, text) in enumerate(src.items()):
        if n_ >= N_TEXTS:
            break
        for label, txt in [("исходный", text)] + [
                (lab, store[lab].get(k)) for lab, _, _ in LLM]:
            if txt is None:
                continue
            r = measure(txt, f"pgl_{label}_{k[-8:]}", tok)
            if r:
                rows.append({"набор": "LLM", "текст": k, "условие": label, **r})
    pd.DataFrame(rows).to_csv(
        os.path.join(BASE, "results", "frame_stratum_full.csv"), index=False)
    print("\nготово")


if __name__ == "__main__":
    main()
