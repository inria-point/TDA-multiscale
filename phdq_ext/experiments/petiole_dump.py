"""Все черешки одиночек поштучно: длина и составляющая вдоль оси.

petiole_variants.py считал сводки при фиксированных порогах, и каждый новый
порог требовал пересчёта. Здесь выгружаются сами рёбра, так что любой порог —
в том числе выбранный из данных, например «столько, чтобы 90% черешков
перемешанного текста проходило» — считается потом мгновенно.

Черешок вершины — самое длинное ребро, которым она держится за дерево; берутся
только одиночки (токен встретился в выборке один раз). Стоки исключены из
облака. Длины в долях средней парной дистанции. Ось — fit_pmi_axis.npz.
"""
import hashlib
import json
import os
import sys
from collections import Counter

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


def petioles(text, key, tok, u):
    e = cached(text, key)
    t_all = tok.tokenize(text)
    if e is None or len(t_all) != e.shape[0]:
        return []
    keep = [j for j, x in enumerate(t_all) if x not in SKIP]
    e, toks = e[keep], [t_all[j] for j in keep]
    ok = np.linalg.norm(e, axis=1) >= NORM_CUT
    e, toks = e[ok], [x for x, k in zip(toks, ok) if k]
    if e.shape[0] < cfg.L_DEFAULT:
        return []
    out = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        idx = rng.choice(e.shape[0], size=cfg.L_DEFAULT, replace=False)
        v, t = e[idx], [toks[j] for j in idx]
        scale = pdist(v).mean()
        mst = minimum_spanning_tree(squareform(pdist(v))).tocoo()
        best = {}
        for a, b, w in zip(mst.row, mst.col, mst.data):
            for x, y in ((a, b), (b, a)):
                if x not in best or w > best[x][0]:
                    best[x] = (w, y)
        cnt = Counter(t)
        # центры ячеек: типы с двумя и более вхождениями. Каркас — это они,
        # и держаться за каркас значит держаться за повторяющийся токен;
        # держаться за другую одиночку — не то же самое
        where = {}
        for j2, x in enumerate(t):
            where.setdefault(x, []).append(j2)
        cen = np.array([v[ix].mean(0) for x, ix in where.items()
                        if len(ix) >= 2])
        for j, (w, k) in best.items():
            if cnt[t[j]] != 1:
                continue
            d = v[j] - v[k]
            d_frame = (float(np.linalg.norm(cen - v[j], axis=1).min()) / scale
                       if len(cen) else np.nan)
            out.append((seed, w / scale, abs(float(d @ u)) / scale,
                        int(cnt[t[k]] == 1), d_frame))
    return out


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    u = np.load(os.path.join(BASE, "results",
                             "fit_pmi_axis.npz"))["перемешивание"]
    u = u / np.linalg.norm(u)

    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in MECH:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            for r in petioles(txt, f"pg_{name}_{i}", tok, u):
                rows.append(("механические", str(i), label) + r)
        if (i + 1) % 10 == 0:
            print(f"  механические: {i + 1}", flush=True)
    src = dict(human_texts(120, min_words=280))
    store = {}
    for label, key, fname in LLM:
        with open(os.path.join(BASE, "results", fname)) as f:
            store[label] = json.load(f)[key]
    for n, (k, text) in enumerate(src.items()):
        if n >= N_TEXTS:
            break
        for label, txt in [("исходный", text)] + [
                (lab, store[lab].get(k)) for lab, _, _ in LLM]:
            if txt is None:
                continue
            for r in petioles(txt, f"pgl_{label}_{k[-8:]}", tok, u):
                rows.append(("LLM", k, label) + r)
    D = pd.DataFrame(rows, columns=["набор", "текст", "условие", "зерно",
                                    "черешок", "вдоль оси",
                                    "сосед-одиночка", "до каркаса"])
    D.to_csv(os.path.join(BASE, "results", "petiole_dump.csv"), index=False)
    print(f"\n{len(D)} черешков, {D['условие'].nunique()} условий")
    q = D[D["условие"] == "перемешивание"]["вдоль оси"]
    for p in (50, 75, 90, 95, 99):
        print(f"  перемешивание, {p}-й перцентиль |вдоль оси|: "
              f"{np.percentile(q, p):.4f}")


if __name__ == "__main__":
    main()
