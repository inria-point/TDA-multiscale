"""Переписать обе рабочие величины в терминах графа каркаса.

В словаре report_1.pdf каркас — это облако центров ВСЕХ ячеек, включая ячейки
кратности 1 (одиночка сама себе центр). Наши две рабочие величины в этот словарь
переводятся плохо:

  «расстояние до каркаса» — на самом деле расстояние от точки каркаса с k = 1 до
      ближайшей точки с k ≥ 2. Это чистая каркасная величина (центр ячейки не
      смещается при её раздувании), но ей нужна разметка точек по кратности.

  «разброс черешков» — вообще не каркасная: черешок считается на полном облаке
      из 201 точки, и дальний конец может быть отдельным членом ячейки, а не её
      центром. Поэтому он смешивает каркас с толщиной ячеек.

Здесь считается батарея величин, определённых ТОЛЬКО на графе каркаса, и
проверяется, есть ли среди них не хуже. Граф каркаса — MST на центрах типов;
одиночечный подграф — граф ближайших соседей, ограниченный на точки с k = 1.

Стоки исключены везде.
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
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


def frame_stats(cen, single):
    """Всё считается на графе каркаса; длины в долях средней парной каркаса."""
    n = len(cen)
    D = squareform(pdist(cen))
    scale = pdist(cen).mean()
    mst = minimum_spanning_tree(D).tocoo()
    w = mst.data / scale
    a, b = mst.row, mst.col
    out = {"каркас: ребро ср": float(w.mean()),
           "каркас: ребро CV": float(w.std() / w.mean())}

    inc_s = single[a] | single[b]                 # ребро касается одиночки
    both_s = single[a] & single[b]                # оба конца одиночки
    out["каркас: ребро у одиночек ср"] = float(w[inc_s].mean())
    out["каркас: ребро у одиночек CV"] = float(w[inc_s].std() / w[inc_s].mean())
    out["каркас: доля рёбер одиночка-одиночка, %"] = float(both_s.mean() * 100)

    # черешок в каркасе: самое длинное ребро вершины, только по одиночкам
    longest = np.zeros(n)
    for x, y, ww in zip(a, b, w):
        longest[x] = max(longest[x], ww)
        longest[y] = max(longest[y], ww)
    ls = longest[single]
    out["каркас: черешок ср"] = float(ls.mean())
    out["каркас: черешок CV"] = float(ls.std() / ls.mean())

    # кросс-типовые расстояния и обменность
    Dn = D.copy()
    np.fill_diagonal(Dn, np.inf)
    rep = ~single
    if rep.sum() >= 5 and single.sum() >= 5:
        d_s2r = Dn[np.ix_(single, rep)].min(1).mean() / scale
        d_r2r = Dn[np.ix_(rep, rep)].min(1).mean() / scale
        out["до ближайшего повторного"] = float(d_s2r)
        out["R = обменность"] = float(d_s2r / d_r2r)

    # одиночечный подграф: ближайший сосед среди одиночек
    idx = np.where(single)[0]
    if len(idx) >= 10:
        Ds = Dn[np.ix_(idx, idx)]
        nn = Ds.argmin(1)
        rr = np.arange(len(idx))
        gph = coo_matrix((np.ones(len(idx)), (rr, nn)),
                         shape=(len(idx), len(idx)))
        _, lab = connected_components(gph, directed=False)
        sizes = np.bincount(lab)
        out["одиночки: размер цепи"] = float(sizes[sizes >= 2].mean()
                                             if (sizes >= 2).any() else np.nan)
        out["одиночки: до ближайшей"] = float(Ds.min(1).mean() / scale)
        out["одиночки: разброс до ближайшей"] = float(Ds.min(1).std() / scale)
    return out


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
        where = defaultdict(list)
        for j, x in enumerate(t):
            where[x].append(j)
        cen = np.array([v[ix].mean(0) for ix in where.values()])
        single = np.array([len(ix) == 1 for ix in where.values()])
        if len(cen) < 40:
            continue
        acc.append(frame_stats(cen, single))
    if not acc:
        return None
    keys = set().union(*[a.keys() for a in acc])
    return {k: float(np.nanmean([a.get(k, np.nan) for a in acc])) for k in keys}


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
        if (i + 1) % 15 == 0:
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
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "frame_graph_stats.csv"),
             index=False)
    print(f"\nНеиспорченный текст:\n")
    print(D[D["условие"] == "исходный"].mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
