"""Из чего сделан длинный конец: страты рёбер MST по кратности концов.

Крупная полоса усредняется по q ∈ [0.5, 0.9] с отбрасыванием q самых коротких
рёбер, то есть смотрит только на верхнюю половину распределения длин. Теорема о
ячеечном облаке говорит, что выше точки q* = (n−N)/(n−1) ≈ 0.394 внутриячеечных
рёбер не остаётся совсем, и крупная полоса оказывается функцией одного лишь
каркаса. Старая таксономия (edge_taxonomy) это подтверждает, но делит рёбра по
классам слов, а не по кратности типа, поэтому не отвечает на вопрос, какая
именно часть каркаса сидит в длинном конце.

Здесь каждое ребро дерева относится к одной из четырёх страт по кратностям
типов на его концах:

    внутри ячейки     оба конца — один и тот же тип (k ≥ 2)
    одиночка–одиночка оба типа встретились по разу
    одиночка–ячейка   один разу, другой — повторяющийся
    ячейка–ячейка     разные повторяющиеся типы

Первая страта — ячеечная, остальные три — каркас. Внутри каркаса интересно, в
какой мере длинный конец держится на одиночках: одиночка не имеет собственного
центра, её точка и есть точка каркаса, и именно она чувствительна к тому, как
далеко её унесло от «дома» своего типа.

Стоки исключены (норма < 30). Длины в долях средней парной дистанции.
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
STRATA = ["внутри ячейки", "одиночка–одиночка", "одиночка–ячейка",
          "ячейка–ячейка"]


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def edges(text, key, tok):
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
        cnt = Counter(t)
        w = np.asarray(mst.data) / scale
        rank = w.argsort().argsort() / (len(w) - 1)   # перцентиль длины, 0..1
        for a, b, ww, rr in zip(mst.row, mst.col, w, rank):
            ka, kb = cnt[t[a]], cnt[t[b]]
            if t[a] == t[b]:
                s = 0
            elif ka == 1 and kb == 1:
                s = 1
            elif ka == 1 or kb == 1:
                s = 2
            else:
                s = 3
            out.append((seed, float(ww), float(rr), s))
    return out


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in MECH:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            for r in edges(txt, f"pg_{name}_{i}", tok):
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
            for r in edges(txt, f"pgl_{label}_{k[-8:]}", tok):
                rows.append(("LLM", k, label) + r)
        if (n + 1) % 10 == 0:
            print(f"  LLM: {n + 1}", flush=True)
    D = pd.DataFrame(rows, columns=["набор", "текст", "условие", "зерно",
                                    "длина", "перцентиль", "страта"])
    D["страта"] = D["страта"].map(dict(enumerate(STRATA)))
    D.to_csv(os.path.join(BASE, "results", "edge_strata.csv"), index=False)
    print(f"\n{len(D)} рёбер, {D['условие'].nunique()} условий")

    O = D[D["условие"] == "исходный"]
    print("\nСостав по децилям длины, исходный текст, % рёбер:\n")
    dec = (O.groupby([pd.cut(O["перцентиль"], np.arange(0, 1.01, .1)),
                      "страта"], observed=True).size().unstack("страта")
           .reindex(columns=STRATA).fillna(0))
    print((dec.div(dec.sum(1), axis=0) * 100).round(1).to_string())
    print("\nДоля страты выше порога, исходный текст, %:\n")
    for thr in (0.394, 0.5, 0.697, 0.8, 0.9):
        sub = O[O["перцентиль"] >= thr]["страта"].value_counts(normalize=True)
        print(f"  q ≥ {thr:.3f}: " + "  ".join(
            f"{s} {100 * sub.get(s, 0):.1f}" for s in STRATA))


if __name__ == "__main__":
    main()
