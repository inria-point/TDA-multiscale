"""Где именно на оси q живёт влияние каркасных величин?

Теорема из report_1.pdf: каркасная страта — окно [q*, 1], где q* = (n−N)/(n−1),
на наших текстах 0.394. Величины, определённые на каркасе, могут влиять только
на те окна, которые лежат выше q*. Отсюда предсказание:

  q_small     удерживает [q, 1]      → эффект включается при q ≳ 0.39
  q_large     удерживает [0, 1−q]    → эффект выключается при q ≳ 0.61
  q0.5_range  удерживает [q, q+0.5]  → эффект включается при q ≳ 0.39

Проверяется так: для каждой точки (режим, q) сетки считается сдвиг d̂ по 26
условиям и его связь со сдвигами двух каркасных величин — среднего ребра MST
каркаса и коэффициента вариации длин его рёбер. Получается кривая «R² против q»
на каждый режим, и по ней видно, где влияние начинается и где кончается.

Сохраняется полная кривая d̂(q), а не три полосы, — этого в прежних прогонах не
делалось. Стоки исключены.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
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
from qphd import qphd

BASE = os.path.join(HERE, "..")
N_TEXTS, NORM_CUT, SEEDS = 50, 30.0, 2


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def curve(text, key, tok):
    e = cached(text, key)
    t_all = tok.tokenize(text)
    if e is None or len(t_all) != e.shape[0]:
        return None
    keep = [j for j, x in enumerate(t_all) if x not in SKIP]
    e = e[keep]
    e = e[np.linalg.norm(e, axis=1) >= NORM_CUT]
    if e.shape[0] < cfg.L_DEFAULT:
        return None
    acc = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        idx = rng.choice(e.shape[0], size=cfg.L_DEFAULT, replace=False)
        df = qphd(e[idx], q_list=cfg.Q_GRID, rng=rng,
                  **cfg.qphd_kwargs(L=cfg.L_DEFAULT,
                                    pool=cfg.make_pool(e, cfg.L_DEFAULT, rng)))
        acc.append(df[["mode", "q", "d_hat"]])
    return (pd.concat(acc).groupby(["mode", "q"], as_index=False)["d_hat"]
            .mean())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in MECH:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            c = curve(txt, f"pg_{name}_{i}", tok)
            if c is not None:
                c = c.assign(набор="механические", текст=str(i), условие=label)
                rows.append(c)
        if (i + 1) % 10 == 0:
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
            c = curve(txt, f"pgl_{label}_{k[-8:]}", tok)
            if c is not None:
                rows.append(c.assign(набор="LLM", текст=k, условие=label))
        if (n_ + 1) % 10 == 0:
            print(f"  LLM: {n_+1}", flush=True)
    D = pd.concat(rows, ignore_index=True)
    D.to_csv(os.path.join(BASE, "results", "curve_vs_frame.csv"), index=False)
    print(f"\n{len(D)} строк, {D['условие'].nunique()} условий")


if __name__ == "__main__":
    main()
