"""Что именно сжимается и что растёт при перемешивании, в абсолютных единицах.

Размерность считается по расстояниям, поэтому всё, что на неё влияет, обязано
выражаться через расстояния. «Вектора уносит от домов на 44%» — геометрическое
утверждение; вопрос в его структуре.

Известно: смещение от дома +44%, согласие направлений внутри типа 0.348 → 0.618,
а средняя парная дистанция облака при этом ПАДАЕТ на 5.4%. Это возможно только
если общая часть сноса велика (перенос, геометрически инертен), а индивидуальная
сжимается.

Отсюда гипотеза: отношение r/δ растёт с двух сторон сразу — ячейки раздуваются,
каркас сжимается. В долях масштаба видно только первое (радиус +17.4%), потому
что масштабом и служит сжимающееся облако.

Здесь всё меряется в АБСОЛЮТНЫХ единицах эмбеддинга:

  радиус ячейки         среднее расстояние вхождений типа до их центра
  размер каркаса        средняя парная дистанция облака центров
  каркасное ребро       среднее ребро MST облака центров
  r/δ                   отношение радиуса к каркасному ребру

Если каркас сжимается, а ячейки растут, механизм перемешивания объясняется
одним отношением, и «половина, остающаяся текстовой» — это вторая половина того
же отношения, которую прежние замеры не видели из-за нормировки.
"""
import hashlib
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
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
NORM_CUT, SEEDS, N_TEXTS = 30.0, 3, 50
CONDS = [("исходный", "identity"), ("перемешивание", "shuffle_words"),
         ("эхо x4", "loop_local_3"), ("чужие хапаксы", "hapax_swap_wide"),
         ("без пунктуации", "strip_punctuation")]


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def absolute_geometry(text, key, tok):
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
        rad = [np.linalg.norm(v[ix] - v[ix].mean(0), axis=1).mean()
               for ix in where.values() if len(ix) >= 2]
        if len(rad) < 4 or len(cen) < 20:
            continue
        r = float(np.mean(rad))
        Dc = squareform(pdist(cen))
        wc = minimum_spanning_tree(Dc).tocoo().data
        np.fill_diagonal(Dc, np.inf)
        acc.append({"радиус, абс": r,
                    "каркас: парная, абс": float(pdist(cen).mean()),
                    "каркас: ребро MST, абс": float(wc.mean()),
                    "каркас: до ближайшего, абс": float(Dc.min(1).mean()),
                    "облако: парная, абс": float(pdist(v).mean()),
                    "r / ребро каркаса": r / float(wc.mean()),
                    "норма вектора": float(np.linalg.norm(v, axis=1).mean())})
    if not acc:
        return None
    return {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in CONDS:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            r = absolute_geometry(txt, f"pg_{name}_{i}", tok)
            if r:
                rows.append({"текст": i, "условие": label, **r})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "shuffle_absolute.csv"), index=False)
    cols = [c for c in D.columns if c not in ("текст", "условие")]
    print("Абсолютные величины (единицы эмбеддинга), среднее по 50 текстам:\n")
    print(D.groupby("условие")[cols].mean().round(3).T.to_string())
    print("\n\nСдвиг против исходного, %:\n")
    piv = {c: D.pivot(index="текст", columns="условие", values=c) for c in cols}
    hdr = f"{'величина':28s}" + "".join(
        f"{c:>18s}" for c in ("перемешивание", "эхо x4", "чужие хапаксы",
                              "без пунктуации"))
    print(hdr)
    for c in cols:
        line = f"{c:28s}"
        for cond in ("перемешивание", "эхо x4", "чужие хапаксы",
                     "без пунктуации"):
            a, b = piv[c]["исходный"], piv[c][cond]
            m = a.notna() & b.notna()
            d = ((b[m] - a[m]) / a[m]).median() * 100
            p = wilcoxon(a[m], b[m]).pvalue
            line += f"{d:+12.1f}{'*' if p < 0.01 else ' ':5s}"
        print(line)
    print("\n  * = p < 0.01")


if __name__ == "__main__":
    main()
