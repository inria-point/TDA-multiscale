"""Почему перемешивание роняет крупную полосу, если каркас не двигается.

Перемешивание сохраняет мешок токенов точно, поэтому N, TTR, β, q* и доля
одиночек заморожены по построению. Модель на каркасных величинах предсказывает
ему подъём (+6.5), наблюдается падение (−10.9); остаток −17.4 — наибольший из 26
условий.

Гипотеза. Каркасное ребро в ПОЛНОМ облаке — это расстояние по ближайшей связи
между двумя ячейками, минимум по парам их членов, а не расстояние между
центрами. Раздувание ячейки центра не смещает, а минимум по парам уменьшает:
предложение 3.2 из report_1.pdf даёт w_ij ≈ ‖C_i − C_j‖ − c·r. Вычитание
постоянной из всех каркасных рёбер обязано ронять показатель, потому что
суммарная длина растёт как N^(1−1/d), а число рёбер как N: вычитается быстрее
растущий член, наклон падает.

Проверяются два звена.

  ЗВЕНО 1 (текст): укорачиваются ли каркасные рёбра полного облака сильнее, чем
      расстояния между центрами. Считается среднее рёбер MST выше q* на полном
      облаке против среднего рёбер MST на облаке центров.

  ЗВЕНО 2 (синтетика): раздуваем ТОЛЬКО ячейки, каркас держим неподвижно, и
      смотрим, воспроизводится ли и укорочение каркасных рёбер, и падение
      полосы. Здесь центры фиксированы принудительно, так что всё, что
      изменится, — следствие одного раздувания.
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
from qphd import qphd
from sink_cluster import bands_of
from synthetic_regimes import DC, KS, N_CELL, N_SING

NORM_CUT, SEEDS, N_TEXTS = 30.0, 3, 50


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def edges_two_ways(v, t):
    """Каркасные рёбра полного облака против рёбер облака центров."""
    where = defaultdict(list)
    for j, x in enumerate(t):
        where[x].append(j)
    cen = np.array([v[ix].mean(0) for ix in where.values()])
    n, N = len(v), len(cen)
    q_star = (n - N) / (n - 1)
    scale = pdist(v).mean()
    w = minimum_spanning_tree(squareform(pdist(v))).tocoo().data / scale
    w = np.sort(w)
    k = int(round(q_star * len(w)))
    frame_full = w[k:]                       # рёбра выше q* в полном облаке
    sc_c = pdist(cen).mean()
    wc = minimum_spanning_tree(squareform(pdist(cen))).tocoo().data / sc_c
    return {"каркас в полном облаке": float(frame_full.mean()),
            "каркас по центрам": float(wc.mean()),
            "масштаб полного": scale, "масштаб центров": sc_c,
            "q*": q_star}


def link_one(text, key, tok):
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
        acc.append(edges_two_ways(e[idx], [toks[j] for j in idx]))
    return {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}


def synth(r_target, rng, cen_fixed=None, blobs_fixed=None):
    """Облако с ФИКСИРОВАННЫМ каркасом и переменным радиусом ячеек."""
    ks = rng.choice(KS, N_CELL)
    need = cfg.L_DEFAULT - N_SING
    while ks.sum() < need:
        ks[rng.integers(N_CELL)] += 1
    while ks.sum() > need:
        j = rng.integers(N_CELL)
        if ks[j] > 2:
            ks[j] -= 1
    cen = cen_fixed if cen_fixed is not None else rng.normal(
        size=(N_SING + N_CELL, DC))
    sing, ccen = cen[:N_SING], cen[N_SING:]
    blobs = blobs_fixed if blobs_fixed is not None else [
        rng.normal(size=(k, DC)) for k in ks]
    X0 = np.vstack([sing] + [c + b for c, b in zip(ccen, blobs)])
    r0 = np.mean([np.linalg.norm(b - b.mean(0), axis=1).mean() for b in blobs])
    f = r_target * pdist(X0).mean() / r0
    X = np.vstack([sing] + [c + b * f for c, b in zip(ccen, blobs)])
    t = [f"s{i}" for i in range(N_SING)]
    for i, k in enumerate(ks):
        t += [f"c{i}"] * k
    return X, t, cen, blobs


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    pool = pd.read_parquet(os.path.join(BASE := os.path.join(HERE, ".."),
                                        "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in [("исходный", "identity"),
                            ("перемешивание", "shuffle_words")]:
            r = link_one(s if name == "identity" else perturb(name, s, seed=i),
                         f"pg_{name}_{i}", tok)
            if r:
                rows.append({"текст": i, "условие": label, **r})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "shuffle_mechanism.csv"),
             index=False)
    print("ЗВЕНО 1. Текст: каркасные рёбра полного облака против центров\n")
    piv = {c: D.pivot(index="текст", columns="условие", values=c)
           for c in ("каркас в полном облаке", "каркас по центрам", "q*")}
    for c in piv:
        a, b = piv[c]["исходный"], piv[c]["перемешивание"]
        m = a.notna() & b.notna()
        print(f"  {c:26s} {a[m].mean():.4f} → {b[m].mean():.4f}  "
              f"({(b[m]-a[m]).div(a[m]).median()*100:+.1f}%)  "
              f"p={wilcoxon(a[m], b[m]).pvalue:.1e}")

    print("\n\nЗВЕНО 2. Синтетика: раздуть ТОЛЬКО ячейки, каркас неподвижен\n")
    out = []
    for rep in range(10):
        rng = np.random.default_rng(rep)
        X0, t0, cen, blobs = synth(0.232, rng)
        rng2 = np.random.default_rng(rep)
        X1, t1, _, _ = synth(0.272, rng2, cen_fixed=cen, blobs_fixed=blobs)
        e0, e1 = edges_two_ways(X0, t0), edges_two_ways(X1, t1)
        b0 = bands_of(qphd(X0, q_list=cfg.Q_GRID, rng=np.random.default_rng(rep),
                           **cfg.qphd_kwargs(L=cfg.L_DEFAULT, pool=X0,
                                             replicates=16)))
        b1 = bands_of(qphd(X1, q_list=cfg.Q_GRID, rng=np.random.default_rng(rep),
                           **cfg.qphd_kwargs(L=cfg.L_DEFAULT, pool=X1,
                                             replicates=16)))
        out.append({"каркас полн. 0": e0["каркас в полном облаке"],
                    "каркас полн. 1": e1["каркас в полном облаке"],
                    "центры 0": e0["каркас по центрам"],
                    "центры 1": e1["каркас по центрам"],
                    "крупная 0": b0["крупный"], "крупная 1": b1["крупный"],
                    "мелкая 0": b0["мелкий"], "мелкая 1": b1["мелкий"]})
    O = pd.DataFrame(out).mean()
    print(f"  радиус ячейки      0.232 → 0.272  (+17.2%, как при перемешивании)")
    print(f"  каркас в полном облаке {O['каркас полн. 0']:.4f} → {O['каркас полн. 1']:.4f}"
          f"  ({(O['каркас полн. 1']/O['каркас полн. 0']-1)*100:+.1f}%)")
    print(f"  каркас по центрам      {O['центры 0']:.4f} → {O['центры 1']:.4f}"
          f"  ({(O['центры 1']/O['центры 0']-1)*100:+.1f}%)")
    print(f"  крупная полоса         {O['крупная 0']:.2f} → {O['крупная 1']:.2f}"
          f"  ({(O['крупная 1']/O['крупная 0']-1)*100:+.1f}%)")
    print(f"  мелкая полоса          {O['мелкая 0']:.2f} → {O['мелкая 1']:.2f}"
          f"  ({(O['мелкая 1']/O['мелкая 0']-1)*100:+.1f}%)")
    print("\n  реальность: перемешивание даёт крупная −10.9%, мелкая +36.5%")


if __name__ == "__main__":
    main()
