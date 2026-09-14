"""Решающая проверка: мелкая полоса растёт из-за самого маркера или из-за текста?

`marker_p3` даёт мелкую полосу +71% при радиусе ячеек +5…15% — самый большой
выпад из зависимости «мелкая ~ радиус». marker_cell.py показал, что ячейка
маркера не рыхлая (0.323 против 0.309 у прочих), но даёт **65% всех коротких
рёбер**: короткий конец перестаёт быть набором пар-дубликатов и становится одним
облаком из 42 почти равноудалённых точек.

Если это так, то достаточно выбросить маркерные токены из облака перед оценкой,
и подъём должен исчезнуть — при том, что текст остаётся испорченным ровно так
же. Если подъём останется, дело в самом тексте, а не в комке.

Три условия на одних текстах и одних зёрнах: исходный; маркер; маркер без
маркерных токенов в облаке (выборка L = 201 берётся из оставшихся).
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
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

BASE = os.path.join(HERE, "..")
SEEDS = 3
B = ["крупный", "мелкий", "средний"]


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def bands(e, seed):
    if e.shape[0] < cfg.L_DEFAULT:
        return None
    rng = np.random.default_rng(1000 + seed)
    idx = rng.choice(e.shape[0], size=cfg.L_DEFAULT, replace=False)
    return bands_of(qphd(e[idx], q_list=cfg.Q_GRID, rng=rng,
                         **cfg.qphd_kwargs(L=cfg.L_DEFAULT,
                                           pool=cfg.make_pool(e, cfg.L_DEFAULT,
                                                              rng))))


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mark = tok.tokenize(" item")[0]
    rows = []
    for i, s in enumerate(hum[:50]):
        got = {}
        for label, name in [("исходный", "identity"),
                            ("маркер", "marker_p3")]:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            e = cached(txt, f"pg_{name}_{i}")
            t_all = tok.tokenize(txt)
            if e is None or len(t_all) != e.shape[0]:
                continue
            keep = [j for j, x in enumerate(t_all) if x not in SKIP]
            got[label] = (e[keep], [t_all[j] for j in keep])
        if len(got) < 2:
            continue
        em, tm = got["маркер"]
        no_mark = em[[j for j, x in enumerate(tm) if x != mark]]
        for seed in range(SEEDS):
            r = {"текст": i, "зерно": seed}
            for lab, arr in [("исходный", got["исходный"][0]),
                             ("маркер", em), ("маркер без маркера", no_mark)]:
                b = bands(arr, seed)
                if b is None:
                    r = None
                    break
                for k, v in b.items():
                    r[f"{lab}: {k}"] = v
            if r:
                rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1} текстов", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "marker_drop.csv"), index=False)
    G = D.groupby("текст").mean(numeric_only=True)
    print(f"\n{len(G)} текстов, {SEEDS} зерна\n")
    print(f"{'полоса':10s}{'исходный':>12s}{'маркер':>12s}{'без маркера':>14s}"
          f"{'сдвиг маркера':>16s}{'сдвиг без него':>16s}{'p':>10s}")
    for b in B:
        a = G[f"исходный: {b}"]
        m = G[f"маркер: {b}"]
        n = G[f"маркер без маркера: {b}"]
        p = wilcoxon(a, n).pvalue
        print(f"{b:10s}{a.mean():12.2f}{m.mean():12.2f}{n.mean():14.2f}"
              f"{((m - a) / a).median() * 100:+15.1f}%"
              f"{((n - a) / a).median() * 100:+15.1f}%{p:>10.1e}")


if __name__ == "__main__":
    main()
