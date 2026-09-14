"""Куда деваются короткие парные рёбра при перемешивании — и чьи они.

Перемешивание растягивает горб парных рёбер (медиана 13.18 → 14.30 в
абсолютных единицах, +8.5%) и слегка уменьшает их долю. Вопрос: у каких токенов
это происходит. Пунктуация и служебные слова повторяются по необходимости и
своего лексического содержания почти не имеют; смысловые повторяются потому,
что текст о них говорит. Если растягивание идёт от потери контекста, оно должно
быть сильнее там, где контекст вообще что-то решал.

Считается на тех же выборках: парные рёбра (оба конца — вхождения одного и того
же токена) разбиты по классу токена, для каждого класса число рёбер на текст,
медиана длины в абсолютных единицах и доля коротких (ниже исходной медианы
парных, 13.2).
"""
import hashlib
import os
import sys
from collections import Counter

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
from edge_taxonomy import SKIP, corpus_ranks, token_class
from embedder import CACHE_DIR, MODEL_NAME
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
N_TEXTS = 50
NORM_CUT = 30.0
SHORT = 13.2  # медиана парных рёбер неиспорченного текста, абсолютная
NAMES = {"пункт": "пунктуация", "служ": "служебные",
         "смысл-част": "смысловые частые", "смысл-редк": "смысловые редкие",
         "подслово": "подслова"}


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def pairs(text, key, tok, ranks):
    e = cached(text, key)
    t_all = tok.tokenize(text)
    if e is None or len(t_all) != e.shape[0]:
        return None
    keep = [j for j, x in enumerate(t_all) if x not in SKIP]
    if len(keep) < cfg.L_DEFAULT:
        return None
    e, toks = e[keep], [t_all[j] for j in keep]
    idx = np.random.default_rng(1000).choice(len(toks), size=cfg.L_DEFAULT,
                                             replace=False)
    v, t = e[idx], [toks[j] for j in idx]
    sink = np.linalg.norm(v, axis=1) < NORM_CUT
    scale = pdist(v[~sink]).mean()
    mst = minimum_spanning_tree(squareform(pdist(v))).tocoo()
    out = []
    for a, b, w in zip(mst.row, mst.col, mst.data):
        if sink[a] or sink[b] or t[a] != t[b]:
            continue
        out.append((NAMES.get(token_class(t[a], ranks), "прочее"), w,
                    w / scale))
    return out, scale


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ranks = corpus_ranks(hum, type("E", (), {"tokenizer": tok})())
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in [("исходный", "identity"),
                            ("перемешивание", "shuffle_words"),
                            ("чужие хапаксы", "hapax_swap_wide")]:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            r = pairs(txt, f"pg_{name}_{i}", tok, ranks)
            if r:
                rows += [{"текст": i, "условие": label, "класс": c,
                          "длина": w, "длина норм": wn} for c, w, wn in r[0]]
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "pair_edges_by_class.csv"),
             index=False)
    pd.set_option("display.width", 220)

    print(f"{len(D)} парных рёбер\n")
    print("Парные рёбра по классам: сколько их на текст и какой длины "
          "(абсолютные единицы)\n")
    hdr = f"{'класс':20s}"
    for c in ["исходный", "перемешивание", "чужие хапаксы"]:
        hdr += f"{c:>26s}"
    print(hdr)
    print(f"{'':20s}" + "   рёбер  медиана  короткие" * 3)
    for k in ["пунктуация", "служебные", "смысловые частые",
              "смысловые редкие", "подслова"]:
        line = f"{k:20s}"
        for c in ["исходный", "перемешивание", "чужие хапаксы"]:
            g = D[(D["условие"] == c) & (D["класс"] == k)]
            n = len(g) / D[D["условие"] == c]["текст"].nunique()
            line += f"{n:9.1f}{g['длина'].median():9.2f}"
            line += f"{(g['длина'] < SHORT).mean() * 100:9.1f}"
        print(line)

    print("\nСдвиг при перемешивании, парно по текстам:")
    for k in ["пунктуация", "служебные", "смысловые частые",
              "смысловые редкие", "подслова"]:
        a = (D[(D["условие"] == "исходный") & (D["класс"] == k)]
             .groupby("текст")["длина"].median())
        b = (D[(D["условие"] == "перемешивание") & (D["класс"] == k)]
             .groupby("текст")["длина"].median())
        na = (D[(D["условие"] == "исходный") & (D["класс"] == k)]
              .groupby("текст").size())
        nb = (D[(D["условие"] == "перемешивание") & (D["класс"] == k)]
              .groupby("текст").size())
        ix = a.index.intersection(b.index)
        ixn = na.index.intersection(nb.index)
        print(f"  {k:20s} медиана {((b[ix] - a[ix]) / a[ix]).median() * 100:+6.1f}%"
              f"  (p={wilcoxon(a[ix], b[ix]).pvalue:.1e})   "
              f"число рёбер {((nb[ixn] - na[ixn]) / na[ixn]).median() * 100:+6.1f}%")


if __name__ == "__main__":
    main()
