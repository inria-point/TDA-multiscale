"""Что это за одно ребро, на котором держится весь прирост разброса.

long_end_split.py показал: у перемешивания весь рост разброса длин на длинном
конце создаётся **одним** ребром на текст (выбросить его — и разброс падает
ниже исходного), у подстановки хапаксов — тем же одним ребром наполовину.
Здесь это ребро опознаётся: кто на концах, к какому классу они относятся, не
сток ли это, сколько точек отрезает разрез и насколько оно длиннее второго.

Печатается сводка по 50 текстам и десяток примеров дословно.
"""
import hashlib
import os
import sys
from collections import Counter

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
from edge_taxonomy import SKIP, corpus_ranks, token_class
from embedder import CACHE_DIR, MODEL_NAME
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
N_TEXTS = 50
NORM_CUT = 30.0


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def top_edge(text, key, tok, ranks):
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
    scale = pdist(v).mean()
    mst = minimum_spanning_tree(squareform(pdist(v))).tocoo()
    L = mst.data / scale
    o = np.argsort(L)[::-1]
    i0 = o[0]
    a, b = int(mst.row[i0]), int(mst.col[i0])
    nrm = np.linalg.norm(v, axis=1)
    cnt = Counter(t)
    m = np.ones(len(mst.data), bool)
    m[i0] = False
    g = coo_matrix((mst.data[m], (mst.row[m], mst.col[m])),
                   shape=(len(v), len(v)))
    _, lab = connected_components(g, directed=False)
    side = int(min((lab == lab[a]).sum(), (lab == lab[b]).sum()))
    cls = [token_class(t[a], ranks), token_class(t[b], ranks)]
    if nrm[a] < NORM_CUT:
        cls[0] = "сток"
    if nrm[b] < NORM_CUT:
        cls[1] = "сток"
    return {"длина": L[i0], "второе": L[o[1]], "отрыв": L[i0] / L[o[1]],
            "медиана рёбер": float(np.median(L)),
            "к медиане": L[i0] / float(np.median(L)),
            "отрезает точек": side,
            "токен A": t[a], "токен B": t[b],
            "класс A": cls[0], "класс B": cls[1],
            "норма A": nrm[a], "норма B": nrm[b],
            "разрыв норм": abs(nrm[a] - nrm[b]),
            "косинус": float(v[a] @ v[b] / (nrm[a] * nrm[b])),
            "одиночка A": cnt[t[a]] == 1, "одиночка B": cnt[t[b]] == 1,
            "есть сток": "сток" in cls}


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ranks = corpus_ranks(hum, type("E", (), {"tokenizer": tok})())
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in [("исходный", "identity"),
                            ("чужие хапаксы", "hapax_swap_wide"),
                            ("перемешивание", "shuffle_words")]:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            r = top_edge(txt, f"pg_{name}_{i}", tok, ranks)
            if r:
                rows.append({"текст": i, "условие": label, **r})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "longest_edge.csv"), index=False)
    pd.set_option("display.width", 250)
    num = ["длина", "второе", "отрыв", "к медиане", "отрезает точек",
           "разрыв норм", "косинус"]
    print("Самое длинное ребро текста, среднее по 50 текстам:\n")
    print(D.groupby("условие")[num].mean().round(3).to_string())
    print("\nДоля случаев:")
    for cond, g in D.groupby("условие"):
        print(f"  {cond:16s} со стоком {g['есть сток'].mean()*100:5.1f}%   "
              f"оба одиночки {(g['одиночка A'] & g['одиночка B']).mean()*100:5.1f}%   "
              f"отрезает одну точку {(g['отрезает точек'] == 1).mean()*100:5.1f}%")
    print("\nКлассы концов (доля рёбер, где класс встречается):")
    for cond, g in D.groupby("условие"):
        c = Counter(list(g["класс A"]) + list(g["класс B"]))
        tot = sum(c.values())
        print(f"  {cond:16s} " + "  ".join(
            f"{k} {v/tot*100:.0f}%" for k, v in c.most_common()))
    print("\nПримеры (первые восемь текстов):")
    for i in range(8):
        for cond in ["исходный", "чужие хапаксы", "перемешивание"]:
            g = D[(D["текст"] == i) & (D["условие"] == cond)]
            if not len(g):
                continue
            r = g.iloc[0]
            print(f"  т{i:<3d} {cond:15s} {r['токен A']!r:>16s} ({r['класс A']:>11s}, "
                  f"|v|={r['норма A']:.0f}) — {r['токен B']!r:<16s} "
                  f"({r['класс B']:>11s}, |v|={r['норма B']:.0f})   "
                  f"длина {r['длина']:.2f}, к медиане {r['к медиане']:.2f}, "
                  f"отрезает {r['отрезает точек']:.0f}")
        print()


if __name__ == "__main__":
    main()
