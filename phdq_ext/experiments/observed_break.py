"""Совпадает ли наблюдаемый разрыв спектра с лексическим q* после порчи?

Замечание пользователя: MST не знает меток токенов, он видит только расстояния.
Если две одиночки сблизились сильнее типичного радиуса ячейки, для дерева они
и есть одна ячейка из двух точек — ребро между ними уходит из каркасной страты
в ячеечную, эффективное число ячеек падает, q* растёт. Тогда склейка есть
скрытое изменение словаря, теорема её покрывает через q*, и вводить
неоднородность каркаса не нужно.

Проверяется прямо, и это основной замер их работы (рис. 1c): наблюдаемая точка
разрыва — квантиль, который лучше всего отделяет рёбра внутри типа от рёбер
между типами, — против лексического q* = (n−N)/(n−1), считаемого по меткам.

  разрыв поехал вверх от q*   → появились геометрические ячейки без лексических,
                                то есть склейка действительно меняет «словарь»
                                в том смысле, в каком его видит дерево
  разрыв остался на q*        → пары не стали ячейками, и падение полосы идёт
                                не через счётные величины

Качество разделения (доля правильно классифицированных рёбер при лучшем
разрезе) сообщается рядом: если расщепления нет вовсе, точка разрыва бессмысленна.
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
from scipy.stats import wilcoxon
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
        n = len(v)
        N = len(set(t))
        q_lex = (n - N) / (n - 1)
        D = squareform(pdist(v))
        mst = minimum_spanning_tree(D).tocoo()
        order = np.argsort(mst.data)
        same = np.array([t[mst.row[i]] == t[mst.col[i]] for i in order])
        m = len(same)
        # лучший разрез: максимум доли верно отнесённых рёбер
        # (ниже разреза ждём «тот же тип», выше — «разные»)
        below = np.cumsum(same)
        above = (~same)[::-1].cumsum()[::-1]
        acc_at = np.empty(m + 1)
        acc_at[0] = (~same).sum() / m
        acc_at[1:] = (below + np.r_[above[1:], 0]) / m
        k = int(np.argmax(acc_at))
        acc.append({"q* лексический": q_lex,
                    "разрыв наблюдаемый": k / m,
                    "качество разреза": float(acc_at[k]),
                    "доля одинаковых": float(same.mean())})
    return {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}


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
            r = measure(txt, f"pgl_{label}_{k[-8:]}", tok)
            if r:
                rows.append({"набор": "LLM", "текст": k, "условие": label, **r})
    D = pd.DataFrame(rows)
    D["разрыв − q*"] = D["разрыв наблюдаемый"] - D["q* лексический"]
    D.to_csv(os.path.join(BASE, "results", "observed_break.csv"), index=False)

    b = D[D["условие"] == "исходный"]
    print(f"\nНеиспорченный текст: q* лексический {b['q* лексический'].mean():.3f}, "
          f"наблюдаемый разрыв {b['разрыв наблюдаемый'].mean():.3f}, "
          f"расхождение {b['разрыв − q*'].mean():+.3f}, "
          f"качество разреза {b['качество разреза'].mean():.3f}\n")
    rows = []
    for nab, g in D.groupby("набор"):
        piv = {c: g.pivot_table(index="текст", columns="условие", values=c)
               for c in ("q* лексический", "разрыв наблюдаемый", "разрыв − q*",
                         "качество разреза")}
        for cond in piv["разрыв − q*"].columns:
            if cond == "исходный":
                continue
            r = {"условие": cond}
            for c in piv:
                a, x = piv[c]["исходный"], piv[c][cond]
                m = a.notna() & x.notna()
                if m.sum() < 8:
                    continue
                r["Δ " + c] = (x[m] - a[m]).median()
                if c == "разрыв − q*":
                    r["p"] = wilcoxon(a[m], x[m]).pvalue
            rows.append(r)
    S = pd.DataFrame(rows).set_index("условие")
    print("Сдвиги (абсолютные, в единицах квантиля):\n")
    print(S.round(4).sort_values("Δ разрыв − q*").to_string())


if __name__ == "__main__":
    main()
