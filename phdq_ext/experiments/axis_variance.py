"""Среднее по оси — перенос и потому инертно. Работает ли разброс?

axis_transfer.py показал, что средняя проекция сноса на ось «контекст ничего не
диктует» отделяет испорченный текст от целого почти без ошибок. Но объяснить ею
падение крупной полосы нельзя даже в принципе: общий сдвиг всех точек в одну
сторону есть параллельный перенос, он не меняет ни одного расстояния, ни одного
ребра MST и ни одной оценки размерности.

Геометрию может менять только то, насколько проекция **различается** между
токенами. Поэтому здесь считаются обе величины, по смысловым словам и отдельно
по одиночкам:

  среднее    ⟨proj⟩ -- перенос, геометрически инертно
  разброс    sd(proj) -- растяжение облака вдоль оси, геометрически действует

Вопрос: какая из них связана с крупной полосой и с неровностью длинного конца.
Предсказание, вытекающее из §8: работать должен разброс среди одиночек, потому
что именно расслоение одиночек на «унесённые» и «оставшиеся» делает длинный
конец смесью двух популяций.
"""
import hashlib
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import SKIP, corpus_ranks, token_class
from embedder import CACHE_DIR, MODEL_NAME
from pert_geometry import PERTS as MECH
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
HOME_TEXTS, HOME_MIN, NORM_CUT = 200, 5, 30.0
CONTENT = {"смысл-част", "смысл-редк"}
N_TEXTS = 50


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def home_vectors(texts, tok):
    tot, cnt = {}, Counter()
    for i, s in enumerate(texts):
        e, t = cached(s, f"home_{i}"), tok.tokenize(s)
        if e is None or len(t) != e.shape[0]:
            continue
        for x, vv in zip(t, e):
            tot[x] = tot[x] + vv if x in tot else vv.copy()
            cnt[x] += 1
    return {x: tot[x] / cnt[x] for x in tot if cnt[x] >= HOME_MIN}


def stats(text, key, tok, home, ranks, u):
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
    ix = [j for j, x in enumerate(t) if x in home and not sink[j]]
    if len(ix) < 40:
        return None
    scale = pdist(v).mean()
    proj = (np.array([v[j] - home[t[j]] for j in ix]) @ u) / scale
    cnt = Counter(t)
    cm = np.array([token_class(t[j], ranks) in CONTENT for j in ix])
    sg = np.array([cnt[t[j]] == 1 for j in ix])
    out = {"среднее: смысловые": proj[cm].mean(),
           "разброс: смысловые": proj[cm].std()}
    m = cm & sg
    if m.sum() >= 10:
        out["среднее: смысл. одиночки"] = proj[m].mean()
        out["разброс: смысл. одиночки"] = proj[m].std()
    out["разброс: все"] = proj.std()
    out["среднее: все"] = proj.mean()
    # сколько токенов унесено СИЛЬНО: если это не все, длинный конец
    # расслаивается на две популяции, а вот это уже не перенос
    for th in (0.1, 0.2, 0.3, 0.5):
        out[f"смысловых > {th}, %"] = float((proj[cm] > th).mean() * 100)
        if m.sum() >= 10:
            out[f"одиночек > {th}, %"] = float((proj[m] > th).mean() * 100)
    # и насколько сама популяция проекций расслоена: разрыв между
    # верхней и нижней третью среди одиночек
    if m.sum() >= 10:
        q = np.percentile(proj[m], [33, 67])
        out["расслоение одиночек"] = float(q[1] - q[0])
    return out


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    home = home_vectors(hum[:HOME_TEXTS], tok)
    ranks = corpus_ranks(hum, type("E", (), {"tokenizer": tok})())
    u = np.load(os.path.join(BASE, "results",
                             "fit_pmi_axis.npz"))["перемешивание"]
    u = u / np.linalg.norm(u)

    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in MECH:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            r = stats(txt, f"pg_{name}_{i}", tok, home, ranks, u)
            if r:
                rows.append({"текст": i, "условие": label, **r})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1} текстов", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "axis_variance.csv"), index=False)

    G = (pd.read_csv(os.path.join(BASE, "results", "pert_geometry.csv"))
         .groupby(["текст", "условие"]).mean(numeric_only=True).reset_index())
    M = D.merge(G[["текст", "условие", "крупный", "мелкий", "средний",
                   "длинные: CV", "черешок одиночки"]],
                on=["текст", "условие"])
    cols = [c for c in D.columns if c.startswith(("среднее", "разброс"))]

    print("\nПо условиям (средние значения):")
    print(M.groupby("условие")[cols + ["крупный", "длинные: CV"]].mean()
          .round(4).sort_values("разброс: смысловые").to_string())

    base = M[M["условие"] == "исходный"].set_index("текст")
    sh = {}
    for cond, g in M.groupby("условие"):
        if cond == "исходный":
            continue
        g = g.set_index("текст")
        ix = base.index.intersection(g.index)
        r = {}
        for c in cols:
            r[c] = (g.loc[ix, c] - base.loc[ix, c]).median()
        for c in ["крупный", "длинные: CV", "черешок одиночки"]:
            r[c] = ((g.loc[ix, c] - base.loc[ix, c])
                    / base.loc[ix, c]).median() * 100
        sh[cond] = r
    S = pd.DataFrame(sh).T
    print("\nСдвиги по условиям:")
    print(S.round(3).sort_values("крупный").to_string())
    print("\nКорреляция по условиям:")
    for c in cols:
        line = f"   {c:26s}"
        for y in ["крупный", "длинные: CV", "черешок одиночки"]:
            m = S[[c, y]].dropna()
            line += f"  {y}: r={pearsonr(m[c], m[y])[0]:+.2f}"
        print(line)


if __name__ == "__main__":
    main()
