"""Средняя проекция на «плохое» направление как величина на один текст.

fit_pmi_sink.py нашёл направление, куда сносит вектор токена, когда ближайший
контекст ничего про него не диктует, и показал, что порча разворачивает на него
общий снос. Но всё это — различия между условиями. Величины, считаемой с одного
текста и годной в признак, из этого ещё не следует.

Здесь она и считается: средняя проекция сноса на ось, со знаком и в долях
масштаба облака,

    proj = ⟨ (v − дом(тип)) · u ⟩ / масштаб

отдельно по смысловым словам, по смысловым одиночкам и по всем токенам.

**Ось оценивается на отложенных текстах.** Она была получена как среднее
смещений нижней трети по PMI, по тем же сорока текстам; проецировать их же на
неё — частично круговая процедура. Поэтому тексты делятся пополам, ось строится
на одной половине, проекции считаются на другой, и наоборот.

Два вопроса: (1) различает ли проекция условия, (2) связана ли она с крупной
полосой **по текстам внутри условия** — то есть годится ли в признак. Второе
важнее: разницу между условиями мы уже знаем, а связи «доза — отклик» по
текстам до сих пор не нашлось ни у одной величины.
"""
import hashlib
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, wilcoxon
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import SKIP, corpus_ranks, token_class
from embedder import CACHE_DIR, MODEL_NAME
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
HOME_TEXTS, HOME_MIN, NORM_CUT = 200, 5, 30.0
CONDS = [("исходный", None), ("чужие хапаксы", "hapax_swap_wide"),
         ("перемешивание", "shuffle_words")]
CONTENT = {"смысл-част", "смысл-редк"}


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


def sample(text, key, tok):
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
    return e[idx], [toks[j] for j in idx]


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    home = home_vectors(hum[:HOME_TEXTS], tok)
    # corpus_ranks ждёт Embedder; здесь модель не нужна, только токенизатор
    ranks = corpus_ranks(hum, type("E", (), {"tokenizer": tok})())
    P = pd.read_csv(os.path.join(BASE, "results", "fit_pmi.csv"))
    n_texts = P["текст"].max() + 1

    # собрать смещения один раз
    store = {}
    for i, s in enumerate(hum[:n_texts]):
        for label, name in CONDS:
            txt = s if name is None else perturb(name, s, seed=i)
            got = sample(txt, f"pg_{name or 'identity'}_{i}", tok)
            if got is None:
                continue
            v, t = got
            g = P[(P["текст"] == i) & (P["условие"] == label)]
            if len(g) != len(t) or list(g["токен"]) != t:
                continue
            sink = np.linalg.norm(v, axis=1) < NORM_CUT
            keep = [j for j, x in enumerate(t)
                    if x in home and not sink[j]]
            D = np.array([v[j] - home[t[j]] for j in keep])
            cnt = Counter(t)
            store[(i, label)] = dict(
                D=D, scale=pdist(v).mean(), pmi=g["PMI"].to_numpy()[keep],
                cls=[token_class(t[j], ranks) for j in keep],
                single=np.array([cnt[t[j]] == 1 for j in keep]))
    print(f"собрано {len(store)} пар текст x условие", flush=True)

    # ось по одной половине текстов, проекции по другой
    ids = sorted({i for i, _ in store})
    folds = [(ids[:len(ids) // 2], ids[len(ids) // 2:]),
             (ids[len(ids) // 2:], ids[:len(ids) // 2])]
    rows = []
    for fit_ids, use_ids in folds:
        acc = []
        for i in fit_ids:
            d = store.get((i, "перемешивание"))
            if d is None:
                continue
            lo = np.percentile(d["pmi"], 33)
            mu = d["D"][d["pmi"] <= lo].mean(0)
            acc.append(mu / np.linalg.norm(mu))
        u = np.mean(acc, 0)
        u /= np.linalg.norm(u)
        for i in use_ids:
            for label, _ in CONDS:
                d = store.get((i, label))
                if d is None:
                    continue
                proj = (d["D"] @ u) / d["scale"]
                cm = np.array([c in CONTENT for c in d["cls"]])
                r = {"текст": i, "условие": label,
                     "проекция: все": float(proj.mean()),
                     "проекция: смысловые": float(proj[cm].mean()),
                     "проекция: смысловые одиночки":
                         float(proj[cm & d["single"]].mean())
                         if (cm & d["single"]).any() else np.nan,
                     "доля смысловых с проекцией > 0":
                         float((proj[cm] > 0).mean() * 100)}
                rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "axis_projection.csv"), index=False)

    cols = [c for c in D.columns if c.startswith(("проекция", "доля"))]
    print("\nПроекция на ось, в долях масштаба облака (ось с отложенной "
          "половины текстов):")
    print(D.groupby("условие")[cols].mean().round(4).to_string())
    piv = D.pivot(index="текст", columns="условие")
    print("\nПарный тест против исходного (смысловые):")
    for cond in ["чужие хапаксы", "перемешивание"]:
        a = piv[("проекция: смысловые", "исходный")]
        b = piv[("проекция: смысловые", cond)]
        m = a.notna() & b.notna()
        print(f"   {cond:16s} {a[m].mean():+.4f} → {b[m].mean():+.4f}   "
              f"{sum(b[m] > a[m])}/{m.sum()} текстов   "
              f"p={wilcoxon(a[m], b[m]).pvalue:.1e}")

    G = (pd.read_csv(os.path.join(BASE, "results", "pert_geometry.csv"))
         .groupby(["текст", "условие"]).mean(numeric_only=True).reset_index())
    M = D.merge(G[["текст", "условие", "крупный", "мелкий", "средний",
                   "длинные: CV"]], on=["текст", "условие"])
    print("\nСвязь с полосами ВНУТРИ условия (по текстам) — годится ли в признак:")
    for cond, g in M.groupby("условие"):
        out = [f"   {cond:16s}"]
        for y in ["крупный", "длинные: CV"]:
            r = pearsonr(g["проекция: смысловые"], g[y])[0]
            out.append(f"{y}: r={r:+.2f}")
        print("   ".join(out) + f"   n={len(g)}")


if __name__ == "__main__":
    main()
