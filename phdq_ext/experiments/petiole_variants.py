"""Черешок одиночки без «бессмысленного» направления: три варианта.

Удаление стоков резко улучшило связь крупной полосы с неровностью длинного
конца (r −0.57 → −0.74 по 27 условиям): сток был в ней шумом. Ось «контекст
ничего не диктует» (fit_pmi_axis.npz) — второй кандидат в такой же шум, и
пользователь предложил проверить три способа её убрать.

  A. проекция   выбросить ось из пространства: спроецировать все векторы на
                ортогональное дополнение u, заново построить MST и померить
                всё там. Это вопрос «как выглядит геометрия, если бессмысленное
                направление просто не существует»
  B. отношение  у каждой одиночки разложить её черешок на составляющую вдоль
                оси и поперёк, и смотреть на поперечную и на их отношение
  C. порог      считать черешок только по «осмысленным» одиночкам — нижней
                трети по проекции на ось, — и отдельно по верхней

Всё на кэше эмбеддингов, qPHD не запускается: полосы берутся из прогонов,
меняется только объясняющая величина. Стоки исключены везде.
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
from scipy.stats import pearsonr, spearmanr
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
N_TEXTS, NORM_CUT, HOME_MIN, HOME_TEXTS = 50, 30.0, 5, 200


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


def tree(v):
    mst = minimum_spanning_tree(squareform(pdist(v))).tocoo()
    L = mst.data / pdist(v).mean()
    deg = np.zeros(len(v), int)
    longest = np.zeros(len(v))
    for a, b, w in zip(mst.row, mst.col, L):
        deg[a] += 1
        deg[b] += 1
        longest[a] = max(longest[a], w)
        longest[b] = max(longest[b], w)
    top = L[L >= np.percentile(L, 80)]
    return longest, deg, top.std() / top.mean(), mst, L


def measure(text, key, tok, home, u):
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
    out = {}
    for seed in range(3):
        rng = np.random.default_rng(1000 + seed)
        idx = rng.choice(e.shape[0], size=cfg.L_DEFAULT, replace=False)
        v, t = e[idx], [toks[j] for j in idx]
        cnt = Counter(t)
        sg = np.array([cnt[x] == 1 for x in t])
        r = {}
        # исходное пространство
        longest, deg, cvt, mst, L = tree(v)
        r["черешок"] = longest[sg].mean()
        r["CV длинного"] = cvt
        # A. пространство без оси
        vp = v - np.outer(v @ u, u)
        longest_p, _, cvp, _, _ = tree(vp)
        r["черешок без оси"] = longest_p[sg].mean()
        r["CV без оси"] = cvp
        # B. разложение черешка одиночки на вдоль и поперёк оси
        scale = pdist(v).mean()
        par, orth = [], []
        for j in np.where(sg)[0]:
            k = None
            for a, b, w in zip(mst.row, mst.col, L):
                if (a == j or b == j) and abs(w - longest[j]) < 1e-9:
                    k = b if a == j else a
                    break
            if k is None:
                continue
            d = v[j] - v[k]
            p = float(abs(d @ u)) / scale
            par.append(p)
            orth.append(float(np.sqrt(max(np.dot(d, d) / scale ** 2
                                          - p ** 2, 0))))
        if par:
            par_a, orth_a = np.array(par), np.array(orth)
            tot_a = np.sqrt(par_a ** 2 + orth_a ** 2)
            r["черешок вдоль оси"] = float(par_a.mean())
            r["черешок поперёк оси"] = float(orth_a.mean())
            r["доля вдоль оси"] = float(np.mean(par_a / (par_a + orth_a)))
            # гипотеза пользователя: считать длину только тех черешков, чья
            # составляющая ВДОЛЬ оси мала по модулю, то есть которые почти
            # ортогональны оси определённости
            for th in (0.05, 0.1, 0.15, 0.2, 0.3):
                m = par_a < th
                r[f"черешок при |вдоль|<{th}"] = (float(tot_a[m].mean())
                                                  if m.sum() >= 5 else np.nan)
                r[f"доля |вдоль|<{th}, %"] = float(m.mean() * 100)
            m = par_a >= 0.2
            r["черешок при |вдоль|>=0.2"] = (float(tot_a[m].mean())
                                             if m.sum() >= 5 else np.nan)
        # C. порог по проекции: только «осмысленные» одиночки
        ix = [j for j in np.where(sg)[0] if t[j] in home]
        if len(ix) >= 12:
            proj = np.array([(v[j] - home[t[j]]) @ u for j in ix]) / scale
            lo, hi = np.percentile(proj, [33, 67])
            r["черешок, осмысленные"] = float(
                np.mean([longest[j] for j, p in zip(ix, proj) if p <= lo]))
            r["черешок, бессмысленные"] = float(
                np.mean([longest[j] for j, p in zip(ix, proj) if p >= hi]))
        for k_, v_ in r.items():
            out.setdefault(k_, []).append(v_)
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    home = home_vectors(hum[:HOME_TEXTS], tok)
    u = np.load(os.path.join(BASE, "results",
                             "fit_pmi_axis.npz"))["перемешивание"]
    u = u / np.linalg.norm(u)

    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in MECH:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            r = measure(txt, f"pg_{name}_{i}", tok, home, u)
            if r:
                rows.append({"набор": "механические", "текст": i,
                             "условие": label, **r})
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
            r = measure(txt, f"pgl_{label}_{k[-8:]}", tok, home, u)
            if r:
                rows.append({"набор": "LLM", "текст": k, "условие": label,
                             **r})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "petiole_variants.csv"),
             index=False)

    B = pd.concat([pd.read_csv(os.path.join(BASE, "results", f)).set_index(
        "условие") for f in ("pert_geometry_shifts.csv",
                             "pert_geometry_llm_shifts.csv")])
    B = B[~B.index.duplicated(keep="first")]
    cols = [c for c in D.columns if c not in ("набор", "текст", "условие")]
    sh = {}
    for nab, g in D.groupby("набор"):
        for c in cols:
            piv = g.pivot_table(index="текст", columns="условие", values=c)
            if "исходный" not in piv:
                continue
            for cond in piv.columns:
                if cond == "исходный":
                    continue
                m = piv["исходный"].notna() & piv[cond].notna()
                if m.sum() < 8:
                    continue
                sh.setdefault(cond, {})[c] = (
                    (piv.loc[m, cond] - piv.loc[m, "исходный"])
                    / piv.loc[m, "исходный"]).median() * 100
    S = pd.DataFrame(sh).T.join(B[["крупный", "мелкий", "средний"]]).dropna(
        subset=["крупный"])
    S.to_csv(os.path.join(BASE, "results", "petiole_variants_shifts.csv"))
    print(f"\n{len(S)} условий\n")
    print(f"{'величина':26s}{'все':>16s}{'механ.':>16s}{'LLM':>16s}")
    mech = S.index.isin(pd.read_csv(os.path.join(
        BASE, "results", "pert_geometry_shifts.csv"))["условие"])
    for c in cols:
        if c not in S:
            continue
        line = f"{c:26s}"
        for m in (np.ones(len(S), bool), mech, ~mech):
            g = S[m].dropna(subset=[c, "крупный"])
            line += (f"{pearsonr(g[c], g['крупный'])[0]:+8.2f}/"
                     f"{spearmanr(g[c], g['крупный'])[0]:+7.2f}"
                     if len(g) > 4 else f"{'—':>16s}")
        print(line)


if __name__ == "__main__":
    main()
