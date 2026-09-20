"""Меняется ли размерность самого каркаса d_g при пертурбациях?

report_1.pdf, теорема B.3: каркасная полоса зависит ровно от β и d_g. Показатель
Хипса β при перемешивании и при подстановке чужих хапаксов не меняется
(heaps_beta.py: Δβ = +0.001 и +0.007), а крупная полоса падает на 11% и 17%.
Остаётся d_g. Если она меняется — теория справляется и неоднородность вводить
не нужно; если нет — нужна.

d_g меряется прямо: каждый тип токена сворачивается в среднее своих вхождений,
и qPHD считается на получившемся облаке центров. На нём каждый тип встречается
ровно один раз, значит q* = 0 и весь спектр каркасный — полосы этого облака
измеряют каркас и ничего кроме.

Сток исключается до всего: это ячейка, чей центр лежит вне носителя каркасной
меры (норма 26 против 37), и в каркас ей входить незачем.
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
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
from qphd import qphd
from sink_cluster import bands_of

BASE = os.path.join(HERE, "..")
N_TEXTS, NORM_CUT, SEEDS = 50, 30.0, 3
L_FRAME = 101          # точек каркаса; типов в выборке L=201 обычно ~122


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def frame_bands(text, key, tok):
    e = cached(text, key)
    t_all = tok.tokenize(text)
    if e is None or len(t_all) != e.shape[0]:
        return None
    keep = [j for j, x in enumerate(t_all) if x not in SKIP]
    e, toks = e[keep], [t_all[j] for j in keep]
    ok = np.linalg.norm(e, axis=1) >= NORM_CUT          # без стоков
    e, toks = e[ok], [x for x, k in zip(toks, ok) if k]
    if e.shape[0] < cfg.L_DEFAULT:
        return None
    out = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        idx = rng.choice(e.shape[0], size=cfg.L_DEFAULT, replace=False)
        v, t = e[idx], [toks[j] for j in idx]
        where = defaultdict(list)
        for j, x in enumerate(t):
            where[x].append(j)
        cen = np.array([v[ix].mean(0) for ix in where.values()])
        if len(cen) < L_FRAME:
            continue
        df = qphd(cen, q_list=cfg.Q_GRID, rng=rng,
                  **cfg.qphd_kwargs(L=L_FRAME,
                                    pool=cfg.make_pool(cen, L_FRAME, rng)))
        out.append({**bands_of(df), "типов": len(cen)})
    if not out:
        return None
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in MECH:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            r = frame_bands(txt, f"pg_{name}_{i}", tok)
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
    for n, (k, text) in enumerate(src.items()):
        if n >= N_TEXTS:
            break
        for label, txt in [("исходный", text)] + [
                (lab, store[lab].get(k)) for lab, _, _ in LLM]:
            if txt is None:
                continue
            r = frame_bands(txt, f"pgl_{label}_{k[-8:]}", tok)
            if r:
                rows.append({"набор": "LLM", "текст": k, "условие": label, **r})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "frame_dimension.csv"), index=False)

    B = pd.concat([pd.read_csv(os.path.join(BASE, "results", f)).set_index(
        "условие") for f in ("pert_geometry_shifts.csv",
                             "pert_geometry_llm_shifts.csv")])
    B = B[~B.index.duplicated(keep="first")]
    base = D[D["условие"] == "исходный"]
    print(f"\nКаркас неиспорченных текстов: {base['типов'].mean():.0f} центров, "
          f"d_g по полосам — крупная {base['крупный'].mean():.1f}, "
          f"мелкая {base['мелкий'].mean():.1f}, средняя {base['средний'].mean():.1f}\n")
    rows = []
    for nab, g in D.groupby("набор"):
        piv = {c: g.pivot_table(index="текст", columns="условие", values=c)
               for c in ("крупный", "мелкий", "средний", "типов")}
        for cond in piv["крупный"].columns:
            if cond == "исходный":
                continue
            r = {"условие": cond}
            for c in ("крупный", "мелкий", "типов"):
                a, b = piv[c]["исходный"], piv[c][cond]
                m = a.notna() & b.notna()
                if m.sum() < 8:
                    continue
                r[f"каркас: {c}"] = ((b[m] - a[m]) / a[m]).median() * 100
                if c == "крупный":
                    r["p"] = wilcoxon(a[m], b[m]).pvalue
            rows.append(r)
    S = pd.DataFrame(rows).set_index("условие").join(B[["крупный"]].rename(
        columns={"крупный": "облако: крупная"}))
    S = S.dropna(subset=["облако: крупная"]).sort_values("облако: крупная")
    print("Сдвиг, %:  каркас (облако центров) против полной полосы\n")
    print(S.round(2).to_string())


if __name__ == "__main__":
    main()
