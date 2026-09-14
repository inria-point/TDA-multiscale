"""Переживает ли связь «крупная полоса ~ неровность длинного конца» удаление стоков.

CV верхних 20% рёбер — единственная величина, коррелирующая с крупной полосой в
обоих семействах условий (−0.60 механические, −0.66 LLM). Но разбор показал,
что её прирост наполовину, а у перемешивания целиком, создаётся одним ребром —
перемычкой к группе стока внимания, свойству энкодера. На трёх условиях с
удалением стоков прирост у перемешивания исчезал (+2.2%), у подстановки
хапаксов оставался (+31.7%).

Здесь то же самое на всех условиях обоих прогонов: стоковые токены выбрасываются
из облака до выборки, считается CV длинного конца, и связь с уже посчитанными
полосами пересчитывается. qPHD не запускается — полосы берутся из прогонов,
меняется только объясняющая величина.
"""
import hashlib
import json
import os
import sys

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
N_TEXTS, NORM_CUT, SEEDS = 50, 30.0, 3


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def cv(text, key, tok, drop_sink):
    e = cached(text, key)
    t_all = tok.tokenize(text)
    if e is None or len(t_all) != e.shape[0]:
        return None
    keep = [j for j, x in enumerate(t_all) if x not in SKIP]
    e = e[keep]
    if drop_sink:
        e = e[np.linalg.norm(e, axis=1) >= NORM_CUT]
    if e.shape[0] < cfg.L_DEFAULT:
        return None
    out = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        v = e[rng.choice(e.shape[0], size=cfg.L_DEFAULT, replace=False)]
        w = minimum_spanning_tree(squareform(pdist(v))).tocoo().data / pdist(v).mean()
        top = w[w >= np.percentile(w, 80)]
        out.append(top.std() / top.mean())
    return float(np.mean(out))


def collect(tok):
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in MECH:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            for ds, tag in [(False, "со стоками"), (True, "без стоков")]:
                c = cv(txt, f"pg_{name}_{i}", tok, ds)
                if c is not None:
                    rows.append({"набор": "механические", "текст": i,
                                 "условие": label, "вариант": tag, "CV": c})
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
            for ds, tag in [(False, "со стоками"), (True, "без стоков")]:
                c = cv(txt, f"pgl_{label}_{k[-8:]}", tok, ds)
                if c is not None:
                    rows.append({"набор": "LLM", "текст": k, "условие": label,
                                 "вариант": tag, "CV": c})
    return pd.DataFrame(rows)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    D = collect(tok)
    D.to_csv(os.path.join(BASE, "results", "cv_nosink_all.csv"), index=False)
    bands = {
        "механические": pd.read_csv(os.path.join(
            BASE, "results", "pert_geometry_shifts.csv")).set_index("условие"),
        "LLM": pd.read_csv(os.path.join(
            BASE, "results", "pert_geometry_llm_shifts.csv")).set_index("условие")}
    print(f"{len(D)} замеров\n")
    all_rows = []
    for nab, g in D.groupby("набор"):
        print(f"=== {nab} ===")
        B = bands[nab]
        for var, gg in g.groupby("вариант"):
            piv = gg.pivot_table(index="текст", columns="условие", values="CV")
            if "исходный" not in piv:
                continue
            sh = {}
            for c in piv.columns:
                if c == "исходный":
                    continue
                m = piv["исходный"].notna() & piv[c].notna()
                sh[c] = ((piv.loc[m, c] - piv.loc[m, "исходный"])
                         / piv.loc[m, "исходный"]).median() * 100
            S = pd.Series(sh).to_frame("ΔCV").join(B[["крупный", "мелкий",
                                                      "средний"]])
            S = S.dropna()
            r = pearsonr(S["ΔCV"], S["крупный"])
            rho = spearmanr(S["ΔCV"], S["крупный"])
            print(f"  {var:12s} n={len(S):2d}  r(крупная, ΔCV) = {r[0]:+.2f}"
                  f" / ро {rho[0]:+.2f}")
            all_rows.append(S.assign(набор=nab, вариант=var))
    A = pd.concat(all_rows)
    print("\n=== вместе ===")
    for var, g in A.groupby("вариант"):
        print(f"  {var:12s} n={len(g):2d}  "
              f"r = {pearsonr(g['ΔCV'], g['крупный'])[0]:+.2f} / "
              f"ро {spearmanr(g['ΔCV'], g['крупный'])[0]:+.2f}")
    print("\nΔCV по условиям (без стоков против со стоками):")
    P = A.pivot_table(index=A.index, columns="вариант", values="ΔCV")
    P["крупная"] = A.groupby(A.index)["крупный"].first()
    print(P.round(1).sort_values("крупная").to_string())


if __name__ == "__main__":
    main()
