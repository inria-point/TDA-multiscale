"""Остаётся ли каркас регулярной мерой — или перестаёт ею быть.

report_1.pdf определяет d_g через условие Альфорса: мера d-регулярна, если
c₁ρ^d ≤ μ(B(x,ρ)) ≤ c₂ρ^d для ВСЕХ точек носителя с одними и теми же
константами. Показатель d существует только вместе с этим условием, поэтому
«изменилось d_g» и «каркас стал нерегулярным» — не альтернативы: второе
означает, что первого просто нет.

Значит замера эффективного показателя на облаке центров (frame_dimension.py)
мало. Нужно проверить само условие: одинаков ли локальный показатель ПО ТОЧКАМ.

Для каждого центра x берётся число соседей в шарах растущего радиуса, и по
наклону log #{y : ‖y−x‖ ≤ ρ} против log ρ считается локальный показатель d(x).
Регулярность означает, что разброс d(x) по точкам мал и не растёт при порче.
Склейка одиночек обязана его поднять: у склеенной точки сосед есть на радиусе,
на котором у прочих пусто.

Сообщается: средний d(x), его разброс по точкам, и отдельно — разброс у
одиночек против повторяющихся типов.
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
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
# радиусы, на которых считается локальный показатель: доли средней парной
# дистанции каркаса. В многомерном облаке расстояния концентрированы — до
# ближайшего центра 0.79 при средней парной 1.0, — поэтому шары меньше 0.8
# почти всегда пусты, и сетка начинается оттуда
RHO = np.array([0.80, 0.88, 0.96, 1.04, 1.12])


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def local_dims(text, key, tok):
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
        where = defaultdict(list)
        for j, x in enumerate(t):
            where[x].append(j)
        keys = list(where)
        cen = np.array([v[where[x]].mean(0) for x in keys])
        single = np.array([len(where[x]) == 1 for x in keys])
        if len(cen) < 60:
            continue
        D = squareform(pdist(cen))
        scale = pdist(cen).mean()
        counts = np.array([[np.sum(D[i] <= r * scale) - 1 for r in RHO]
                           for i in range(len(cen))], float)
        # log(1 + n) вместо отбора точек с достаточным числом соседей:
        # выбрасывались бы ровно изолированные точки, ради которых замер и
        # делается, и разброс показателя получался бы заниженным
        lg = np.log1p(counts)
        d_loc = np.polyfit(np.log(RHO), lg.T, 1)[0]
        acc.append({"d(x) среднее": float(d_loc.mean()),
                    "d(x) разброс": float(d_loc.std()),
                    "d(x) разброс, одиночки":
                        float(d_loc[single].std()) if single.sum() > 5 else np.nan,
                    "d(x) разброс, повторные":
                        float(d_loc[~single].std()) if (~single).sum() > 5 else np.nan,
                    "d(x) одиночки": float(d_loc[single].mean())
                        if single.sum() > 5 else np.nan,
                    "d(x) повторные": float(d_loc[~single].mean())
                        if (~single).sum() > 5 else np.nan,
                    "центров": int(len(cen))})
    if not acc:
        return None
    return {k: float(np.nanmean([a[k] for a in acc])) for k in acc[0]}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in MECH:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            r = local_dims(txt, f"pg_{name}_{i}", tok)
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
            r = local_dims(txt, f"pgl_{label}_{k[-8:]}", tok)
            if r:
                rows.append({"набор": "LLM", "текст": k, "условие": label, **r})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "frame_regularity.csv"), index=False)

    B = pd.concat([pd.read_csv(os.path.join(BASE, "results", f)).set_index(
        "условие") for f in ("pert_geometry_shifts.csv",
                             "pert_geometry_llm_shifts.csv")])
    B = B[~B.index.duplicated(keep="first")]
    base = D[D["условие"] == "исходный"]
    print(f"\nКаркас неиспорченного текста: {base['центров'].mean():.0f} центров, "
          f"d(x) = {base['d(x) среднее'].mean():.2f} ± "
          f"{base['d(x) разброс'].mean():.2f} по точкам\n")
    cols = ["d(x) среднее", "d(x) разброс", "d(x) разброс, одиночки",
            "d(x) разброс, повторные", "d(x) одиночки", "d(x) повторные"]
    rows = []
    for nab, g in D.groupby("набор"):
        piv = {c: g.pivot_table(index="текст", columns="условие", values=c)
               for c in cols}
        for cond in piv[cols[0]].columns:
            if cond == "исходный":
                continue
            r = {"условие": cond}
            for c in cols:
                a, b = piv[c]["исходный"], piv[c][cond]
                m = a.notna() & b.notna()
                if m.sum() < 8:
                    continue
                r[c] = ((b[m] - a[m]) / a[m]).median() * 100
                if c == "d(x) разброс":
                    r["p"] = wilcoxon(a[m], b[m]).pvalue
            rows.append(r)
    S = pd.DataFrame(rows).set_index("условие").join(
        B[["крупный"]].rename(columns={"крупный": "полоса"}))
    S = S.dropna(subset=["полоса"]).sort_values("полоса")
    print("Сдвиг, %: нарушается ли регулярность каркаса\n")
    print(S.round(2).to_string())


if __name__ == "__main__":
    main()
