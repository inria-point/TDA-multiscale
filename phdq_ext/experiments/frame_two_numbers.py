"""Два числа, каждое из которых обнуляется на модели report_1.pdf.

У них каркас однороден дважды, и обе однородности у нас нарушены — но разными
ручками, и по-разному.

  определение 2.5(a): центры ВСЕХ ячеек — выборка из ОДНОЙ меры μ_g.
      Значит ячейки с k = 1 и k ≥ 2 обменны, и ближайший сосед точки каркаса
      обязан быть «своего» вида ровно с частотой, задаваемой долями видов.

      S = P(ближайший сосед того же вида) − P при обменности

      Ноль при одной мере. Положительно, когда популяции разнесены.

  определение 2.4: мера d-регулярна, c₁ρ^d ≤ μ(B(x,ρ)) ≤ c₂ρ^d с одними
      константами по всему носителю.

      C = Var(#соседей в шаре) / E(#соседей в шаре)

      Единица при i.i.d. выборке (пуассоновский предел), больше при сгустках.

Радиус шара берётся как медиана расстояний до ближайшего соседа, умноженная на
1.5: достаточно мал, чтобы видеть локальную плотность, достаточно велик, чтобы
счёт не был нулевым.

Проверяется главное: раздвигание должно двигать S и не трогать C, сгущение —
наоборот. Если так, числа независимы и годятся как координаты.
"""
import hashlib
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wilcoxon
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import SKIP
from embedder import CACHE_DIR, MODEL_NAME
from perturb import apply as perturb
from synthetic_regimes import N_SING, build

NORM_CUT, SEEDS, N_TEXTS = 30.0, 3, 50
CONDS = [("исходный", "identity"), ("чужие хапаксы", "hapax_swap_wide"),
         ("хапаксы одного домена", "hapax_swap_local"),
         ("перемешивание", "shuffle_words"), ("эхо x4", "loop_local_3"),
         ("все слова разные", "expand_vocab_100")]


def two_numbers(points, is_single):
    """S и C на облаке каркаса с метками вида."""
    n = len(points)
    D = squareform(pdist(points))
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    same = is_single[nn] == is_single
    ns, nc = int(is_single.sum()), int((~is_single).sum())
    # ожидание при обменности: сосед своего вида с вероятностью (n_вида−1)/(n−1)
    exp = np.where(is_single, (ns - 1) / (n - 1), (nc - 1) / (n - 1))
    S = float(same.mean() - exp.mean())
    rho = 1.5 * np.median(D.min(1))
    cnt = (D <= rho).sum(1).astype(float)
    C = float(cnt.var() / cnt.mean()) if cnt.mean() > 0 else np.nan
    return S, C


def frame_of(text, key, tok, seed):
    e_ = cached(text, key)
    t_all = tok.tokenize(text)
    if e_ is None or len(t_all) != e_.shape[0]:
        return None
    keep = [j for j, x in enumerate(t_all) if x not in SKIP]
    e_, toks = e_[keep], [t_all[j] for j in keep]
    ok = np.linalg.norm(e_, axis=1) >= NORM_CUT
    e_, toks = e_[ok], [x for x, k in zip(toks, ok) if k]
    if e_.shape[0] < cfg.L_DEFAULT:
        return None
    rng = np.random.default_rng(1000 + seed)
    idx = rng.choice(e_.shape[0], size=cfg.L_DEFAULT, replace=False)
    v, t = e_[idx], [toks[j] for j in idx]
    where = defaultdict(list)
    for j, x in enumerate(t):
        where[x].append(j)
    cen = np.array([v[ix].mean(0) for ix in where.values()])
    sing = np.array([len(ix) == 1 for ix in where.values()])
    return cen, sing


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def main():
    print("=== синтетика: разделяют ли S и C две ручки? ===\n")
    knobs = [("база", {}), ("раздвинуть +10%", dict(push=0.10)),
             ("раздвинуть +20%", dict(push=0.20)),
             ("утопить 30%", dict(pull_share=0.30)),
             ("спарить 50%", dict(glue_share=0.50)),
             ("спарить 100%", dict(glue_share=1.00))]
    rows = []
    for name, kw in knobs:
        for rep in range(12):
            rng = np.random.default_rng(hash((name, rep)) % 2 ** 31)
            X, g, rad = build(0.06, rng=rng, **kw)
            # каркас: одиночки как есть, ячейки свёрнуты в центры
            cell_pts = X[N_SING:]
            ks = len(cell_pts)
            # восстановить центры не можем — но в этой синтетике ячейки идут
            # подряд; берём среднее каждой группы по разбиению на равные части
            # невозможно, поэтому каркас = одиночки + центроиды кластеров MST
            from scipy.cluster.hierarchy import fcluster, linkage
            Z = linkage(cell_pts, method="single")
            lab = fcluster(Z, t=2.5 * rad, criterion="distance")
            cen = np.array([cell_pts[lab == c].mean(0) for c in np.unique(lab)])
            pts = np.vstack([X[:N_SING], cen])
            sing = np.r_[np.ones(N_SING, bool), np.zeros(len(cen), bool)]
            S, C = two_numbers(pts, sing)
            rows.append({"ручка": name, "S": S, "C": C})
    D = pd.DataFrame(rows).groupby("ручка").mean()
    base = D.loc["база"]
    D["ΔS"] = D["S"] - base["S"]
    D["ΔC"] = D["C"] - base["C"]
    print(D.reindex([n for n, _ in knobs]).round(4).to_string())

    print("\n\n=== текст ===\n")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    pool = pd.read_parquet(os.path.join(HERE, "..", "..", "coling",
                                        "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in CONDS:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            vals = []
            for seed in range(SEEDS):
                f = frame_of(txt, f"pg_{name}_{i}", tok, seed)
                if f:
                    vals.append(two_numbers(*f))
            if vals:
                rows.append({"текст": i, "условие": label,
                             "S": float(np.mean([x[0] for x in vals])),
                             "C": float(np.mean([x[1] for x in vals]))})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(HERE, "..", "results", "frame_two_numbers.csv"),
             index=False)
    print(T.groupby("условие")[["S", "C"]].mean().round(4).to_string())
    print("\nПарно против исходного:\n")
    for c in ("S", "C"):
        piv = T.pivot(index="текст", columns="условие", values=c)
        for cond in [x for x, _ in CONDS if x != "исходный"]:
            a, b = piv["исходный"], piv[cond]
            m = a.notna() & b.notna()
            print(f"  {c}  {cond:24s} {a[m].mean():+.4f} → {b[m].mean():+.4f}  "
                  f"({(b[m]-a[m]).median():+.4f})  p={wilcoxon(a[m], b[m]).pvalue:.1e}")


if __name__ == "__main__":
    main()
