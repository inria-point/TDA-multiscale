"""Меняется ли САМА размерность, если выбросить ось из пространства.

Аргумент против того, что ось важна для геометрии: однородный сдвиг всех точек
вдоль неё есть параллельный перенос, а перенос не меняет расстояний при любой
величине. Значение имеет только разнородная часть, а она почти не растёт
(разброс проекций 0.064 → 0.072 при сдвиге среднего на 0.32).

Аргумент за: длины рёбер в многомерном облаке концентрированы, CV длинного
конца всего 0.071, и вклад в полпроцента — это десятая часть наблюдаемого
разброса, а не пыль.

Спор решается прямым замером, которого не делалось: посчитать полосы на облаке
и на том же облаке, спроецированном на ортогональное дополнение к оси. Если
полосы не двигаются, ось геометрически инертна и её роль исчерпывается ролью
признака. Если двигаются — значит маленький сдвиг в 768 измерениях размерность
всё-таки меняет.

Контроль: то же самое с выброшенным СЛУЧАЙНЫМ направлением. Выбрасывание любого
направления само по себе что-то делает (размерность падает на единицу), и без
контроля эффект оси не отделить от эффекта процедуры.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import SKIP
from embedder import CACHE_DIR, MODEL_NAME
from perturb import apply as perturb
from qphd import qphd
from sink_cluster import bands_of

BASE = os.path.join(HERE, "..")
N_TEXTS, NORM_CUT, SEEDS = 40, 30.0, 3
B = ["крупный", "мелкий", "средний"]
CONDS = [("исходный", "identity"), ("чужие хапаксы", "hapax_swap_wide"),
         ("перемешивание", "shuffle_words")]


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def bands(e, seed):
    if e.shape[0] < cfg.L_DEFAULT:
        return None
    rng = np.random.default_rng(1000 + seed)
    idx = rng.choice(e.shape[0], size=cfg.L_DEFAULT, replace=False)
    return bands_of(qphd(e[idx], q_list=cfg.Q_GRID, rng=rng,
                         **cfg.qphd_kwargs(L=cfg.L_DEFAULT,
                                           pool=cfg.make_pool(e, cfg.L_DEFAULT,
                                                              rng))))


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    u = np.load(os.path.join(BASE, "results",
                             "fit_pmi_axis.npz"))["перемешивание"]
    u = u / np.linalg.norm(u)
    rg = np.random.default_rng(0)
    ur = rg.normal(size=u.shape)
    ur /= np.linalg.norm(ur)

    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in CONDS:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            e = cached(txt, f"pg_{name}_{i}")
            t_all = tok.tokenize(txt)
            if e is None or len(t_all) != e.shape[0]:
                continue
            keep = [j for j, x in enumerate(t_all) if x not in SKIP]
            e = e[keep]
            e = e[np.linalg.norm(e, axis=1) >= NORM_CUT]   # без стоков
            if e.shape[0] < cfg.L_DEFAULT:
                continue
            variants = {"как есть": e,
                        "без оси": e - np.outer(e @ u, u),
                        "без случайного": e - np.outer(e @ ur, ur)}
            for seed in range(SEEDS):
                r = {"текст": i, "условие": label, "зерно": seed}
                ok = True
                for vname, arr in variants.items():
                    bb = bands(arr, seed)
                    if bb is None:
                        ok = False
                        break
                    for k, v in bb.items():
                        r[f"{vname}: {k}"] = v
                if ok:
                    rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1} текстов", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "bands_without_axis.csv"),
             index=False)
    G = D.groupby(["текст", "условие"]).mean(numeric_only=True).reset_index()
    print(f"\n{G['текст'].nunique()} текстов, {SEEDS} зерна\n")
    for cond, g in G.groupby("условие"):
        print(f"--- {cond} ---")
        for b in B:
            a = g[f"как есть: {b}"]
            for v in ("без оси", "без случайного"):
                c = g[f"{v}: {b}"]
                print(f"  {b:9s} {v:16s} {a.mean():6.2f} → {c.mean():6.2f}  "
                      f"({((c - a) / a).median() * 100:+5.2f}%)  "
                      f"p={wilcoxon(a, c).pvalue:.1e}")
        print()


if __name__ == "__main__":
    main()
