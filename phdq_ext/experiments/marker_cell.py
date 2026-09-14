"""Как выглядит комок вставленного слова `item`, и почему он ломает зависимость.

Условие `marker_p3` вставляет слово `item` каждые три слова, то есть занимает
четверть выборки, и даёт мелкую полосу +71% при радиусе смысловых ячеек всего
+15% — единственный сильный выпад из самой надёжной зависимости прогона
(мелкая ~ радиус, R² = 0.83 на 26 условиях).

Подозрение: радиус усредняется **по типам**, а ячейка маркера — одна из
примерно девяти смысловых, но держит четверть всех точек. Если она при этом
сильно рыхлее прочих, усреднение по типам её почти не видит, а мелкая полоса
видит только её.

Меряется на тех же выборках, что и основной прогон (кэш эмбеддингов, энкодер не
запускается):

  сколько экземпляров        и какую долю выборки они занимают
  радиус ячейки маркера      против радиуса прочих смысловых ячеек
  радиус по токенам          усреднение с весом по числу вхождений
  до ближайшего соседа       у маркерных токенов против прочих
  зазор                      далеко ли ячейка маркера от чужих центров
  доля коротких рёбер        сколько из 20% самых коротких рёбер маркерные
  проекция на ось            насколько снос маркера лежит на оси
                             «контекст ничего не диктует» (fit_pmi_axis.npz)
"""
import hashlib
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import SKIP
from embedder import CACHE_DIR, MODEL_NAME
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
HOME_TEXTS, HOME_MIN = 200, 5
SEEDS = 3


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


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    home = home_vectors(hum[:HOME_TEXTS], tok)
    ax = np.load(os.path.join(BASE, "results", "fit_pmi_axis.npz"))["перемешивание"]
    ax = ax / np.linalg.norm(ax)
    mark = tok.tokenize(" item")[0]
    print(f"маркерный токен: {mark!r}; есть ли дом: {mark in home}", flush=True)

    rows = []
    for i, s in enumerate(hum[:50]):
        for label, name in [("исходный", "identity"), ("маркер каждые 3",
                                                       "marker_p3")]:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            e = cached(txt, f"pg_{name}_{i}")
            t_all = tok.tokenize(txt)
            if e is None or len(t_all) != e.shape[0]:
                continue
            keep = [j for j, x in enumerate(t_all) if x not in SKIP]
            if len(keep) < cfg.L_DEFAULT:
                continue
            e, toks = e[keep], [t_all[j] for j in keep]
            for seed in range(SEEDS):
                rng = np.random.default_rng(1000 + seed)
                idx = rng.choice(len(toks), size=cfg.L_DEFAULT, replace=False)
                v, t = e[idx], [toks[j] for j in idx]
                scale = pdist(v).mean()
                cnt = Counter(t)
                where = defaultdict(list)
                for j, x in enumerate(t):
                    where[x].append(j)

                D = squareform(pdist(v))
                np.fill_diagonal(D, np.inf)
                nn = D.min(1) / scale
                w = minimum_spanning_tree(squareform(pdist(v))).tocoo()
                short = w.data <= np.percentile(w.data, 20)
                is_m = np.array([x == mark for x in t])

                # ячейки: маркер против прочих типов с k>=2
                cells = {x: ix for x, ix in where.items() if len(ix) >= 2}
                rad = {x: np.linalg.norm(v[ix] - v[ix].mean(0),
                                         axis=1).mean() / scale
                       for x, ix in cells.items()}
                cen = {x: v[ix].mean(0) for x, ix in cells.items()}
                r = {"текст": i, "условие": label, "зерно": seed,
                     "типов k>=2": len(cells),
                     "радиус по типам": float(np.mean(list(rad.values()))),
                     "радиус по токенам": float(
                         np.average([rad[x] for x in cells],
                                    weights=[len(cells[x]) for x in cells])),
                     "до ближайшего, прочие": float(nn[~is_m].mean())}
                if is_m.any() and mark in rad:
                    others = [x for x in cells if x != mark]
                    dc = [np.linalg.norm(cen[mark] - cen[x]) / scale
                          for x in others]
                    r.update({
                        "экземпляров маркера": int(is_m.sum()),
                        "доля выборки, %": float(is_m.mean() * 100),
                        "радиус маркера": rad[mark],
                        "радиус прочих": float(np.mean(
                            [rad[x] for x in others])),
                        "до ближайшего, маркер": float(nn[is_m].mean()),
                        "до ближайшего центра": float(np.min(dc)),
                        "зазор маркера": float(np.min(
                            [np.linalg.norm(cen[mark] - cen[x]) / scale
                             / (rad[mark] + rad[x]) for x in others])),
                        "коротких рёбер маркерных, %": float(np.mean(
                            [is_m[a] and is_m[b] for a, b, k in
                             zip(w.row, w.col, short) if k]) * 100)})
                    if mark in home:
                        dm = v[is_m] - home[mark]
                        dm = dm / np.linalg.norm(dm, axis=1, keepdims=True)
                        r["маркер: доля на оси"] = float(np.mean((dm @ ax) ** 2))
                        r["маркер: смещение"] = float(np.linalg.norm(
                            v[is_m] - home[mark], axis=1).mean() / scale)
                oth = [j for j, x in enumerate(t) if x in home and x != mark]
                if oth:
                    do = v[oth] - np.array([home[t[j]] for j in oth])
                    dn = do / np.linalg.norm(do, axis=1, keepdims=True)
                    r["прочие: доля на оси"] = float(np.mean((dn @ ax) ** 2))
                    r["прочие: смещение"] = float(
                        np.linalg.norm(do, axis=1).mean() / scale)
                rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "marker_cell.csv"), index=False)
    pd.set_option("display.width", 250)
    print(D.groupby("условие").mean(numeric_only=True).round(3).T.to_string())


if __name__ == "__main__":
    main()
