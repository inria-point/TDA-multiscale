"""Куда именно сносит неуместное слово: не в сторону ли стока внимания?

fit_pmi_direction.py показал, что смещения неуместных токенов согласованы между
собой (+52% при подстановке, +139% при перемешивании) и что такие токены
сближаются друг с другом, при этом не приближаясь к центру облака. Общее
направление есть, а что это за направление — не спрашивалось.

У облака ModernBERT есть одно выделенное направление: сток внимания. Около 2%
токенов (`the`, точка, `to`, запятая) стоят почти в одной точке с нормой 26.3
против 36-40 у всех остальных, взаимный косинус внутри группы 0.987, и
направление одно и то же во всех текстах. Это свойство энкодера, а не текста.
Оно стоит сбоку от облака, а не в его середине, и потому объяснило бы сразу и
согласованность смещений, и сближение неуместных токенов, и то, что до центра
облака им при этом не ближе.

Здесь три замера:

  косинус смещения со стоком   по терцилям PMI и по условиям
  расстояние до центра стока   то же
  общее направление без стока  контроль: сами стоковые токены из выборки
                               не исключались, и часть согласованности могли
                               давать они
  одно ли направление          косинус между средними направлениями сноса
                               разных текстов и разных условий: своё оно у
                               каждого текста или общее
  разложение сноса             какая доля направления приходится на общую ось,
                               какая на направление своего текста и какая
                               остаётся индивидуальной

Направление стока оценивается по норме: она у него 26.3 при 36-40 у прочих, так
что порог 30 отделяет группу без всякого обучения. Эмбеддинги берутся из кэша,
PMI из results/fit_pmi.csv, энкодер не запускается.
"""
import hashlib
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import SKIP
from embedder import CACHE_DIR, MODEL_NAME
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
HOME_TEXTS, HOME_MIN, NORM_CUT = 200, 5, 30.0
CONDS = [("исходный", None), ("чужие хапаксы", "hapax_swap_wide"),
         ("перемешивание", "shuffle_words")]


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


def sink_axis(texts, tok):
    """Направление стока: среднее по токенам с нормой ниже порога."""
    acc, n = None, 0
    for i, s in enumerate(texts):
        e = cached(s, f"home_{i}")
        t = tok.tokenize(s)
        if e is None or len(t) != e.shape[0]:
            continue
        m = np.linalg.norm(e, axis=1) < NORM_CUT
        if m.sum() == 0:
            continue
        acc = e[m].sum(0) if acc is None else acc + e[m].sum(0)
        n += int(m.sum())
    return acc / n, n


def mean_cos(M, u):
    U = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return float(np.mean(U @ (u / np.linalg.norm(u))))


def pair_cos(M, types):
    if len(M) < 3:
        return np.nan
    U = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    C = U @ U.T
    same = np.array(types)[:, None] == np.array(types)[None, :]
    C[same] = np.nan
    return float(np.nanmean(C))


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    home = home_vectors(hum[:HOME_TEXTS], tok)
    centre, n_sink = sink_axis(hum[:HOME_TEXTS], tok)
    print(f"домашних векторов {len(home)}; сток оценён по {n_sink} токенам, "
          f"его норма {np.linalg.norm(centre):.1f}", flush=True)

    P = pd.read_csv(os.path.join(BASE, "results", "fit_pmi.csv"))
    rows, units = [], []
    for i, s in enumerate(hum[:P["текст"].max() + 1]):
        for label, name in CONDS:
            txt = s if name is None else perturb(name, s, seed=i)
            e = cached(txt, f"pg_{name or 'identity'}_{i}")
            t_all = tok.tokenize(txt)
            if e is None or len(t_all) != e.shape[0]:
                continue
            keep = [j for j, x in enumerate(t_all) if x not in SKIP]
            if len(keep) < cfg.L_DEFAULT:
                continue
            e, toks = e[keep], [t_all[j] for j in keep]
            rng = np.random.default_rng(1000)
            idx = rng.choice(len(toks), size=cfg.L_DEFAULT, replace=False)
            v, t = e[idx], [toks[j] for j in idx]
            g = P[(P["текст"] == i) & (P["условие"] == label)]
            if len(g) != len(t) or list(g["токен"]) != t:
                continue
            pmi = g["PMI"].to_numpy()
            scale = pdist(v).mean()
            is_sink = np.linalg.norm(v, axis=1) < NORM_CUT

            have = np.array([j for j, x in enumerate(t)
                             if x in home and not is_sink[j]])
            if len(have) < 40:
                continue
            D = np.array([v[j] - home[t[j]] for j in have])
            pm, types = pmi[have], [t[j] for j in have]
            lo, hi = np.percentile(pm, [33, 67])
            d_sink = np.linalg.norm(v[have] - centre, axis=1) / scale
            # разложение направления сноса: общая ось (оценивается вторым
            # проходом, см. AXIS), направление своего текста, остаток
            nrm_d = np.linalg.norm(D, axis=1)
            Un = D / nrm_d[:, None]
            cen_t = v.mean(0)
            cen_t = cen_t / np.linalg.norm(cen_t)
            r = {"текст": i, "условие": label, "стоков, %": is_sink.mean() * 100,
                 "концентрация": float(np.linalg.norm(D.mean(0)) / nrm_d.mean()),
                 "доля на центре текста": float(np.mean((Un @ cen_t) ** 2)),
                 "направление без стоков": pair_cos(D, types),
                 "косинус со стоком": mean_cos(D, centre),
                 "до стока": float(d_sink.mean())}
            for tag, m in [("низкий PMI", pm <= lo), ("высокий PMI", pm >= hi)]:
                r[f"{tag}: косинус со стоком"] = mean_cos(D[m], centre)
                r[f"{tag}: до стока"] = float(d_sink[m].mean())
            # само направление сноса: сохраняем, чтобы сравнить между текстами
            for tag, m in [("низкий PMI", pm <= lo), ("все", pm == pm)]:
                mu = D[m].mean(0)
                r[f"вектор {tag}"] = mu / (np.linalg.norm(mu) + 1e-9)
            r["центр текста"] = v.mean(0) / np.linalg.norm(v.mean(0))
            rows.append(r)
            units.append(Un)
    D = pd.DataFrame(rows)
    # одно ли это направление на все тексты
    print("\nСогласие направления сноса МЕЖДУ текстами (косинус):")
    for cond, g in D.groupby("условие"):
        out = [f"{cond:16s}"]
        for tag in ["низкий PMI", "все"]:
            M = np.vstack(g[f"вектор {tag}"].to_numpy())
            C = M @ M.T
            np.fill_diagonal(C, np.nan)
            out.append(f"{tag}: {np.nanmean(C):+.3f}")
        # и со средним вектором своего текста
        cc = np.vstack(g["центр текста"].to_numpy())
        mm = np.vstack(g["вектор низкий PMI"].to_numpy())
        out.append(f"с центром своего текста: {np.mean(np.sum(cc * mm, 1)):+.3f}")
        print("   " + "   ".join(out))
    # одно ли направление у двух несвязанных порч
    glob = {}
    for cond, g in D.groupby("условие"):
        M = np.vstack(g["вектор низкий PMI"].to_numpy()).mean(0)
        glob[cond] = M / np.linalg.norm(M)
    names = list(glob)
    print("\nКосинусы между глобальными направлениями сноса:")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            print(f"   {names[a]:16s} x {names[b]:16s} "
                  f"{glob[names[a]] @ glob[names[b]]:+.3f}")
    for cond in names:
        print(f"   {cond:16s} x сток внимания   "
              f"{glob[cond] @ (centre / np.linalg.norm(centre)):+.3f}")
    np.savez(os.path.join(BASE, "results", "fit_pmi_axis.npz"),
             sink=centre, **{k: v for k, v in glob.items()})

    # доля направления на общей оси. Считается по отдельным токенам, а не по
    # среднему вектору текста, поэтому единичные сносы держались до сих пор.
    # Случайное направление в 768 измерениях дало бы 1/768 = 0.0013
    ax = glob["перемешивание"]
    D["доля на общей оси"] = [float(np.mean((U @ ax) ** 2)) for U in units]
    print("\nРазложение направления сноса (случайное дало бы 0.0013):")
    print(D.groupby("условие")[["концентрация", "доля на общей оси",
                                "доля на центре текста"]].mean().round(4)
          .to_string())
    D = D.drop(columns=[c for c in D.columns if c.startswith("вектор ")]
               + ["центр текста"])
    D.to_csv(os.path.join(BASE, "results", "fit_pmi_sink.csv"), index=False)
    print(f"\n{len(D)} пар текст x условие\n")
    print(D.groupby("условие").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
