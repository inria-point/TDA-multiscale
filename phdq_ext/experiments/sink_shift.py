"""Сток сдвигается в испорченных текстах — или всё сдвигается относительно него?

longest_edge.py: самое длинное ребро текста в 86-96% случаев — перемычка от
группы стока (норма 26) к остальному облаку (норма 34-37), и при порче она
удлиняется. long_end_split.py с DROP_SINK: у перемешивания весь рост разброса
длинного конца этим и создавался, без стоков он исчезает.

Вопрос: что именно движется. Есть готовый ответ-кандидат. Ось «контекст ничего
не диктует» почти ортогональна направлению стока (косинус +0.022, §8.5), а
стоковые токены — это `the`, точка, `to`, запятая, у которых своего контекста и
так нет. Если при порче сносит обычные токены и не сносит стоковые, то
расстояние между группами растёт само собой, по теореме Пифагора, без всякого
движения стока.

Меряется на текст и условие:

  норма стока, норма прочих      двигается ли группа по радиусу
  косинус центра стока с общим   поворачивается ли она
  проекция на ось порчи          отдельно для стоковых и для прочих токенов
  расстояние сток - облако       и предсказание Пифагора из сдвига прочих
"""
import hashlib
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import wilcoxon
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
N_TEXTS = 50
CONDS = [("исходный", "identity"), ("чужие хапаксы", "hapax_swap_wide"),
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


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    home = home_vectors(hum[:HOME_TEXTS], tok)
    Z = np.load(os.path.join(BASE, "results", "fit_pmi_axis.npz"))
    u = Z["перемешивание"] / np.linalg.norm(Z["перемешивание"])
    s_dir = Z["sink"] / np.linalg.norm(Z["sink"])

    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in CONDS:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            e = cached(txt, f"pg_{name}_{i}")
            t_all = tok.tokenize(txt)
            if e is None or len(t_all) != e.shape[0]:
                continue
            keep = [j for j, x in enumerate(t_all) if x not in SKIP]
            if len(keep) < cfg.L_DEFAULT:
                continue
            e, toks = e[keep], [t_all[j] for j in keep]
            idx = np.random.default_rng(1000).choice(len(toks),
                                                     size=cfg.L_DEFAULT,
                                                     replace=False)
            v, t = e[idx], [toks[j] for j in idx]
            nrm = np.linalg.norm(v, axis=1)
            sk = nrm < NORM_CUT
            if sk.sum() < 3:
                continue
            scale = pdist(v[~sk]).mean()
            cs, co = v[sk].mean(0), v[~sk].mean(0)
            r = {"текст": i, "условие": label, "стоков": int(sk.sum()),
                 "норма стока": nrm[sk].mean(), "норма прочих": nrm[~sk].mean(),
                 "косинус стока с общим": float(cs @ s_dir
                                                / np.linalg.norm(cs)),
                 "разброс внутри стока": float(
                     np.linalg.norm(v[sk] - cs, axis=1).mean() / scale),
                 "сток → облако": float(np.linalg.norm(cs - co) / scale),
                 "масштаб": scale}
            for tag, m in [("сток", sk), ("прочие", ~sk)]:
                ix = [j for j in np.where(m)[0] if t[j] in home]
                if len(ix) < 3:
                    continue
                D = np.array([v[j] - home[t[j]] for j in ix])
                r[f"{tag}: проекция на ось"] = float((D @ u).mean() / scale)
                r[f"{tag}: смещение"] = float(
                    np.linalg.norm(D, axis=1).mean() / scale)
            rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "sink_shift.csv"), index=False)
    pd.set_option("display.width", 220)
    G = D.groupby("условие").mean(numeric_only=True)
    T = G.T
    for c in ["чужие хапаксы", "перемешивание"]:
        T[f"Δ {c}, %"] = (T[c] / T["исходный"] - 1) * 100
    print(T.round(4).to_string())

    print("\nПарные тесты против исходного:")
    piv = D.pivot(index="текст", columns="условие")
    for c in ["норма стока", "косинус стока с общим", "сток → облако",
              "сток: проекция на ось", "прочие: проекция на ось"]:
        for cond in ["чужие хапаксы", "перемешивание"]:
            a, b = piv[(c, "исходный")], piv[(c, cond)]
            m = a.notna() & b.notna()
            print(f"  {c:26s} {cond:15s} {a[m].mean():+.4f} → {b[m].mean():+.4f}"
                  f"   p={wilcoxon(a[m], b[m]).pvalue:.1e}")


if __name__ == "__main__":
    main()
