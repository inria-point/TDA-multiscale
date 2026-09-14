"""Проекция каждого токена на ось определённости, поштучно.

Ось построена по сдвигу ТОКЕНОВ (fit_pmi_sink.py): d = v − дом(тип), среднее по
нижней трети по PMI в перемешанных текстах. Черешок — разность двух токенов, и
при вычитании общая компонента сокращается, поэтому черешки оказались почти
ортогональны оси и дали мало. Естественная единица — сам токен.

Здесь выгружается по токену: проекция его сноса на ось, однократность, класс,
норма. Любой порог и любой разрез считаются потом без пересчёта.

Гипотеза, ради которой это делается (сформулирована пользователем): крупная
полоса растёт только при росте разнообразия ОСМЫСЛЕННЫХ токенов, а рост числа
бессмысленных её роняет. «Бессмысленность» здесь операционализуется проекцией
на ось.
"""
import hashlib
import json
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
from coling_data import human_texts
from edge_taxonomy import SKIP, corpus_ranks, token_class
from embedder import CACHE_DIR, MODEL_NAME
from pert_geometry import PERTS as MECH
from pert_geometry_llm import PERTS as LLM
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
N_TEXTS, NORM_CUT, HOME_MIN, HOME_TEXTS, SEEDS = 50, 30.0, 5, 200, 3
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


def rows_for(text, key, tok, home, ranks, u):
    e = cached(text, key)
    t_all = tok.tokenize(text)
    if e is None or len(t_all) != e.shape[0]:
        return []
    keep = [j for j, x in enumerate(t_all) if x not in SKIP]
    e, toks = e[keep], [t_all[j] for j in keep]
    if e.shape[0] < cfg.L_DEFAULT:
        return []
    out = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        idx = rng.choice(e.shape[0], size=cfg.L_DEFAULT, replace=False)
        v, t = e[idx], [toks[j] for j in idx]
        nrm = np.linalg.norm(v, axis=1)
        sink = nrm < NORM_CUT
        scale = pdist(v[~sink]).mean() if (~sink).sum() > 2 else pdist(v).mean()
        cnt = Counter(t)
        for j, x in enumerate(t):
            if sink[j] or x not in home:
                continue
            out.append((seed, x, token_class(x, ranks),
                        int(token_class(x, ranks) in CONTENT),
                        int(cnt[x] == 1),
                        float((v[j] - home[x]) @ u) / scale))
    return out


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    home = home_vectors(hum[:HOME_TEXTS], tok)
    ranks = corpus_ranks(hum, type("E", (), {"tokenizer": tok})())
    u = np.load(os.path.join(BASE, "results",
                             "fit_pmi_axis.npz"))["перемешивание"]
    u = u / np.linalg.norm(u)

    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in MECH:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            for r in rows_for(txt, f"pg_{name}_{i}", tok, home, ranks, u):
                rows.append(("механические", str(i), label) + r)
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
            for r in rows_for(txt, f"pgl_{label}_{k[-8:]}", tok, home, ranks, u):
                rows.append(("LLM", k, label) + r)
    D = pd.DataFrame(rows, columns=["набор", "текст", "условие", "зерно",
                                    "токен", "класс", "смысловой",
                                    "одиночка", "проекция"])
    D.to_csv(os.path.join(BASE, "results", "token_projection.csv"),
             index=False)
    print(f"\n{len(D)} токенов, {D['условие'].nunique()} условий")
    q = D[D["условие"] == "исходный"]["проекция"]
    for p in (50, 75, 90, 95, 99):
        print(f"  исходный текст, {p}-й перцентиль проекции: "
              f"{np.percentile(q, p):+.4f}")


if __name__ == "__main__":
    main()
