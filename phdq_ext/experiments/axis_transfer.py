"""Переносится ли ось на порчи, которых она не видела.

axis_projection.py показал, что средняя проекция сноса смысловых слов на ось
«контекст ничего не диктует» отделяет испорченный текст от целого на 40 текстах
из 40. Но ось построена по перемешиванию, и проверялась на перемешивании и
подстановке хапаксов. Вопрос, годится ли она как общая мера повреждения,
решается переносом: те же проекции на всех условиях обоих прогонов, без
единого нового замера PMI.

Ось берётся из fit_pmi_axis.npz (среднее смещений нижней трети по PMI на
перемешанных текстах). Считается только проекция, поэтому нужен лишь кэш
эмбеддингов и домашние векторы.
"""
import hashlib
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from transformers import AutoTokenizer

import config as cfg
from edge_taxonomy import SKIP, corpus_ranks, token_class
from embedder import CACHE_DIR, MODEL_NAME
from pert_geometry import PERTS as MECH
from pert_geometry_llm import PERTS as LLM
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
HOME_TEXTS, HOME_MIN, NORM_CUT = 200, 5, 30.0
CONTENT = {"смысл-част", "смысл-редк"}
N_TEXTS = 50


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


def project(text, key, tok, home, ranks, u):
    e = cached(text, key)
    t_all = tok.tokenize(text)
    if e is None or len(t_all) != e.shape[0]:
        return None
    keep = [j for j, x in enumerate(t_all) if x not in SKIP]
    if len(keep) < cfg.L_DEFAULT:
        return None
    e, toks = e[keep], [t_all[j] for j in keep]
    idx = np.random.default_rng(1000).choice(len(toks), size=cfg.L_DEFAULT,
                                             replace=False)
    v, t = e[idx], [toks[j] for j in idx]
    sink = np.linalg.norm(v, axis=1) < NORM_CUT
    ix = [j for j, x in enumerate(t) if x in home and not sink[j]]
    if len(ix) < 40:
        return None
    D = np.array([v[j] - home[t[j]] for j in ix])
    proj = (D @ u) / pdist(v).mean()
    cm = np.array([token_class(t[j], ranks) in CONTENT for j in ix])
    if cm.sum() < 10:
        return None
    return proj, cm


def row(i, label, proj, cm):
    return {"текст": i, "условие": label, "проекция": float(proj[cm].mean()),
            "доля > 0, %": float((proj[cm] > 0).mean() * 100)}


def run(tag, texts, keys, tok, home, ranks, u):
    rows = []
    for tid, txt_of in texts:
        for label, key in keys:
            txt = txt_of(label)
            if txt is None:
                continue
            r = project(txt, key(tid), tok, home, ranks, u)
            if r is None:
                continue
            rows.append({"текст": tid, "условие": label,
                         "проекция": r[0], "доля > 0, %": r[1]})
    return pd.DataFrame(rows)


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    home = home_vectors(hum[:HOME_TEXTS], tok)
    ranks = corpus_ranks(hum, type("E", (), {"tokenizer": tok})())
    u = np.load(os.path.join(BASE, "results", "fit_pmi_axis.npz"))["перемешивание"]
    u = u / np.linalg.norm(u)
    # нулевая модель: случайное направление. Если дело просто в том, что при
    # порче снос длиннее, случайная ось даст то же самое
    rg = np.random.default_rng(0)
    u_rand = rg.normal(size=u.shape)
    u_rand /= np.linalg.norm(u_rand)

    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in MECH:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            r = project(txt, f"pg_{name}_{i}", tok, home, ranks, u)
            if r:
                rows.append(row(i, label, *r))
            rn = project(txt, f"pg_{name}_{i}", tok, home, ranks, u_rand)
            if rn:
                rows.append(row(i, label + " [случайная ось]", *rn))
        if (i + 1) % 10 == 0:
            print(f"  механические: {i + 1} текстов", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "axis_transfer.csv"), index=False)

    # LLM-переписывания: тексты другие, поэтому отдельным блоком
    import json
    src = dict(__import__("coling_data").human_texts(120, min_words=280))
    store = {}
    for label, key, fname in LLM:
        with open(os.path.join(BASE, "results", fname)) as f:
            store[label] = json.load(f)[key]
    lrows = []
    for n, (k, text) in enumerate(src.items()):
        if n >= N_TEXTS:
            break
        for label, txt in [("исходный", text)] + [
                (lab, store[lab].get(k)) for lab, _, _ in LLM]:
            if txt is None:
                continue
            r = project(txt, f"pgl_{label}_{k[-8:]}", tok, home, ranks, u)
            if r:
                lrows.append(row(k, label, *r))
    L = pd.DataFrame(lrows)
    L.to_csv(os.path.join(BASE, "results", "axis_transfer_llm.csv"),
             index=False)
    if len(L):
        lb = L[L["условие"] == "исходный"].set_index("текст")["проекция"]
        lo = []
        for cond, g in L.groupby("условие"):
            g = g.set_index("текст")["проекция"]
            ix = lb.index.intersection(g.index)
            lo.append({"условие": cond, "проекция": g[ix].mean(),
                       "выше исходного": f"{int((g[ix] > lb[ix]).sum())}/{len(ix)}",
                       "доля > 0, %": L[L["условие"] == cond]["доля > 0, %"].mean()})
        print("\nLLM-переписывания, та же ось:\n")
        print(pd.DataFrame(lo).set_index("условие").sort_values("проекция")
              .round(4).to_string())

    base = D[D["условие"] == "исходный"].set_index("текст")["проекция"]
    out = []
    for cond, g in D.groupby("условие"):
        g = g.set_index("текст")["проекция"]
        ix = base.index.intersection(g.index)
        p = (wilcoxon(base[ix], g[ix]).pvalue if cond != "исходный"
             else np.nan)
        out.append({"условие": cond, "проекция": g[ix].mean(),
                    "сдвиг": g[ix].mean() - base[ix].mean(),
                    "текстов выше исходного": f"{int((g[ix] > base[ix]).sum())}/{len(ix)}",
                    "p": p,
                    "доля > 0, %": D[D["условие"] == cond]["доля > 0, %"].mean()})
    R = pd.DataFrame(out).set_index("условие").sort_values("проекция")
    print("\nПроекция сноса смысловых слов на ось «контекст ничего не диктует»")
    print("(ось построена ТОЛЬКО по перемешиванию; всё прочее — перенос)\n")
    print(R.round(4).to_string())


if __name__ == "__main__":
    main()
