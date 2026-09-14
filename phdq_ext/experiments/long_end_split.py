"""Откуда берётся расслоение длинного конца.

Крупная полоса падает, когда длинный конец MST перестаёт быть одной популяцией:
при подстановке чужих хапаксов CV верхних 20% рёбер растёт на 23%, а средняя
длина ребра не меняется вовсе (0.609 → 0.604). Что именно расслаивается, до сих
пор не разбиралось.

У подстановки есть свойство, которого нет у прочих условий: **известно, какие
слова заменены**. swap_hapax сохраняет число слов, так что исходный и
испорченный тексты выравниваются по словам, и каждый токен помечается как
подставленный или родной. Тогда длинный конец раскладывается на три вида рёбер:

    подст.–подст.   подст.–родной   родной–родной

Предсказание, вытекающее из §8.3 (неуместные одиночки сближаются между собой на
14%): рёбра между подставленными должны быть **короче** прочих длинных, то есть
внутри длинного конца появляется вторая, более короткая популяция. Это и есть
смесь, и она же — источник роста CV.

Дисперсия длин внутри длинного конца раскладывается на межгрупповую и
внутригрупповую, чтобы отделить «группы разъехались» от «внутри групп стало
разбросаннее».

Перемешивание идёт контролем: там заменённых слов нет, и расслоение, если оно
есть, должно иметь другой источник.
"""
import hashlib
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
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

BASE = os.path.join(HERE, "..")
N_TEXTS = 50
TOP = 20  # доля длинного конца, %
# longest_edge.py: самое длинное ребро текста в 86-96% случаев -- перемычка к
# стоку внимания, а её удлинение и даёт почти весь прирост разброса. DROP_SINK
# убирает стоковые токены из облака до выборки и показывает, что остаётся
DROP_SINK = os.environ.get("DROP_SINK", "0") == "1"


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def word_of_token(toks):
    """Номер слова для каждого токена: Ġ открывает слово."""
    w, out = -1, []
    for k, x in enumerate(toks):
        if k == 0 or x.startswith(("Ġ", "▁")):
            w += 1
        out.append(w)
    return np.array(out)


def _side(mst, n, drop):
    """Какая сторона дерева остаётся при разрезании данного ребра."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    m = np.ones(len(mst.data), bool)
    m[drop] = False
    g = coo_matrix((mst.data[m], (mst.row[m], mst.col[m])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab == lab[mst.row[drop]]


def analyse(text, key, tok, changed_words):
    e = cached(text, key)
    t_all = tok.tokenize(text)
    if e is None or len(t_all) != e.shape[0]:
        return None
    wid_all = word_of_token(t_all)
    keep = [j for j, x in enumerate(t_all) if x not in SKIP]
    if len(keep) < cfg.L_DEFAULT:
        return None
    e, toks, wid = e[keep], [t_all[j] for j in keep], wid_all[keep]
    if DROP_SINK:
        ok = np.linalg.norm(e, axis=1) >= 30.0
        if ok.sum() < cfg.L_DEFAULT:
            return None
        e, toks, wid = e[ok], [x for x, k in zip(toks, ok) if k], wid[ok]
    idx = np.random.default_rng(1000).choice(len(toks), size=cfg.L_DEFAULT,
                                             replace=False)
    v, t, w = e[idx], [toks[j] for j in idx], wid[idx]
    scale = pdist(v).mean()
    mst = minimum_spanning_tree(squareform(pdist(v))).tocoo()
    L = mst.data / scale
    cut = np.percentile(L, 100 - TOP)
    long_ = L >= cut
    a, b = mst.row[long_], mst.col[long_]
    ln = L[long_]

    cnt = Counter(t)
    single = np.array([cnt[x] == 1 for x in t])
    sub = np.array([wi in changed_words for wi in w]) if changed_words else \
        np.zeros(len(t), bool)

    # где именно расширяется длинный конец: перцентили длин внутри него и
    # положение самой границы отсечения
    qs = np.percentile(ln, [0, 10, 25, 50, 75, 90, 100])
    out = {"граница (p80 всех рёбер)": cut,
           "медиана всех рёбер": float(np.median(L)),
           "длинные p0": qs[0], "длинные p10": qs[1], "длинные p25": qs[2],
           "длинные p50": qs[3], "длинные p75": qs[4], "длинные p90": qs[5],
           "длинные p100": qs[6],
           "размах p100/p0": qs[6] / qs[0],
           "верх p100/p50": qs[6] / qs[3], "низ p50/p0": qs[3] / qs[0],
           "CV длинного конца": ln.std() / ln.mean(),
           "средняя длина длинных": ln.mean(),
           "подставленных в выборке, %": sub.mean() * 100}
    groups = {"подст.–подст.": sub[a] & sub[b],
              "подст.–родной": sub[a] ^ sub[b],
              "родной–родной": ~sub[a] & ~sub[b],
              "оба одиночки": single[a] & single[b],
              "один одиночка": single[a] ^ single[b],
              "без одиночек": ~single[a] & ~single[b]}
    for name, m in groups.items():
        out[f"{name}: доля"] = m.mean() * 100
        out[f"{name}: длина"] = ln[m].mean() if m.sum() >= 3 else np.nan
        out[f"{name}: CV"] = (ln[m].std() / ln[m].mean()
                              if m.sum() >= 3 else np.nan)

    # сколько прироста разброса держится на считанных самых длинных рёбрах
    srt = np.sort(ln)[::-1]
    for k in (0, 1, 2, 3, 5, 10):
        rest = srt[k:]
        out[f"CV без верхних {k}"] = rest.std() / rest.mean()
        out[f"дисперсия без верхних {k}"] = rest.var()

    # из чего сделан самый хвост: 5 самых длинных рёбер текста
    ordr = np.argsort(L)[::-1][:5]
    ea, eb = mst.row[ordr], mst.col[ordr]
    el = L[ordr]
    nrm = np.linalg.norm(v, axis=1)
    sink_m = nrm < 30.0
    out["хвост: длина"] = float(el.mean())
    out["хвост: подставленных концов, %"] = float(
        np.mean(np.concatenate([sub[ea], sub[eb]])) * 100)
    out["хвост: одиночек, %"] = float(
        np.mean(np.concatenate([single[ea], single[eb]])) * 100)
    out["хвост: стоков, %"] = float(
        np.mean(np.concatenate([sink_m[ea], sink_m[eb]])) * 100)
    out["хвост: разрыв норм"] = float(np.mean(np.abs(nrm[ea] - nrm[eb])))
    out["хвост: меньшая сторона"] = float(np.median(
        [min(int(_side(mst, len(v), int(i_)).sum()),
             len(v) - int(_side(mst, len(v), int(i_)).sum()))
         for i_ in ordr]))
    out["длинные: стоков, %"] = float(
        np.mean(np.concatenate([sink_m[a], sink_m[b]])) * 100)
    out["длинные: разрыв норм"] = float(np.mean(np.abs(nrm[a] - nrm[b])))

    # разложение дисперсии по разрезу «подставленные»
    for tag, keys in [("подст.", ["подст.–подст.", "подст.–родной",
                                  "родной–родной"]),
                      ("одиночки", ["оба одиночки", "один одиночка",
                                    "без одиночек"])]:
        ms = [groups[k] for k in keys]
        if min(m.sum() for m in ms) < 3:
            continue
        gm = np.array([ln[m].mean() for m in ms])
        gn = np.array([m.sum() for m in ms])
        between = float(np.average((gm - ln.mean()) ** 2, weights=gn))
        within = float(np.average([ln[m].var() for m in ms], weights=gn))
        out[f"{tag}: межгрупповая доля дисперсии"] = between / (between + within)
        out[f"{tag}: дисперсия всего"] = between + within
    return out


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        sw = perturb("hapax_swap_wide", s, seed=i)
        a, b = s.split(), sw.split()
        changed = {k for k, (x, y) in enumerate(zip(a, b)) if x != y} \
            if len(a) == len(b) else set()
        for label, txt, key, ch in [
                ("исходный", s, f"pg_identity_{i}", set()),
                ("чужие хапаксы", sw, f"pg_hapax_swap_wide_{i}", changed),
                ("перемешивание", perturb("shuffle_words", s, seed=i),
                 f"pg_shuffle_words_{i}", set())]:
            r = analyse(txt, key, tok, ch)
            if r:
                rows.append({"текст": i, "условие": label, **r})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1} текстов", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "long_end_split"
                          + ("_nosink" if DROP_SINK else "") + ".csv"),
             index=False)
    pd.set_option("display.width", 250)
    G = D.groupby("условие").mean(numeric_only=True)
    print("\n== состав длинного конца ==")
    cols = [c for c in D.columns if "подст." in c or "одиноч" in c]
    print(G[[c for c in cols if ": доля" in c or ": длина" in c]].round(3).T.to_string())
    print("\n== разложение дисперсии длин внутри длинного конца ==")
    print(G[[c for c in cols if "дисперси" in c]].round(4).T.to_string())
    print("\n== общее ==")
    print(G[["CV длинного конца", "средняя длина длинных",
             "подставленных в выборке, %"]].round(3).to_string())


if __name__ == "__main__":
    main()
