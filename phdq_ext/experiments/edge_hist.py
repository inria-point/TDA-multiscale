"""Гистограммы длин рёбер MST с разбивкой по типам рёбер.

Пять условий на одной сетке: неиспорченный текст, перемешивание слов,
подстановка чужих однократных слов — и, с другой стороны, художественный
пересказ, единственное условие в наборе, где крупная полоса заметно **растёт**
(+13.0%).

Типы рёбер сведены к пяти, чтобы читались:

  сток              хотя бы один конец в группе стока внимания (норма < 30)
  повтор токена     оба конца — вхождения одного и того же токена
  служ./пункт.      два разных служебных слова или знака
  два смысловых     два разных содержательных слова
  прочее            смешанные классы, подслова, куски одного слова

Длины — в долях средней парной дистанции облака, поэтому панели сравнимы между
собой. Высота — число рёбер на текст. Серый контур на панелях с порчей — полное
распределение неиспорченного текста той же выборки.
"""
import hashlib
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from coling_data import human_texts
from edge_taxonomy import SKIP, corpus_ranks, token_class
from embedder import CACHE_DIR, MODEL_NAME
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
N_TEXTS = 50
NORM_CUT = 30.0
# ABS=1 -- всё в абсолютных единицах эмбеддинга, без деления на среднюю парную
# дистанцию. Нормированный вид отвечает на вопрос «что видит оценка
# размерности» (она инвариантна к общему масштабу), абсолютный -- на вопрос
# «что произошло в самом пространстве». Масштаб у условий разный на 4-6%, так
# что это два разных ответа.
ABS = os.environ.get("ABS", "0") == "1"
CONTENT = {"смысл-част", "смысл-редк"}
FUNC = {"служ", "пункт"}

TYPES = ["повтор токена", "служ./пункт.", "прочее", "два смысловых", "сток"]
# слоты валидированной палитры, в фиксированном порядке
COLOR = {"повтор токена": "#2a78d6", "служ./пункт.": "#eb6834",
         "прочее": "#1baf7a", "два смысловых": "#eda100", "сток": "#e87ba4"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d2"


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def edges(text, key, tok, ranks):
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
    nrm = np.linalg.norm(v, axis=1)
    sink = nrm < NORM_CUT
    cls = [token_class(x, ranks) for x in t]
    # масштаб без стоков — как везде в проекте: сток стоит на радиусе 26 при
    # 37 у прочих и тянул бы среднюю парную дистанцию вниз
    scale = pdist(v[~sink]).mean()
    mst = minimum_spanning_tree(squareform(pdist(v))).tocoo()
    out = []
    for a, b, w in zip(mst.row, mst.col,
                       mst.data if ABS else mst.data / scale):
        if sink[a] or sink[b]:
            k = "сток"
        elif t[a] == t[b]:
            k = "повтор токена"
        elif cls[a] in FUNC and cls[b] in FUNC:
            k = "служ./пункт."
        elif cls[a] in CONTENT and cls[b] in CONTENT:
            k = "два смысловых"
        else:
            k = "прочее"
        out.append((w, k))
    return out, scale


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ranks = corpus_ranks(hum, type("E", (), {"tokenizer": tok})())

    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in [("исходный", "identity"),
                            ("перемешивание", "shuffle_words"),
                            ("чужие хапаксы", "hapax_swap_wide")]:
            txt = s if name == "identity" else perturb(name, s, seed=i)
            r = edges(txt, f"pg_{name}_{i}", tok, ranks)
            if r:
                rows += [(label, i, w, k, r[1]) for w, k in r[0]]
    import json
    src = dict(human_texts(120, min_words=280))
    with open(os.path.join(BASE, "results", "coling_temps.json")) as f:
        lit = json.load(f)["style_literary_T1.3"]
    for n, (k, text) in enumerate(src.items()):
        if n >= N_TEXTS:
            break
        for label, txt in [("исходный (LLM-набор)", text),
                           ("художественный стиль", lit.get(k))]:
            if txt is None:
                continue
            r = edges(txt, f"pgl_{label.replace(' (LLM-набор)', '')}_{k[-8:]}"
                      if label.startswith("исходный") else
                      f"pgl_художественный стиль_{k[-8:]}", tok, ranks)
            if r:
                rows += [(label, n, w, kk, r[1]) for w, kk in r[0]]
    D = pd.DataFrame(rows, columns=["условие", "текст", "длина", "тип",
                                    "масштаб"])
    D.to_csv(os.path.join(BASE, "results",
                          "edge_hist_abs.csv" if ABS else "edge_hist.csv"),
             index=False)
    print(D.groupby("условие").size().to_string(), flush=True)

    panels = [("исходный", None), ("перемешивание", "исходный"),
              ("чужие хапаксы", "исходный"),
              ("исходный (LLM-набор)", None),
              ("художественный стиль", "исходный (LLM-набор)")]
    bins = np.linspace(0, 42, 70) if ABS else np.linspace(0, 1.6, 65)
    ntex = {c: max(1, len(D[D["условие"] == c]) / (cfg.L_DEFAULT - 1))
            for c in D["условие"].unique()}

    # цвета разностных кривых — отдельная валидированная четвёрка
    DIFF_C = {"перемешивание": "#2a78d6", "чужие хапаксы": "#eb6834",
              "художественный стиль": "#4a3aa7"}
    XMAX = 42 if ABS else 1.25
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.6), sharex=True)
    axes = axes.ravel()
    for a in axes[1:5]:                      # общая ось Y только у гистограмм
        a.sharey(axes[0])
    for ax, (cond, ref) in zip(axes, panels):
        g = D[D["условие"] == cond]
        bottom = np.zeros(len(bins) - 1)
        for k in TYPES:
            h, _ = np.histogram(g[g["тип"] == k]["длина"], bins=bins)
            h = h / ntex[cond]
            ax.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom,
                   align="edge", color=COLOR[k], linewidth=0, label=k)
            bottom += h
        if ref is not None:
            hr, _ = np.histogram(D[D["условие"] == ref]["длина"], bins=bins)
            ax.step(bins[:-1], hr / ntex[ref], where="post", color=INK2,
                    linewidth=1.2, alpha=0.8)
        mx = g.groupby("текст")["длина"].max()
        med = np.median(g["длина"])
        ax.axvline(med, color=INK, linewidth=1, linestyle=(0, (4, 3)))
        fmt = "{:.1f}" if ABS else "{:.2f}"
        ax.annotate("медиана " + fmt.format(med), xy=(med, 1),
                    xycoords=("data", "axes fraction"), xytext=(4, -4),
                    textcoords="offset points", color=INK2, fontsize=8.5,
                    va="top")
        if ABS:      # средняя парная дистанция — только в абсолютном виде
            sc = g["масштаб"].mean()
            ax.axvline(sc, color=INK2, linewidth=1, linestyle=(0, (1, 2)))
            ax.annotate(f"средняя парная {sc:.1f}", xy=(sc, 1),
                        xycoords=("data", "axes fraction"), xytext=(4, -18),
                        textcoords="offset points", color=INK2, fontsize=8.5,
                        va="top")
        # самое длинное ребро текста, в среднем по текстам: почти всегда
        # перемычка к стоку (longest_edge.py), поэтому помечено отдельно
        ax.plot([mx.mean()], [0], marker="v", markersize=7,
                color=COLOR["сток"], clip_on=False, zorder=5)
        ax.annotate(("самое длинное " + ("{:.1f}" if ABS else "{:.2f}")
                     ).format(mx.mean()), xy=(mx.mean(), 0),
                    xytext=(-4, 16), textcoords="offset points",
                    color=INK2, fontsize=8.5, ha="right")
        # длины нормированы на среднюю парную дистанцию, а она сама зависит от
        # условия (порча сжимает облако на 4-6%), поэтому масштаб подписан
        ax.set_title(cond if ABS else
                     f"{cond}   ·   масштаб {g['масштаб'].mean():.1f}",
                     fontsize=11.5, color=INK, loc="left")
        ax.set_xlim(0, XMAX)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
    # шестая клетка: разность с неиспорченным, все условия на одной оси
    ax = axes[5]
    ax.set_visible(True)
    for cond, ref in panels[1:]:
        if ref is None:
            continue
        h, _ = np.histogram(D[D["условие"] == cond]["длина"], bins=bins)
        hr, _ = np.histogram(D[D["условие"] == ref]["длина"], bins=bins)
        d = h / ntex[cond] - hr / ntex[ref]
        c = DIFF_C[cond]
        ax.plot(bins[:-1] + np.diff(bins) / 2, d, color=c, linewidth=2)
        j = int(np.argmax(np.abs(d)))
        ax.annotate(cond, xy=(bins[j], d[j]), xytext=(6, 6 if d[j] > 0 else -14),
                    textcoords="offset points", color=c, fontsize=9.5)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylabel("разность, рёбер на текст", fontsize=9, color=INK2)
    # разность считается по нормированным длинам, а масштаб у условий разный,
    # так что это ответ на вопрос «что видит оценка размерности» (она
    # инвариантна к общему масштабу), а не «что произошло в самом пространстве»
    ax.set_title("разность с неиспорченным" if ABS else
                 "разность с неиспорченным,\nв нормированных длинах",
                 fontsize=11.5, color=INK, loc="left")
    if not ABS:
        ax.annotate("масштаб у условий разный (см. заголовки),\n"
                    "поэтому это то, что видит оценка размерности,\n"
                    "а не абсолютное изменение длин",
                    xy=(0.02, 0.03), xycoords="axes fraction", fontsize=8.5,
                    color=INK2, va="bottom")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    xlab = ("длина ребра, абсолютная" if ABS
            else "длина ребра, в долях средней парной дистанции")
    for ax in axes[3:6]:
        ax.set_xlabel(xlab, fontsize=9, color=INK2)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("рёбер на текст", fontsize=9, color=INK2)
    fig.legend(handles=[Patch(facecolor=COLOR[k], label=k) for k in TYPES]
               + [plt.Line2D([], [], color=INK2, lw=1.2,
                             label="контур неиспорченного")],
               loc="lower center", ncol=6, frameon=False, fontsize=10,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Длины рёбер MST по типам, "
                 + ("абсолютные единицы" if ABS
                    else "в долях средней парной дистанции")
                 + ": порча и художественный пересказ",
                 fontsize=13.5, color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out = os.path.join(BASE, "figures",
                       "edge_hist_abs.png" if ABS else "edge_hist.png")
    fig.savefig(out, dpi=200, facecolor="#fcfcfb")
    print("→", out)


if __name__ == "__main__":
    main()
