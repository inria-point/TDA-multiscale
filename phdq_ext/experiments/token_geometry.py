"""The geometry the MST is built on, before any edge is chosen.

edge_taxonomy.py reads the MST: it says which pairs end up close. That is a
statement about the extremes, because an MST edge is by definition the shortest
link available. This asks the prior question -- how the cloud is arranged --
over *all* pairs inside a text, so that what the MST picks can be compared
against what was there to pick from.

Everything is on unnormalised hidden states, the same vectors qPHD builds its
distance matrix from. That choice is not neutral, so the two ingredients of a
distance are reported apart:

  norm per class      is a punctuation token a short vector or a long one?
  cosine per pair     and how do the directions sit relative to each other?

The two questions the report needs answered here:

  Does token identity beat token class? Two occurrences of one word against two
  different words of the same class -- if the first is far closer, then the
  short scale is measuring repetition and not lexical similarity.

  Is there a frequency gradient in the norms? The coarse band answers to rare
  tokens, and the vocabulary matrix is itself organised by frequency, so a norm
  that grows with rarity would explain part of the band without any appeal to
  text structure at all.
"""
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import SKIP, corpus_ranks, token_class
from embedder import Embedder
from sink_cluster import COS_CUT, NORM_CUT, sink_direction

BASE = os.path.join(HERE, "..")
N_TEXTS = int(os.environ.get("N_TEXTS", 120))
N_TSNE = int(os.environ.get("N_TSNE", 4000))
# "сток" is not a lexical class but a geometric one: the collapsed low-norm
# group documented in sink_cluster.py. It is pulled out because leaving it
# inside служ/пункт mixes a fixed artefact of the encoder into the statistics
# of two real classes -- it is what gives punctuation its 3.6 norm spread.
CLASSES = ["пункт", "служ", "смысл-част", "смысл-редк", "подслово", "сток"]
CCOL = {"пункт": "#9e9ac8", "служ": "#6baed6", "смысл-част": "#e6550d",
        "смысл-редк": "#08306b", "подслово": "#74c476", "сток": "#000000"}
# hapax as a colour of its own: it is the axis that predicts hanging on a long
# stalk (leaf_class.py), and the class colouring does not show it at all
HCOL = {"hapax": "#d7191c", "повторяется": "#9ecae1", "сток": "#000000"}


def sample_text(text, emb, ranks, u_sink, L=cfg.L_DEFAULT, seed=0):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v = e[idx]
    t = [toks[i] for i in idx]
    nrm = np.linalg.norm(v, axis=1)
    sink = ((v @ u_sink) / nrm > COS_CUT) & (nrm < NORM_CUT)
    cls = ["сток" if s_ else token_class(x, ranks)
           for x, s_ in zip(t, sink)]
    full = Counter(toks[i] for i in keep)
    hap = ["сток" if c == "сток" else
           ("hapax" if full[x] == 1 else "повторяется")
           for x, c in zip(t, cls)]
    return v, t, cls, hap


def pair_stats(v, t, cls):
    """All within-text pairs: cosine and distance, tagged by class pair and
    by whether the two ends are literally the same token."""
    n = len(t)
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    cos = u @ u.T
    dist = np.linalg.norm(v[:, None] - v[None], axis=-1)
    i, j = np.triu_indices(n, 1)
    c = np.asarray(cls)
    tt = np.asarray(t)
    return pd.DataFrame({
        "ca": c[i], "cb": c[j],
        "same_token": tt[i] == tt[j],
        "cos": cos[i, j], "dist": dist[i, j],
    })


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    print(f"частотный словарь по {len(hum)} текстам", flush=True)
    ranks = corpus_ranks(hum, emb)
    u_sink, n_sink = sink_direction(hum[300:340], emb)
    print(f"направление стока по {n_sink} токенам 40 других текстов",
          flush=True)

    tok_rows, pair_rows, tsne_v, tsne_meta = [], [], [], []
    rng = np.random.default_rng(0)
    for i, s in enumerate(hum[:N_TEXTS]):
        r = sample_text(s, emb, ranks, u_sink, seed=i)
        if r is None:
            continue
        v, t, cls, hap = r
        nrm = np.linalg.norm(v, axis=1)
        centre = v.mean(0)
        cc = ((v @ centre) / (nrm * np.linalg.norm(centre)))
        tok_rows.append(pd.DataFrame({
            "text": i, "tok": t, "cls": cls, "hap": hap, "norm": nrm,
            "cos_centre": cc,
            "dist_centre": np.linalg.norm(v - centre, axis=1),
            "logrank": np.log([ranks.get(x, len(ranks)) for x in t]),
        }))
        p = pair_stats(v, t, cls)
        # all same-token pairs are kept (they are the rare, interesting ones);
        # different-token pairs are subsampled, there are 20k per text
        keep = p["same_token"] | (rng.random(len(p)) < 0.05)
        pair_rows.append(p[keep])
        if sum(len(x) for x in tsne_v) < N_TSNE:
            take = rng.choice(len(v), size=min(40, len(v)), replace=False)
            tsne_v.append(v[take])
            tsne_meta.append(pd.DataFrame({"tok": [t[k] for k in take],
                                           "cls": [cls[k] for k in take],
                                           "hap": [hap[k] for k in take],
                                           "text": i}))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{N_TEXTS}", flush=True)

    toks = pd.concat(tok_rows, ignore_index=True)
    pairs = pd.concat(pair_rows, ignore_index=True)
    pd.set_option("display.width", 250)
    res = os.path.join(BASE, "results")

    # --- per class ---------------------------------------------------------
    by = toks.groupby("cls").agg(
        доля=("norm", lambda x: len(x) / len(toks) * 100),
        норма=("norm", "mean"), сд_нормы=("norm", "std"),
        косинус_с_центром=("cos_centre", "mean"),
        расст_до_центра=("dist_centre", "mean"),
        лог_ранг=("logrank", "mean")).reindex(CLASSES)
    print("\nпо классам токенов\n")
    print(by.round(3).to_string())
    by.round(4).to_csv(os.path.join(res, "token_geometry_classes.csv"))

    # --- same token vs different token, per class --------------------------
    same = pairs[pairs["ca"] == pairs["cb"]]
    st = (same.groupby(["ca", "same_token"])[["cos", "dist"]].mean()
          .unstack("same_token"))
    st.columns = [f"{a}, {'один токен' if b else 'разные'}"
                  for a, b in st.columns]
    st["n одинаковых"] = same[same.same_token].groupby("ca").size()
    st = st.reindex(CLASSES)
    print("\nвнутри класса: два вхождения одного токена против двух разных\n")
    print(st.round(3).to_string())
    st.round(4).to_csv(os.path.join(res, "token_geometry_same.csv"))

    # --- class x class -----------------------------------------------------
    d = pairs[~pairs["same_token"]]
    sym = pd.concat([d, d.rename(columns={"ca": "cb", "cb": "ca"})])
    cosm = sym.groupby(["ca", "cb"])["cos"].mean().unstack().reindex(
        index=CLASSES, columns=CLASSES)
    dism = sym.groupby(["ca", "cb"])["dist"].mean().unstack().reindex(
        index=CLASSES, columns=CLASSES)
    print("\nсредний косинус между классами (разные токены)\n")
    print(cosm.round(3).to_string())
    print("\nсреднее расстояние между классами (разные токены)\n")
    print(dism.round(2).to_string())
    cosm.round(4).to_csv(os.path.join(res, "token_geometry_cos.csv"))
    dism.round(4).to_csv(os.path.join(res, "token_geometry_dist.csv"))

    # --- frequency gradient ------------------------------------------------
    nosink = toks[toks["cls"] != "сток"].copy()
    nosink["дециль ранга"] = pd.qcut(nosink["logrank"], 10, labels=False,
                                      duplicates="drop")
    fr = nosink.groupby("дециль ранга").agg(
        лог_ранг=("logrank", "mean"), норма=("norm", "mean"),
        косинус_с_центром=("cos_centre", "mean"),
        расст_до_центра=("dist_centre", "mean"))
    print("\nнорма по децилям частотного ранга (0 — самые частые)\n")
    print(fr.round(3).to_string())
    fr.round(4).to_csv(os.path.join(res, "token_geometry_freq.csv"))
    # the sinks are the lowest norms in the cloud and they are all frequent
    # tokens, so leaving them in manufactures part of the gradient. Both
    # numbers are reported; the second is the one about vocabulary.
    ns = toks[toks["cls"] != "сток"]
    print("\nкорреляция нормы с лог-рангом: "
          f"r = {np.corrcoef(toks.logrank, toks.norm)[0, 1]:.3f} со стоками, "
          f"r = {np.corrcoef(ns.logrank, ns.norm)[0, 1]:.3f} без них")

    plot(by, st, cosm, fr, toks)
    tsne(np.concatenate(tsne_v)[:N_TSNE],
         pd.concat(tsne_meta, ignore_index=True).iloc[:N_TSNE])


def plot(by, st, cosm, fr, toks):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    a.bar(range(len(by)), by["норма"], yerr=by["сд_нормы"], capsize=4,
          color=[CCOL[c] for c in by.index])
    # the axis starts below the smallest class, not at zero: every lexical
    # class sits within 3 units of 37 and the whole story is in that spread
    a.set_ylim(min(by["норма"] - by["сд_нормы"]) - 2, max(by["норма"]) + 3)
    a.set_xticks(range(len(by)))
    a.set_xticklabels(by.index, rotation=20)
    a.set_ylabel("‖v‖")
    a.set_title("Норма вектора по классу токена")

    a = ax[0, 1]
    im = a.imshow(cosm.values, cmap="viridis")
    a.set_xticks(range(len(cosm)))
    a.set_xticklabels(cosm.columns, rotation=25, ha="right")
    a.set_yticks(range(len(cosm)))
    a.set_yticklabels(cosm.index)
    for i in range(len(cosm)):
        for j in range(len(cosm)):
            a.text(j, i, f"{cosm.values[i, j]:.2f}", ha="center", va="center",
                   color="white", fontsize=9)
    fig.colorbar(im, ax=a, fraction=0.046)
    a.set_title("Средний косинус между классами\n(разные токены, внутри текста)")

    a = ax[1, 0]
    w = 0.38
    xs = np.arange(len(st))
    a.bar(xs - w / 2, st["cos, один токен"], w, label="один токен",
          color="#08306b")
    a.bar(xs + w / 2, st["cos, разные"], w, label="разные токены",
          color="#e6550d")
    a.set_xticks(xs)
    a.set_xticklabels(st.index, rotation=20)
    a.set_ylabel("средний косинус")
    a.set_ylim(0, 1)
    a.legend(fontsize=9)
    a.set_title("Тождество токена против класса")

    a = ax[1, 1]
    a.plot(fr["лог_ранг"], fr["норма"], "o-", color="#333")
    a.set_xlabel("средний лог-ранг в корпусе (вправо — реже)")
    a.set_ylabel("‖v‖")
    ns = toks[toks["cls"] != "сток"]
    r = np.corrcoef(ns.logrank, ns.norm)[0, 1]
    a.set_title(f"Норма против редкости токена, r = {r:.3f} (без стоков)")
    fig.tight_layout()
    out = os.path.join(BASE, "figures", "token_geometry.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"рисунок: {out}")


def tsne(v, meta):
    from sklearn.manifold import TSNE
    print(f"\nt-SNE по {len(v)} токенам...", flush=True)
    xy = TSNE(n_components=2, perplexity=30, init="pca",
              random_state=0).fit_transform(v)
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    a = ax[0]
    for c in CLASSES:
        m = (meta["cls"] == c).values
        a.scatter(xy[m, 0], xy[m, 1], s=7, alpha=0.6, c=CCOL[c], label=c,
                  linewidths=0)
    a.legend(fontsize=9, markerscale=2.5)
    a.set_title("t-SNE скрытых состояний, цвет — класс токена")
    a.set_xticks([])
    a.set_yticks([])

    a = ax[1]
    for h in ("повторяется", "hapax", "сток"):
        m = (meta["hap"] == h).values
        if m.sum():
            a.scatter(xy[m, 0], xy[m, 1], s=7, alpha=0.65, c=HCOL[h], label=h,
                      linewidths=0)
    a.legend(fontsize=9, markerscale=2.5)
    a.set_title("цвет — однократность токена в документе")
    a.set_xticks([])
    a.set_yticks([])

    # the same map coloured by norm: if the classes separate, is it because
    # they sit in different directions or at different radii?
    a = ax[2]
    n = np.linalg.norm(v, axis=1)
    s = a.scatter(xy[:, 0], xy[:, 1], s=7, c=n, cmap="magma", linewidths=0)
    fig.colorbar(s, ax=a, fraction=0.046, label="‖v‖")
    a.set_title("тот же снимок, цвет — норма вектора")
    a.set_xticks([])
    a.set_yticks([])
    fig.tight_layout()
    out = os.path.join(BASE, "figures", "token_tsne.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"рисунок: {out}")
    pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "tok": meta["tok"],
                  "cls": meta["cls"], "hap": meta["hap"], "text": meta["text"],
                  "norm": n}).to_csv(
        os.path.join(BASE, "results", "token_tsne.csv"), index=False)


if __name__ == "__main__":
    main()
