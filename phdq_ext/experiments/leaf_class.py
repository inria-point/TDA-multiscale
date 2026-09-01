"""Is "a leaf on a long stalk" a class of vertex we can name in advance?

Two things this settles, both raised by the same objection: the taxonomy splits
content words by corpus frequency and the two halves then behave identically --
norm 36.5 against 36.3, mutual distance 29.8 against 30.0. A split that separates
nothing is not a split, so either frequency does not matter or it was measured
on the wrong axis.

**One.** The reported gradient r = -0.70 between norm and corpus log-rank is
suspect for exactly that reason. The frequent deciles are punctuation (norm
40.0) and function words (38.7) and the rare ones are content words (36.3), so
the correlation may be nothing but "function words are longer vectors", wearing
frequency as a disguise. Measured again inside each class, where the class
cannot do the work.

**Two.** The long end of the tree is 60% single points on long stalks
(mst_picture.py). Those points are a candidate vertex class, and the useful
question is whether they can be picked out by a property of the *token* rather
than of the tree -- a class you can name before building the MST is a class you
can use; one defined by stalk length only restates the edge length.

Three candidate properties, tested as predictors of being such a leaf:

  hapax в тексте     the token occurs once in this document
  редкий в корпусе   corpus rank beyond the cut
  редкий в выборке   occurs once among the L sampled tokens

The three are not the same thing and the difference is the point: a word can be
common in English and used once here, or rare in English and repeated here.
"""
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import COS_CUT, NORM_CUT, SKIP, corpus_ranks, token_class
from embedder import Embedder
from sink_cluster import sink_direction

BASE = os.path.join(HERE, "..")
N_TEXTS = int(os.environ.get("N_TEXTS", 120))
SEEDS = int(os.environ.get("SEEDS", 3))
# a stalk in the top fifth of the text's own edge lengths -- the same cut the
# coarse band's window starts near, so "long" means the same thing here as there
STALK_PCT = 80
CLASSES = ["пункт", "служ", "смысл-част", "смысл-редк", "подслово", "сток"]


def vertices(text, emb, ranks, u_sink, L=cfg.L_DEFAULT, seed=0):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    full = Counter(toks[i] for i in keep)          # counts over the whole text
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v = e[idx]
    t = [toks[i] for i in idx]
    insample = Counter(t)
    nrm = np.linalg.norm(v, axis=1)
    sink = ((v @ u_sink) / nrm > COS_CUT) & (nrm < NORM_CUT)
    cls = ["сток" if s_ else token_class(x, ranks) for x, s_ in zip(t, sink)]

    d = squareform(pdist(v))
    m = minimum_spanning_tree(d).tocoo()
    n = len(t)
    deg = np.bincount(np.concatenate([m.row, m.col]), minlength=n)
    # every vertex's longest incident edge; for a leaf this is its only edge
    longest = np.zeros(n)
    for a, b, w in zip(m.row, m.col, m.data):
        longest[a] = max(longest[a], w)
        longest[b] = max(longest[b], w)
    cut = np.percentile(m.data, STALK_PCT)
    return pd.DataFrame({
        "токен": t, "класс": cls, "норма": nrm,
        "лог_ранг": np.log([ranks.get(x, len(ranks)) for x in t]),
        "степень": deg,
        "черешок": longest,
        "отн_черешок": longest / m.data.mean(),
        "лист": deg == 1,
        "длинный лист": (deg == 1) & (longest >= cut),
        "hapax в тексте": [full[x] == 1 for x in t],
        "один в выборке": [insample[x] == 1 for x in t],
    })


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    ranks = corpus_ranks(hum, emb)
    u_sink, _ = sink_direction(hum[300:340], emb)

    frames = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for seed in range(SEEDS):
            r = vertices(s, emb, ranks, u_sink, seed=1000 * seed + i)
            if r is not None:
                r["текст"] = i
                frames.append(r)
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{N_TEXTS}", flush=True)
    V = pd.concat(frames, ignore_index=True)
    V.to_csv(os.path.join(BASE, "results", "leaf_class.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"\n{len(V)} вершин, {V['текст'].nunique()} текстов\n")

    # --- 1. is the frequency gradient in the norm real, or a class effect? --
    print("=" * 70)
    print("норма против корпусного лог-ранга, внутри класса\n")
    rows = []
    for c in CLASSES:
        g = V[V["класс"] == c]
        if len(g) < 100 or g["лог_ранг"].std() < 1e-6:
            continue
        rows.append({"класс": c, "n": len(g), "норма": g["норма"].mean(),
                     "разброс лог-ранга": g["лог_ранг"].std(),
                     "r(норма, лог-ранг)": np.corrcoef(g["лог_ранг"],
                                                       g["норма"])[0, 1]})
    R = pd.DataFrame(rows).set_index("класс")
    print(R.round(3).to_string())
    ns = V[V["класс"] != "сток"]
    print(f"\nпо всем классам сразу:      r = "
          f"{np.corrcoef(ns['лог_ранг'], ns['норма'])[0, 1]:+.3f}")
    print("средний r внутри класса:    r = "
          f"{R['r(норма, лог-ранг)'].mean():+.3f}")
    # splitting content words at rank 1000 truncates their log-rank range to
    # ~0.7, and a truncated range attenuates a correlation by construction. The
    # honest test pools the two tiers back together, where the range is full.
    for name, mask in [("смысловые (обе половины вместе)",
                        V["класс"].isin(["смысл-част", "смысл-редк"])),
                       ("служебные + пунктуация",
                        V["класс"].isin(["служ", "пункт"]))]:
        g = V[mask]
        print(f"внутри «{name}»: разброс лог-ранга "
              f"{g['лог_ранг'].std():.2f}, r = "
              f"{np.corrcoef(g['лог_ранг'], g['норма'])[0, 1]:+.3f}")

    # --- 2. what is a long-stalk leaf made of? ------------------------------
    print("\n" + "=" * 70)
    print("класс вершин «лист на длинном черешке»\n")
    L = V["длинный лист"]
    print(f"доля вершин: {L.mean() * 100:.1f}%   "
          f"(листьев вообще {V['лист'].mean() * 100:.1f}%)")
    comp = pd.DataFrame({
        "все вершины, %": V["класс"].value_counts(normalize=True) * 100,
        "длинные листья, %": V[L]["класс"].value_counts(normalize=True) * 100,
    }).reindex(CLASSES)
    comp["обогащение"] = comp["длинные листья, %"] / comp["все вершины, %"]
    print("\nсостав по классу токена\n")
    print(comp.round(2).to_string())

    print("\nсвойства вершины: длинные листья против остальных\n")
    cmpt = pd.DataFrame({
        "длинный лист": V[L][["норма", "лог_ранг", "отн_черешок",
                              "hapax в тексте", "один в выборке"]].mean(),
        "прочие": V[~L][["норма", "лог_ранг", "отн_черешок",
                         "hapax в тексте", "один в выборке"]].mean(),
    })
    print(cmpt.round(3).to_string())

    # --- 3. which token property predicts it -------------------------------
    print("\n" + "=" * 70)
    print("чем такой лист опознаётся заранее, по свойству токена\n")
    W = V[~V["класс"].isin(["сток"])]
    rows = []
    for name, mask in [
            ("hapax в тексте", W["hapax в тексте"]),
            ("один раз в выборке", W["один в выборке"]),
            ("редкий в корпусе (ранг > 1000)", W["лог_ранг"] > np.log(1000)),
            ("смысловое (любое)", W["класс"].isin(["смысл-част",
                                                   "смысл-редк"])),
            ("смысловое редкое", W["класс"] == "смысл-редк")]:
        p1 = W[mask]["длинный лист"].mean() * 100
        p0 = W[~mask]["длинный лист"].mean() * 100
        rows.append({"признак": name, "доля вершин, %": mask.mean() * 100,
                     "P(длинный лист | да), %": p1,
                     "P(длинный лист | нет), %": p0,
                     "отношение": p1 / max(p0, 1e-9)})
    P = pd.DataFrame(rows).set_index("признак")
    print(P.round(2).to_string())

    # hapax and corpus-rarity overlap; each has to be tested with the other held
    # fixed or the stronger one simply borrows the weaker one's association
    print("\nперекрёст: hapax в тексте × редкий в корпусе, "
          "P(длинный лист), %\n")
    W = W.copy()
    W["редкий в корпусе"] = W["лог_ранг"] > np.log(1000)
    x = (W.groupby(["hapax в тексте", "редкий в корпусе"])["длинный лист"]
         .agg(["mean", "size"]))
    x["mean"] *= 100
    print(x.rename(columns={"mean": "P(длинный лист), %",
                            "size": "вершин"}).round(2).to_string())

    plot(V, comp, P)


def plot(V, comp, P):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    a = ax[0]
    for c, col in [("смысл-редк", "#08306b"), ("смысл-част", "#e6550d"),
                   ("служ", "#6baed6"), ("пункт", "#9e9ac8")]:
        g = V[V["класс"] == c]
        a.hist(g["отн_черешок"].clip(upper=2.2), bins=40, histtype="step",
               lw=1.8, color=col, label=c, density=True)
    a.set_xlabel("длина черешка / средняя длина ребра")
    a.set_ylabel("плотность")
    a.set_title("Самое длинное ребро вершины, по классу")
    a.legend(fontsize=9)

    a = ax[1]
    W = V[V["класс"] != "сток"]
    for name, mask, col in [
            ("hapax в тексте", W["hapax в тексте"], "#a63603"),
            ("повторяется", ~W["hapax в тексте"], "#6baed6")]:
        a.hist(W[mask]["отн_черешок"].clip(upper=2.2), bins=40,
               histtype="step", lw=2, color=col, label=name, density=True)
    a.set_xlabel("длина черешка / средняя длина ребра")
    a.set_title("То же, но по однократности в документе")
    a.legend(fontsize=9)

    a = ax[2]
    y = P["P(длинный лист | да), %"]
    a.barh(range(len(y)), y, color="#a63603")
    for i, (lab, val) in enumerate(zip(P.index, y)):
        a.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=9)
    a.set_yticks(range(len(y)))
    a.set_yticklabels(P.index, fontsize=9)
    a.axvline(V["длинный лист"].mean() * 100, color="#333", ls=":",
              label="базовый уровень")
    a.set_xlabel("P(вершина — длинный лист), %")
    a.set_title("Чем такой лист опознаётся заранее")
    a.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(BASE, "figures", "leaf_class.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nрисунок: {out}")


if __name__ == "__main__":
    main()
