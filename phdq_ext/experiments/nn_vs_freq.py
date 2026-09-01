"""Is a function word loose because of what it is, or because it is rare here?

Some function words sit on long stalks and some never do: `when` 54% of its
occurrences, `the` 0.2%. The tempting reading is grammatical -- clause-linking
words are loose, determiners and pronouns are tight. But the two groups also
differ in how often they occur inside one document, and `of`/`in`/`on` (tight)
against `with`/`by`/`from` (loose) are all prepositions, so grammatical role
cannot be the whole story.

The clean test is the continuous one: distance to the nearest neighbour against
how many times the token occurs, with the count taken *inside the document* or
inside the drawn subsample -- not in the corpus. Corpus rank is the wrong axis
here, since what decides whether a token has a near neighbour is whether another
copy of it is present in this cloud.

Distances are divided by the text's own mean pairwise distance, so that texts of
different spread can be pooled.

If the relation is one clean curve, looseness is multiplicity and nothing else.
If function words scatter around it by grammatical class at matched count, then
there is a second effect and it is about the word.
"""
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
# hand-labelled only to read the scatter afterwards; nothing in the measurement
# depends on these lists
LINK = {"Ġwhen", "Ġwhere", "Ġwhich", "Ġthen", "Ġthan", "Ġif", "Ġbecause",
        "Ġso", "Ġafter", "Ġonly", "Ġeven", "Ġbut", "Ġor", "Ġbefore", "Ġsuch",
        "Ġwhile", "Ġthough", "Ġsince", "Ġuntil", "Ġunless", "Ġhowever"}
DET = {"Ġthe", "Ġa", "Ġan", "Ġthis", "Ġthat", "Ġthese", "Ġthose", "Ġtheir",
       "Ġyour", "Ġher", "Ġhis", "Ġits", "Ġmy", "Ġour", "Ġsome", "Ġany"}


def one(text, emb, ranks, u_sink, L=cfg.L_DEFAULT, seed=0):
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    full = Counter(toks[i] for i in keep)
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v, t = e[idx], [toks[i] for i in idx]
    insamp = Counter(t)
    nrm = np.linalg.norm(v, axis=1)
    sink = ((v @ u_sink) / nrm > COS_CUT) & (nrm < NORM_CUT)
    cls = ["сток" if s_ else token_class(x, ranks) for x, s_ in zip(t, sink)]
    D = squareform(pdist(v))
    scale = D[np.triu_indices(len(t), 1)].mean()
    np.fill_diagonal(D, np.inf)
    return pd.DataFrame({
        "токен": t, "класс": cls,
        "в тексте": [full[x] for x in t],
        "в выборке": [insamp[x] for x in t],
        "до ближайшего": D.min(1) / scale,
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
            r = one(s, emb, ranks, u_sink, seed=1000 * seed + i)
            if r is not None:
                r["текст"] = i
                frames.append(r)
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{N_TEXTS}", flush=True)
    V = pd.concat(frames, ignore_index=True)
    V.to_csv(os.path.join(BASE, "results", "nn_vs_freq.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"\n{len(V)} вершин, {V['текст'].nunique()} текстов. "
          "Расстояния — в долях средней парной дистанции текста\n")

    for col in ("в выборке", "в тексте"):
        b = V[V["класс"] != "сток"].groupby(
            [pd.cut(V[col], [0, 1, 2, 3, 5, 9, 10**6],
                    labels=["1", "2", "3", "4-5", "6-9", "10+"]),
             "класс"], observed=True)["до ближайшего"].mean().unstack()
        print(f"расстояние до ближайшего соседа по числу вхождений «{col}»\n")
        print(b.round(3).to_string())
        print()

    # the question itself: function words only, count held fixed
    S = V[V["класс"] == "служ"].copy()
    S["роль"] = np.where(S["токен"].isin(LINK), "связки",
                         np.where(S["токен"].isin(DET), "детерм./местоим.",
                                  "прочие служебные"))
    g = S[S["роль"] != "прочие служебные"]
    t = g.groupby([pd.cut(g["в выборке"], [0, 1, 2, 3, 5, 9, 10**6],
                          labels=["1", "2", "3", "4-5", "6-9", "10+"]),
                   "роль"], observed=True)["до ближайшего"].agg(["mean", "size"])
    print("служебные: связки против детерминативов "
          "при равном числе вхождений в выборке\n")
    print(t.rename(columns={"mean": "до ближайшего", "size": "вершин"})
          .round(3).to_string())

    # per-token summary, to see the scatter around the curve
    per = (V[V["класс"] == "служ"].groupby("токен")
           .agg(вершин=("до ближайшего", "size"),
                в_выборке=("в выборке", "mean"),
                до_ближайшего=("до ближайшего", "mean")))
    per = per[per["вершин"] >= 120].sort_values("до_ближайшего", ascending=False)
    print(f"\nпо токенам (служебные, не реже 120 вхождений), "
          f"корреляция log(вхождений) с расстоянием: "
          f"{np.corrcoef(np.log(per['в_выборке']), per['до_ближайшего'])[0, 1]:+.3f}\n")
    print(per.head(12).round(3).to_string())
    print("...")
    print(per.tail(8).round(3).to_string())
    per.round(4).to_csv(os.path.join(BASE, "results", "nn_vs_freq_tokens.csv"))
    plot(V, per)


def plot(V, per):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    a = ax[0]
    W = V[V["класс"] != "сток"]
    for c, col in [("служ", "#6baed6"), ("смысл-част", "#e6550d"),
                   ("смысл-редк", "#08306b"), ("пункт", "#9e9ac8"),
                   ("подслово", "#74c476")]:
        g = W[W["класс"] == c]
        m = g.groupby(g["в выборке"].clip(upper=8))["до ближайшего"].mean()
        a.plot(m.index, m.values, "o-", color=col, label=c)
    a.set_xlabel("вхождений токена в выборке (8 = «8 и больше»)")
    a.set_ylabel("до ближайшего соседа / средняя парная")
    a.set_title("Расстояние до ближайшего определяется числом вхождений")
    a.legend(fontsize=9)

    a = ax[1]
    a.scatter(per["в_выборке"], per["до_ближайшего"], s=26, c="#6baed6",
              zorder=3, linewidths=0)
    for tok in per.index:
        r = per.loc[tok]
        if r["до_ближайшего"] > 0.62 or r["в_выборке"] > 4 or tok in LINK | DET:
            a.annotate(tok.lstrip("Ġ▁"), (r["в_выборке"], r["до_ближайшего"]),
                       fontsize=8, xytext=(3, 2), textcoords="offset points",
                       color="#31536b")
    a.set_xscale("log")
    a.set_xlabel("среднее число вхождений в выборке (лог)")
    a.set_ylabel("до ближайшего соседа / средняя парная")
    r = np.corrcoef(np.log(per["в_выборке"]), per["до_ближайшего"])[0, 1]
    a.set_title(f"Служебные токены по отдельности, r = {r:+.2f}")
    fig.tight_layout()
    out = os.path.join(BASE, "figures", "nn_vs_freq.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nрисунок: {out}")


if __name__ == "__main__":
    main()
