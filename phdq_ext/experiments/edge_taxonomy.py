"""What kind of token pair sits at each scale of the MST?

d_hat(q) is read off the MST edge lengths after trimming a q-fraction from one
end, so every band is a statement about a particular slice of the edge-length
distribution and about nothing else. This asks what that slice is made of, the
whole way along: the MST edges of a text are sorted by length, cut into twenty
equal bins, and every edge is assigned to exactly one cell of a token-pair
taxonomy. The bins sum to 100% by construction, so a cell that grows somewhere
is a cell that shrinks elsewhere and nothing hides in a remainder.

The taxonomy refines the six cells of short_edge_types.py in three ways that
matter for reading the bands:

  frequency is split off from content-hood. A content word that the corpus uses
  constantly and one it uses once are different objects geometrically, and the
  coarse band is already known to answer to rare tokens.

  punctuation is split off from function words. Both were "служ." before, but
  punctuation is not a word at all: it has no lexical content to be near.

  pieces of one word instance are pulled out first, before anything else is
  asked about the pair. Such an edge says nothing about the text -- only that
  the tokeniser cut a word the encoder then kept together -- so leaving it
  inside "два подслова" would let a tokenisation artefact masquerade as a
  lexical fact.

  the attention sinks get a cell of their own. They are 2.4% of tokens and a
  near-duplicate point mass (sink_cluster.py), so they land almost entirely in
  the shortest bin and would otherwise be read as "punctuation is close to the
  definite article", which is a fact about ModernBERT and not about English.

Alongside the composition, the geometry of the edge itself. Distances here are
Euclidean on unnormalised hidden states, so

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2||a|| ||b|| cos(a, b)

and an edge can be short for two quite different reasons: the two vectors point
the same way, or they are both short. Per bin we report the mean cosine and the
mean norm of the endpoints, which separates the two.
"""
import os
import re
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from embedder import Embedder
from sink_cluster import COS_CUT, NORM_CUT, sink_direction

BASE = os.path.join(HERE, "..")
SKIP = {"[CLS]", "[SEP]", "[PAD]"}
HAS_ALPHA = re.compile(r"[A-Za-z]")
ALL_ALPHA = re.compile(r"^[A-Za-z]+$")

N_TEXTS = int(os.environ.get("N_TEXTS", 120))
SEEDS = int(os.environ.get("SEEDS", 3))
N_BINS = 20
# a content word inside the corpus's thousand commonest tokens is "частое".
# The cut is deliberately generous: the point is to separate the everyday
# vocabulary of the genre from the words a text uses once, not to draw a
# principled line in the middle of a Zipf curve.
RANK_FREQ = 1000

# Function words proper: determiners, prepositions, pronouns, auxiliaries,
# conjunctions, degree words. perturb.FUNCTION_WORDS has 31 entries and was
# built for a perturbation that needed a safe subset; a composition table needs
# the class to be complete or the leftovers land in "смысловые" and inflate it.
FUNCTION = set("""
a an the this that these those such
i me my mine myself you your yours we us our ours they them their theirs
he him his she her hers it its one ones oneself
am is are was were be been being do does did done doing have has had having
can could shall should will would may might must ought need dare
and or but nor for yet so because although though while whereas since unless
until if whether than as when where why how what which who whom whose
in on at by to from of with without within into onto upon about above below
under over between among across through during before after against toward
towards behind beside beyond near off out up down along around per via
not no nor never nothing none neither either both all any some each every
few many much more most less least several other others another same
very too also just only even still already yet again ever always often
there here then now thus hence therefore however moreover furthermore
s t d ll m re ve
""".split())

CELLS = [
    "сток-сток",
    "сток-прочее",
    "куски одного слова",
    "= пунктуация",
    "= служебное",
    "= смысловое частое",
    "= смысловое редкое",
    "= подслово",
    "≠ пунктуация",
    "≠ служебные",
    "≠ смысловые",
    "≠ подслова",
    "≠ смешанные",
]
# drawing order: same-token cells first, then different-token, so the stack
# reads as one block sliding into the other as the scale grows
COLORS = {
    "сток-сток": "#000000",
    "сток-прочее": "#636363",
    "куски одного слова": "#8c6d31",
    "= пунктуация": "#9e9ac8",
    "= служебное": "#6baed6",
    "= смысловое частое": "#2171b5",
    "= смысловое редкое": "#08306b",
    "= подслово": "#74c476",
    "≠ пунктуация": "#fdd0a2",
    "≠ служебные": "#fdae6b",
    "≠ смысловые": "#e6550d",
    "≠ подслова": "#a1d99b",
    "≠ смешанные": "#bdbdbd",
}


def token_class(tok, ranks):
    """пункт / подслово / служ / смысл-част / смысл-редк.

    The tokeniser marks a word opening with Ġ, so a token without it continues
    the previous word. A token with no letter at all is punctuation, a digit or
    a symbol -- it is not a word and cannot be near one lexically.
    """
    w = tok.lstrip("Ġ▁")
    if not HAS_ALPHA.search(w):
        return "пункт"
    if not tok.startswith(("Ġ", "▁")):
        return "подслово"
    if not ALL_ALPHA.match(w):
        return "смысл-редк"  # alphanumeric mixes: codes, "3rd", "covid19"
    if w.lower() in FUNCTION:
        return "служ"
    return "смысл-част" if ranks.get(tok, 10**9) <= RANK_FREQ else "смысл-редк"


SAME = {"пункт": "= пунктуация", "служ": "= служебное",
        "смысл-част": "= смысловое частое", "смысл-редк": "= смысловое редкое",
        "подслово": "= подслово"}
DIFF = {"пункт": "≠ пунктуация", "служ": "≠ служебные",
        "подслово": "≠ подслова"}


def cell_of(ta, tb, ca, cb, same_word):
    if ca == "сток" or cb == "сток":
        return "сток-сток" if ca == cb else "сток-прочее"
    if same_word:
        return "куски одного слова"
    if ta == tb:
        return SAME[ca]
    if ca == cb:
        return DIFF.get(ca, "≠ смысловые")
    if {ca, cb} == {"смысл-част", "смысл-редк"}:
        return "≠ смысловые"
    return "≠ смешанные"


def one_text(text, emb, ranks, u_sink, L=cfg.L_DEFAULT, seed=0):
    """MST over L sampled tokens; per-edge cell, length, cosine, norms.

    Sampling matches the estimator: L tokens drawn without replacement from the
    text's own tokens, so the edge-length distribution here is the one d_hat is
    computed from.
    """
    e, toks = emb.embed(text, return_tokens=True)
    keep = [i for i, t in enumerate(toks) if t not in SKIP]
    if len(keep) < L:
        return None
    # a token that does not open with the space marker continues the previous
    # word, so a running count of openers labels every token with its word
    wid = np.cumsum([t.startswith(("Ġ", "▁")) for t in toks])
    rng = np.random.default_rng(seed)
    idx = np.array(keep)[rng.choice(len(keep), size=L, replace=False)]
    v = e[idx]
    t = [toks[i] for i in idx]
    w = wid[idx]
    nrm0 = np.linalg.norm(v, axis=1)
    sink = ((v @ u_sink) / nrm0 > COS_CUT) & (nrm0 < NORM_CUT)
    cls = ["сток" if s_ else token_class(x, ranks) for x, s_ in zip(t, sink)]

    d = np.linalg.norm(v[:, None] - v[None], axis=-1)
    m = minimum_spanning_tree(d).tocoo()
    o = np.argsort(m.data)
    lens, rows, cols = m.data[o], m.row[o], m.col[o]

    nrm = np.linalg.norm(v, axis=1)
    a, b = v[rows], v[cols]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    cells = [cell_of(t[i], t[j], cls[i], cls[j], w[i] == w[j])
             for i, j in zip(rows, cols)]
    return pd.DataFrame({
        "cell": cells,
        "len": lens,
        "rel_len": lens / lens.mean(),
        "cos": cos,
        "norm_mean": (nrm[rows] + nrm[cols]) / 2,
        "norm_gap": np.abs(nrm[rows] - nrm[cols]),
        "gap": np.abs(idx[rows] - idx[cols]),
        "bin": np.minimum((np.arange(len(lens)) * N_BINS) // len(lens),
                          N_BINS - 1),
    })


def corpus_ranks(texts, emb):
    """Frequency rank of every tokeniser token over the human corpus."""
    c = Counter()
    for s in texts:
        c.update(emb.tokenizer.tokenize(s))
    return {tok: i + 1 for i, (tok, _) in enumerate(c.most_common())}


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    print(f"частотный словарь по {len(hum)} человеческим текстам", flush=True)
    ranks = corpus_ranks(hum, emb)
    # estimated on texts outside the measured set, so the cell is not defined
    # by the very edges it is then counted in
    u_sink, n_sink = sink_direction(hum[300:340], emb)
    print(f"направление стока по {n_sink} токенам 40 других текстов",
          flush=True)

    frames = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for seed in range(SEEDS):
            r = one_text(s, emb, ranks, u_sink, seed=1000 * seed + i)
            if r is not None:
                r["text"] = i
                frames.append(r)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{N_TEXTS}", flush=True)
    edges = pd.concat(frames, ignore_index=True)
    edges["cell"] = pd.Categorical(edges["cell"], CELLS)
    print(f"\n{len(edges)} рёбер, {edges.text.nunique()} текстов, "
          f"{SEEDS} зерна на текст\n")

    # composition per bin: share within the bin, averaged over MSTs so that
    # every text weighs the same regardless of how many edges it contributed
    per = (edges.groupby(["text", "bin", "cell"], observed=False).size()
           .rename("n").reset_index())
    per["share"] = per["n"] / per.groupby(["text", "bin"])["n"].transform("sum")
    comp = (per.groupby(["bin", "cell"], observed=False)["share"].mean()
            .unstack() * 100)
    geom = edges.groupby("bin")[["len", "rel_len", "cos", "norm_mean",
                                 "norm_gap", "gap"]].mean()

    pd.set_option("display.width", 250)
    print("состав по 5%-бинам длины ребра, % рёбер бина\n")
    print(comp.round(1).to_string())
    print("\nгеометрия ребра по бинам\n")
    print(geom.round(3).to_string())

    # aggregation over the cuts the bands actually use
    cuts = {"10% самых коротких": (0, 2), "20% самых коротких": (0, 4),
            "30% самых коротких": (0, 6), "50% самых коротких": (0, 10),
            "50% самых длинных": (10, 20), "30% самых длинных": (14, 20),
            "20% самых длинных": (16, 20), "10% самых длинных": (18, 20)}
    agg = {}
    for name, (lo, hi) in cuts.items():
        sub = edges[(edges["bin"] >= lo) & (edges["bin"] < hi)]
        s = sub.groupby("cell", observed=False).size()
        agg[name] = s / s.sum() * 100
    agg = pd.DataFrame(agg).T
    print("\nагрегация по срезам\n")
    print(agg.round(1).to_string())

    res = os.path.join(BASE, "results")
    comp.round(3).to_csv(os.path.join(res, "edge_taxonomy_bins.csv"))
    geom.round(4).to_csv(os.path.join(res, "edge_taxonomy_geometry.csv"))
    agg.round(3).to_csv(os.path.join(res, "edge_taxonomy_cuts.csv"))

    # per-cell geometry: how long is an edge of each kind, and why
    bycell = edges.groupby("cell", observed=False).agg(
        доля=("len", lambda x: len(x) / len(edges) * 100),
        отн_длина=("rel_len", "mean"),
        косинус=("cos", "mean"),
        норма=("norm_mean", "mean"),
        разн_норм=("norm_gap", "mean"),
        разрыв=("gap", "mean"))
    print("\nпо типам рёбер (отн_длина — к средней длине ребра текста)\n")
    print(bycell.round(3).to_string())
    bycell.round(4).to_csv(os.path.join(res, "edge_taxonomy_by_cell.csv"))
    plot(comp, geom, agg)


def plot(comp, geom, agg):
    fig, axes = plt.subplots(3, 1, figsize=(11, 13),
                             gridspec_kw={"height_ratios": [2.2, 1, 1]})
    x = (comp.index.values + 0.5) * (100 / N_BINS)
    ax = axes[0]
    ax.stackplot(x, [comp[c].values for c in CELLS], labels=CELLS,
                 colors=[COLORS[c] for c in CELLS], edgecolor="white",
                 linewidth=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("перцентиль длины ребра MST")
    ax.set_ylabel("доля рёбер бина, %")
    ax.set_title("Из чего сложено ребро на каждом масштабе")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    # where each band actually looks, from band_composition.weights(): the
    # bar spans the percentiles the band retains, the tick is its centre of
    # mass. Drawn here because every reading of the stack above is a reading
    # of one of these three windows.
    spans = [("мелкая", 0, 40, 14, "#08306b"), ("средняя", 30, 100, 65, "#7f7f7f"),
             ("крупная", 50, 100, 82, "#a63603")]
    for k, (name, lo, hi, com, col) in enumerate(spans):
        y = 104 + k * 5
        ax.plot([lo, hi], [y, y], color=col, lw=5, solid_capstyle="butt",
                clip_on=False)
        ax.plot([com], [y], marker="|", color="white", ms=9, mew=2,
                clip_on=False)
        ax.text(hi + 1.5, y, f"{name} полоса", va="center", fontsize=9,
                color=col, clip_on=False)
    ax.set_ylim(0, 100)

    ax = axes[1]
    ax.plot(x, geom["rel_len"], color="#333", lw=2)
    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_ylabel("длина ребра / средняя")
    ax.set_xlabel("перцентиль длины ребра MST")
    ax.axhline(1.0, color="#999", lw=0.8, ls=":")
    ax.set_title("Во что перцентиль переводится по длине")

    ax = axes[2]
    ax.plot(x, geom["cos"], color="#08306b", lw=2, label="косинус концов")
    ax.set_xlim(0, 100)
    ax.set_ylabel("косинус")
    ax.set_xlabel("перцентиль длины ребра MST")
    ax2 = ax.twinx()
    ax2.plot(x, geom["norm_mean"], color="#e6550d", lw=2,
             label="средняя норма концов")
    ax2.set_ylabel("норма вектора")
    ln = ax.get_lines() + ax2.get_lines()
    ax.legend(ln, [l.get_label() for l in ln], fontsize=9, loc="center right")
    ax.set_title("Ребро коротко от угла или от малой нормы?")

    fig.tight_layout()
    out = os.path.join(BASE, "figures", "edge_taxonomy.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nрисунок: {out}")


if __name__ == "__main__":
    main()
