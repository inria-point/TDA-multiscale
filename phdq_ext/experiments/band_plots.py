"""Three-band plots, with perturbations coded by what they do to the text.

Two of the three attributes are measured rather than assigned, because a
hand label of "makes the vocabulary richer" is exactly the kind of judgement
that quietly follows the result it is meant to explain:

  лексика   from the paired shift in type-token ratio
  повтор    from the paired shift in the share of repeated word trigrams
  структура assigned by construction -- whether the output is still a
            well-formed text, only its typography was touched, or its grammar
            and discourse were destroyed. This one cannot be measured without
            assuming what we are testing, so it is a label, and the label is
            written from the procedure, not from the coordinates.
"""
import json
import os
import re
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
from descriptions import DESC
from three_bands import BANDS, SUF, generators

BASE = os.path.join(HERE, "..")
WORD = re.compile(r"[a-z']+")

TEXT_FILES = [
    "perturbed_texts_coling.json", "perturbed_texts_colingv2.json",
    "perturbed_texts_loops.json", "perturbed_texts_ngram.json",
    "perturbed_texts_ngramwide.json", "perturbed_texts_expand.json",
    "perturbed_texts_hapax.json", "perturbed_texts_pc2neg.json",
    "perturbed_texts_composed.json", "perturbed_texts_rarify4.json",
    "coling_style_edits.json", "coling_v2_edits.json", "coling_hard_edits.json",
    "coling_pc2_probes.json", "coling_collapse_probes.json",
    "coling_down_more2.json", "coling_syn.json",
]

# broken = grammar or discourse destroyed by construction;
# поверхность = only orthography, punctuation or line breaks;
# цела = still a text a human could have written
BROKEN = {
    "shuffle/words", "shuffle_within_sentences", "syntax/drop_funcwords",
    "vocab/top10", "vocab/top25", "vocab/top60", "vocab/expand100",
    "vocab/expand50", "rarify_types", "rarify_types_50", "commonize_types",
    "hapax_swap_wide", "hapax_swap_wide_50", "hapax_swap_local",
    "splice/every2nd_sent", "splice/all_sents", "ngram/n2", "ngram/n4",
    "ngram_3", "ngram_wide_2", "ngram_wide_3", "marker_p3", "marker_p6",
    "marker_p12", "literary_echo25", "literary_echo50",
}
SURFACE = {
    "surface/lowercase", "surface/numbered_list", "surface/digits_add",
    "surface/punct_add", "surface/punct_strip", "surface/caps_terms",
    "surface/typos", "strip_digits", "syntax/no_sent_bounds",
    "syntax/burst_1to4", "burst_flatten",
}


def structure(name):
    if name.startswith("loop/") or name.startswith("echo_"):
        return "сломана"
    if name in BROKEN:
        return "сломана"
    if name in SURFACE:
        return "поверхность"
    return "цела"


def lexical_stats():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    orig = {f"coling::{r.id}": r.text for r in pool.itertuples()}

    def stats(t):
        w = WORD.findall(t.lower())
        if len(w) < 40:
            return None
        tri = list(zip(w, w[1:], w[2:]))
        c = Counter(tri)
        return (len(set(w)) / len(w),
                sum(v for v in c.values() if v > 1) / len(tri))

    texts = {}
    for fn in TEXT_FILES:
        p = os.path.join(BASE, "results", fn)
        if os.path.exists(p):
            with open(p) as f:
                for k, v in json.load(f).items():
                    texts.setdefault(k, v)
    base, rows = {}, []
    for name, by in texts.items():
        acc = []
        for k, v in list(by.items())[:40]:
            if k not in orig:
                continue
            if k not in base:
                base[k] = stats(orig[k])
            a, b = base[k], stats(v)
            if a and b:
                acc.append((b[0] - a[0], b[1] - a[1]))
        if len(acc) >= 8:
            rows.append({"перт": name, "d_ttr": np.mean([x[0] for x in acc]),
                         "d_rep": np.mean([x[1] for x in acc])})
    return pd.DataFrame(rows).set_index("перт")


def main():
    P = pd.read_csv(os.path.join(BASE, "results", f"bands_perts{SUF}.csv"),
                    index_col=0)
    G = generators()
    L = lexical_stats()
    P = P.join(L)
    P["структура"] = [structure(i) for i in P.index]
    P["лексика"] = pd.cut(P["d_ttr"], [-9, -0.03, 0.03, 9],
                          labels=["беднее", "≈", "богаче"])
    P["повтор"] = pd.cut(P["d_rep"], [-9, -0.03, 0.03, 9],
                         labels=["меньше", "≈", "больше"])
    P.round(3).to_csv(os.path.join(BASE, "results", f"bands_typed{SUF}.csv"))

    schemes = [
        ("структура", {"цела": "#2f855a", "поверхность": "#a0aec0",
                       "сломана": "#c53030"}),
        ("лексика", {"богаче": "#2b6cb0", "≈": "#a0aec0",
                     "беднее": "#b7791f"}),
        ("повтор", {"больше": "#b7791f", "≈": "#a0aec0",
                    "меньше": "#2b6cb0"}),
    ]
    pairs = [("крупный", "мелкий"), ("крупный", "средний"),
             ("мелкий", "средний")]
    fig, axes = plt.subplots(len(schemes), 3, figsize=(17, 14))
    for row, (attr, colours) in enumerate(schemes):
        for col, (a, b) in enumerate(pairs):
            ax = axes[row, col]
            ax.axhline(0, c="k", lw=1)
            ax.axvline(0, c="k", lw=1)
            for val, c in colours.items():
                s = P[P[attr].astype(str) == val]
                if s.empty:
                    continue
                ax.scatter(s[a], s[b], s=42, c=c, alpha=0.85,
                           label=f"{attr}: {val}" if col == 0 else None)
            ax.scatter(G[a], G[b], s=95, marker="^", facecolors="none",
                       edgecolors="black", linewidths=1.3,
                       label="генераторы" if col == 0 else None, zorder=5)
            ax.scatter(0, 0, s=380, marker="*", c="#1a7f37", zorder=6,
                       edgecolors="black", linewidths=0.8)
            ax.set_xlabel(f"{a}, %")
            ax.set_ylabel(f"{b}, %")
            ax.grid(alpha=0.25)
        axes[row, 0].legend(fontsize=8.5, loc="best")
    fig.suptitle("Три полосы: пертурбации по типам, генераторы — "
                 "полые треугольники, человек — звезда")
    fig.tight_layout()
    path = os.path.join(BASE, "figures", f"bands_typed{SUF}.png")
    fig.savefig(path, dpi=140)
    print("saved", path)

    pd.set_option("display.width", 210)
    for attr, _ in schemes:
        print(f"\n=== средние по полосам, {attr}")
        print(P.groupby(attr, observed=True)[list(BANDS)]
              .agg(["mean", "size"]).round(1).to_string())


if __name__ == "__main__":
    main()
