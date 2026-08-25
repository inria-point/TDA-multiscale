"""Do structure-preserving perturbations stay near the origin?

The claim to test: everything that leaves the text well-formed sits close to
the unchanged human baseline, and the far regions of the map -- different as
they are from each other -- are all reached by breaking the text.

The labels are assigned from *what the procedure does*, before looking at any
coordinate, otherwise the test is circular:

  intact   the output is a text a human could have written: every sentence
           grammatical, the discourse coherent. All the LLM rewrites are here,
           however strong the instruction was.
  surface  wording and syntax untouched; only orthography, punctuation or
           where the sentence breaks fall. The word sequence survives.
  broken   grammar or discourse destroyed by construction: order permuted,
           content words substituted, text collapsed into a repeat, sentences
           taken from unrelated documents, sampled from an n-gram table.

Separation is reported as AUC: the probability that a randomly chosen broken
perturbation sits further from the origin than a randomly chosen intact one.
0.5 is no separation, 1.0 is perfect.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import taxonomy as T

BASE = os.path.join(os.path.dirname(__file__), "..")

STRUCTURE = {
    "intact": [
        "enrich/lexdiv_up", "enrich/topics_up_hard",
        "deplete/lexdiv_down", "deplete/ideas_down",
        "rarity/common_words", "rarity/rare_words",
        "style/simplified", "style/dialogue_interj", "style/telegraphic",
        "style/scientific_terms", "style/bulletin_abbrev", "style/news_dates",
        "style/literary",
    ],
    "surface": [
        "surface/lowercase", "surface/numbered_list", "surface/digits_add",
        "surface/punct_add", "surface/punct_strip", "surface/caps_terms",
        "surface/typos", "syntax/no_sent_bounds", "syntax/burst_1to4",
    ],
    "broken": [
        "shuffle/words", "syntax/drop_funcwords",
        "vocab/top10", "vocab/top25", "vocab/top60",
        "loop/echo6w_x2", "loop/echo6w_x4", "loop/phrase12w_x4",
        "loop/tail_keep20_unit3w", "loop/tail_keep40_unit3w",
        "loop/tail_keep40_unit3w_drift", "loop/tail_keep60_unit1w",
        "loop/tail_keep60_unit3w",
        "splice/every2nd_sent", "splice/all_sents",
        "ngram/n2", "ngram/n4",
    ],
}


def auc(pos, neg):
    """P(a random pos exceeds a random neg), ties counted as half."""
    wins = sum((a > b) + 0.5 * (a == b)
               for a, b in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


def main():
    P = pd.read_csv(os.path.join(BASE, "results", "pert_map.csv"), index_col=0)
    lab = {n: k for k, v in STRUCTURE.items() for n in v}
    missing = set(P.index) - set(lab)
    if missing:
        raise SystemExit(f"не размечены: {sorted(missing)}")
    P["structure"] = [lab[n] for n in P.index]
    P["dist"] = np.linalg.norm(P[["PC1", "PC2", "PC3"]].values, axis=1)

    pd.set_option("display.width", 200)
    print("расстояние от неизменённого человеческого текста, по классам\n")
    print(P.groupby("structure")["dist"]
          .agg(["count", "min", "median", "max"]).round(0)
          .loc[["intact", "surface", "broken"]].to_string())

    d = {k: P.loc[P["structure"] == k, "dist"].values for k in STRUCTURE}
    print(f"\nAUC broken против intact:  {auc(d['broken'], d['intact']):.2f}")
    print(f"AUC broken против surface: {auc(d['broken'], d['surface']):.2f}")

    print("\nчто ломает картину — сохранившие структуру дальше 100:")
    far = P[(P["structure"] != "broken") & (P["dist"] > 100)]
    print(far[["PC1", "PC2", "PC3", "dist", "structure"]]
          .sort_values("dist", ascending=False).round(0).to_string())
    print("\nсломанные ближе 100:")
    near = P[(P["structure"] == "broken") & (P["dist"] < 100)]
    print(near[["PC1", "PC2", "PC3", "dist"]]
          .sort_values("dist").round(0).to_string())

    # ranks, not a threshold: picking a cut-off that makes the table look
    # clean would be choosing the answer. Sorted by distance, the question is
    # simply how long each end of the list stays pure.
    order = P.sort_values("dist")["structure"].values
    pure_near = next(i for i, k in enumerate(order) if k == "broken")
    pure_far = next(i for i, k in enumerate(order[::-1]) if k != "broken")
    mid = order[pure_near:len(order) - pure_far]
    print(f"\nпо возрастанию расстояния: первые {pure_near} из {len(order)} "
          f"— все сохраняют структуру;\nпоследние {pure_far} — все ломают; "
          f"в середине {len(mid)} вперемешку "
          f"({(mid != 'broken').sum()} сохраняющих, "
          f"{(mid == 'broken').sum()} ломающих)")

    # the two classes abut rather than separate: 95.7 against 96.8. The single
    # perturbation at the seam is also the one whose label is arguable --
    # dropping function words is what a telegraphic register does anyway, and
    # style/telegraphic is labelled intact. Relabelling it moves the seam to a
    # real gap, 96 to 136.
    print(f"\nшов между классами: surface максимум "
          f"{d['surface'].max():.1f}, broken минимум {d['broken'].min():.1f}")

    # the claim is directional: the two halves of PC1 are not symmetric
    print("\nто же отдельно по направлению вдоль PC1\n")
    for side, sel in [("d растёт (PC1 > 0)", P["PC1"] > 0),
                      ("d падает (PC1 < 0)", P["PC1"] < 0)]:
        s = P[sel]
        g = s.groupby("structure")["dist"].agg(["count", "max"])
        print(f"{side}:")
        print(g.round(0).to_string(), "\n")

    P.round(1).to_csv(os.path.join(BASE, "results", "structure_test.csv"))


if __name__ == "__main__":
    main()
